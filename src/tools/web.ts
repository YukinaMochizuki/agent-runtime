import { isIP } from "node:net";
import { lookup } from "node:dns/promises";
import { z } from "zod";
import { err, ok, type ToolBundle, type ToolDef } from "../tool.ts";

/**
 * Web tools: outbound network access, shared by every host (`agent-runtime/tools/web`). They are
 * stateless external calls, so they belong to no host; a host passes in its configuration and decides
 * which runs get them. Three tools, each present only when its dependency is configured (an
 * unconfigured tool is simply not in the bundle, so an agent never sees a dead tool):
 *   - web_search  → Tavily (`tavilyApiKey`)
 *   - web_scrape  → a Crawl4AI instance, one page → markdown (`crawl4ai`)
 *   - web_fetch   → a plain GET returning text (always present)
 *
 * This module reads no environment and keeps no state. Whether a Crawl4AI instance is up is the
 * host's question (it may be the one starting and stopping it): pass `crawl4ai` only while it is
 * reachable, and use `probeCrawl4ai` to find out.
 *
 * Every agent-supplied URL (web_scrape target, web_fetch target) is run through `assertSafeUrl`, which
 * resolves the host and refuses loopback / private / link-local / cloud-metadata addresses (SSRF
 * guard). Every request is bounded by a timeout, and every response body handed back to the model is
 * length-capped so a large page cannot blow out the context.
 */

const FETCH_TIMEOUT_MS = 30_000;
const PROBE_TIMEOUT_MS = 2_000;
const MAX_RETURN_CHARS = 60_000;
const TAVILY_ENDPOINT = "https://api.tavily.com/search";

export interface WebToolsConfig {
  /** Enables `web_search`. */
  tavilyApiKey?: string;
  /** Enables `web_scrape`. Pass it only while the instance is reachable (see `probeCrawl4ai`). */
  crawl4ai?: { url: string; token?: string };
  /** The bundle's server name (the `mcp__<name>__…` prefix on Claude). Default `web`. */
  serverName?: string;
}

function capText(text: string): string {
  if (text.length <= MAX_RETURN_CHARS) return text;
  return `${text.slice(0, MAX_RETURN_CHARS)}\n… (truncated, showing ${MAX_RETURN_CHARS} of ${text.length} chars)`;
}

async function fetchWithTimeout(url: string, init: RequestInit, ms = FETCH_TIMEOUT_MS): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

/** Is a Crawl4AI instance answering its /health? Never throws. */
export async function probeCrawl4ai(url: string, token?: string, timeoutMs = PROBE_TIMEOUT_MS): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${url.replace(/\/$/, "")}/health`, { method: "GET", headers: token ? { authorization: `Bearer ${token}` } : {} }, timeoutMs);
    return res.ok;
  } catch {
    return false;
  }
}

/** Block a single resolved IP literal: loopback, private, link-local (incl. cloud metadata), unspecified. */
function isBlockedIp(ip: string): boolean {
  const v = isIP(ip);
  if (v === 4) {
    const p = ip.split(".").map(Number);
    if (p.length !== 4 || p.some((n) => Number.isNaN(n))) return true; // malformed → refuse
    const [a, b] = p as [number, number, number, number];
    if (a === 0) return true; // 0.0.0.0/8 (incl. unspecified)
    if (a === 127) return true; // loopback
    if (a === 10) return true; // private
    if (a === 172 && b >= 16 && b <= 31) return true; // private
    if (a === 192 && b === 168) return true; // private
    if (a === 169 && b === 254) return true; // link-local incl. 169.254.169.254 metadata
    if (a === 100 && b >= 64 && b <= 127) return true; // CGNAT 100.64/10
    return false;
  }
  if (v === 6) {
    const lower = ip.toLowerCase();
    if (lower === "::1" || lower === "::") return true; // loopback / unspecified
    // IPv4-mapped (::ffff:a.b.c.d) → judge the embedded v4 address.
    const mapped = lower.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/);
    if (mapped) return isBlockedIp(mapped[1]!);
    if (lower.startsWith("fe80")) return true; // link-local
    const head = parseInt(lower.split(":")[0] || "0", 16);
    if ((head & 0xfe00) === 0xfc00) return true; // fc00::/7 unique-local
    return false;
  }
  return true; // not a recognisable IP → refuse
}

/**
 * Validate an agent-supplied URL before it is fetched (by us, or by Crawl4AI on our behalf). Rejects
 * non-http(s) schemes and any host that resolves to a blocked address. It resolves at check time; it
 * does not defend against DNS rebinding between check and fetch (a documented limitation).
 */
async function assertSafeUrl(raw: string): Promise<void> {
  let u: URL;
  try {
    u = new URL(raw);
  } catch {
    throw new Error(`invalid URL: ${raw}`);
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") {
    throw new Error(`unsupported scheme "${u.protocol}" (only http/https)`);
  }
  const host = u.hostname;
  if (isIP(host)) {
    if (isBlockedIp(host)) throw new Error(`refusing to fetch a private/loopback address: ${host}`);
    return;
  }
  if (host.toLowerCase() === "localhost") throw new Error("refusing to fetch localhost");
  let addrs: { address: string }[];
  try {
    addrs = await lookup(host, { all: true });
  } catch {
    throw new Error(`could not resolve host: ${host}`);
  }
  const bad = addrs.find((a) => isBlockedIp(a.address));
  if (bad) throw new Error(`host ${host} resolves to a private/loopback address (${bad.address}); refusing`);
}

interface TavilyResult {
  title?: string;
  url?: string;
  content?: string;
}

function tavilySearch(apiKey: string): ToolDef {
  return {
    name: "web_search",
    description:
      "Search the web (Tavily). Returns a ranked list of results as title / url / snippet. Use this to FIND pages (docs, API references, error explanations, prior art); use web_scrape to read a specific page in full.",
    inputSchema: {
      query: z.string().describe("the search query"),
      max_results: z.number().int().positive().max(20).optional().describe("how many results (default 5)"),
      topic: z.enum(["general", "news"]).optional().describe("search topic (default general)"),
    },
    handler: async (args: { query: string; max_results?: number; topic?: "general" | "news" }) => {
      try {
        const res = await fetchWithTimeout(TAVILY_ENDPOINT, {
          method: "POST",
          headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
          body: JSON.stringify({
            query: args.query,
            max_results: args.max_results ?? 5,
            topic: args.topic ?? "general",
            search_depth: "basic",
          }),
        });
        if (!res.ok) return err(`tavily search failed: HTTP ${res.status} ${(await res.text()).slice(0, 500)}`);
        const data = (await res.json()) as { results?: TavilyResult[] };
        const results = data.results ?? [];
        if (results.length === 0) return ok("(no results)");
        const body = results.map((r, i) => `${i + 1}. ${r.title ?? "(untitled)"}\n   ${r.url ?? ""}\n   ${(r.content ?? "").trim()}`).join("\n\n");
        return ok(capText(body));
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  };
}

/** Pull markdown out of Crawl4AI's response defensively: the field is a string in some versions, `{ raw_markdown }` in others. */
function extractMarkdown(data: unknown): string | null {
  if (typeof data !== "object" || data === null) return null;
  const d = data as Record<string, unknown>;
  const md = d.markdown ?? d.result ?? (Array.isArray(d.results) ? (d.results[0] as Record<string, unknown> | undefined)?.markdown : undefined);
  if (typeof md === "string") return md;
  if (md && typeof md === "object") {
    const m = md as Record<string, unknown>;
    if (typeof m.raw_markdown === "string") return m.raw_markdown;
    if (typeof m.fit_markdown === "string") return m.fit_markdown;
  }
  return null;
}

function crawl4aiScrape(baseUrl: string, token: string | undefined): ToolDef {
  const endpoint = `${baseUrl.replace(/\/$/, "")}/md`;
  return {
    name: "web_scrape",
    description:
      "Fetch a single web page and return its main content as markdown (self-hosted Crawl4AI). Use this to READ a page you already have the URL for; use web_search to find URLs first.",
    inputSchema: { url: z.string().describe("the page URL to scrape") },
    handler: async (args: { url: string }) => {
      try {
        await assertSafeUrl(args.url);
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
      try {
        const res = await fetchWithTimeout(endpoint, {
          method: "POST",
          headers: { "content-type": "application/json", ...(token ? { authorization: `Bearer ${token}` } : {}) },
          body: JSON.stringify({ url: args.url }),
        });
        if (!res.ok) return err(`crawl4ai scrape failed: HTTP ${res.status} ${(await res.text()).slice(0, 500)}`);
        const md = extractMarkdown(await res.json());
        if (md == null) return err("crawl4ai returned no markdown for this page");
        return ok(capText(md));
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  };
}

function webFetch(): ToolDef {
  return {
    name: "web_fetch",
    description:
      "HTTP GET a URL and return the raw response body as text. For raw files, JSON/REST endpoints, robots.txt, etc. — NOT for reading human web pages (use web_scrape for those; it renders and returns clean markdown).",
    inputSchema: { url: z.string().describe("the URL to GET") },
    handler: async (args: { url: string }) => {
      try {
        await assertSafeUrl(args.url);
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
      try {
        const res = await fetchWithTimeout(args.url, { method: "GET", redirect: "follow" });
        const ctype = res.headers.get("content-type") ?? "(unknown)";
        const text = await res.text();
        const header = `HTTP ${res.status} • ${ctype}`;
        if (!res.ok) return err(`${header}\n\n${capText(text)}`);
        return ok(`${header}\n\n${capText(text)}`);
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  };
}

/** The web tools for a run, as one bundle: web_search and web_scrape when configured, web_fetch always. */
export function webTools(cfg: WebToolsConfig = {}): ToolBundle {
  const tools: ToolDef[] = [];
  if (cfg.tavilyApiKey) tools.push(tavilySearch(cfg.tavilyApiKey));
  if (cfg.crawl4ai?.url) tools.push(crawl4aiScrape(cfg.crawl4ai.url, cfg.crawl4ai.token || undefined));
  tools.push(webFetch());
  return { serverName: cfg.serverName ?? "web", tools };
}

// exported for unit tests
export const __test = { isBlockedIp, assertSafeUrl, capText, extractMarkdown, MAX_RETURN_CHARS };
