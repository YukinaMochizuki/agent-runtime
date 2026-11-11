import { afterEach, describe, expect, test } from "bun:test";
import { __test, probeCrawl4ai, webTools } from "../src/tools/web.ts";

const { isBlockedIp, assertSafeUrl, capText, extractMarkdown, MAX_RETURN_CHARS } = __test;

describe("isBlockedIp", () => {
  test("blocks loopback / private / link-local / metadata", () => {
    for (const ip of ["127.0.0.1", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.169.254", "0.0.0.0", "100.64.0.1", "::1", "::", "fe80::1", "fc00::1", "::ffff:127.0.0.1"]) {
      expect(isBlockedIp(ip)).toBe(true);
    }
  });
  test("allows public addresses", () => {
    for (const ip of ["8.8.8.8", "1.1.1.1", "172.15.0.1", "172.32.0.1", "93.184.216.34", "2606:4700:4700::1111"]) {
      expect(isBlockedIp(ip)).toBe(false);
    }
  });
});

describe("assertSafeUrl", () => {
  test("rejects non-http schemes", async () => {
    await expect(assertSafeUrl("ftp://example.com/x")).rejects.toThrow(/scheme/);
    await expect(assertSafeUrl("file:///etc/passwd")).rejects.toThrow(/scheme/);
  });
  test("rejects invalid URLs", async () => {
    await expect(assertSafeUrl("not a url")).rejects.toThrow(/invalid URL/);
  });
  test("rejects localhost + private IP literals", async () => {
    await expect(assertSafeUrl("http://localhost:8080/")).rejects.toThrow(/localhost/);
    await expect(assertSafeUrl("http://127.0.0.1/")).rejects.toThrow(/private|loopback/);
    await expect(assertSafeUrl("http://169.254.169.254/latest/meta-data/")).rejects.toThrow(/private|loopback/);
    await expect(assertSafeUrl("http://10.0.0.5/")).rejects.toThrow(/private|loopback/);
  });
  test("allows a public IP literal", async () => {
    await expect(assertSafeUrl("https://8.8.8.8/")).resolves.toBeUndefined();
  });
});

describe("capText", () => {
  test("passes short text through", () => {
    expect(capText("hello")).toBe("hello");
  });
  test("truncates over the cap with a marker", () => {
    const out = capText("a".repeat(MAX_RETURN_CHARS + 100));
    expect(out.length).toBeLessThan(MAX_RETURN_CHARS + 100);
    expect(out).toContain("truncated");
  });
});

describe("extractMarkdown", () => {
  test("handles string, nested, results[], and missing shapes", () => {
    expect(extractMarkdown({ markdown: "# hi" })).toBe("# hi");
    expect(extractMarkdown({ markdown: { raw_markdown: "# raw" } })).toBe("# raw");
    expect(extractMarkdown({ markdown: { fit_markdown: "# fit" } })).toBe("# fit");
    expect(extractMarkdown({ results: [{ markdown: "# in results" }] })).toBe("# in results");
    expect(extractMarkdown({ nothing: true })).toBeNull();
    expect(extractMarkdown(null)).toBeNull();
  });
});

describe("webTools", () => {
  const names = (cfg: Parameters<typeof webTools>[0]) => webTools(cfg).tools.map((t) => t.name);

  test("web_fetch is always present; web_search needs a key; web_scrape needs a Crawl4AI instance", () => {
    expect(names({})).toEqual(["web_fetch"]);
    expect(names({ tavilyApiKey: "tvly-test" })).toEqual(["web_search", "web_fetch"]);
    expect(names({ tavilyApiKey: "tvly-test", crawl4ai: { url: "http://c4ai:11235" } })).toEqual(["web_search", "web_scrape", "web_fetch"]);
  });

  test("the bundle is named `web` unless the host names it", () => {
    expect(webTools().serverName).toBe("web");
    expect(webTools({ serverName: "net" }).serverName).toBe("net");
  });

  test("a URL the guard refuses never reaches the network", async () => {
    const fetchTool = webTools().tools.find((t) => t.name === "web_fetch")!;
    const res = await fetchTool.handler({ url: "http://169.254.169.254/latest/meta-data/" });
    expect(res.isError).toBe(true);
  });
});

describe("probeCrawl4ai", () => {
  let server: ReturnType<typeof Bun.serve> | null = null;
  afterEach(() => {
    server?.stop(true);
    server = null;
  });

  test("true when /health answers OK (with the token), false when it does not or nothing listens", async () => {
    let auth = null as string | null;
    server = Bun.serve({ port: 0, fetch: (req) => ((auth = req.headers.get("authorization")), new URL(req.url).pathname === "/health" ? new Response("ok") : new Response("no", { status: 404 })) });
    expect(await probeCrawl4ai(`http://127.0.0.1:${server.port}/`, "t0k")).toBe(true);
    expect(auth).toBe("Bearer t0k");
    expect(await probeCrawl4ai(`http://127.0.0.1:${server.port}/nope`)).toBe(false);
    expect(await probeCrawl4ai("http://127.0.0.1:1", undefined, 500)).toBe(false);
  });
});
