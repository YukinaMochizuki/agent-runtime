/**
 * The Claude driver end to end through the REAL Claude Code CLI, against a FAKE Anthropic Messages API on
 * localhost: no login, no quota. What the unit tests cannot see is the subprocess itself: that the
 * options this package passes are ones the CLI accepts, that a tool call reaches an in-process MCP tool
 * through the boundary, that the boundary's appended blocks (text and an image) reach the model with the
 * tool's result, that a prompt of parts reaches it as content blocks (an image included) on a fresh run and
 * on a resume, that a `settings` file's apiKeyHelper is what authenticates a turn (no settings source is
 * read otherwise), and that the session id, the prose and the usage come back.
 *
 *   bun scripts/verify-claude-fake.ts
 */
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { usageOfFailedTurn } from "../src/adapter.ts";
import { ClaudeBackend } from "../src/claude/index.ts";
import { ok, type ToolBundle, type ToolContent } from "../src/tool.ts";

// Three distinct, valid 1x1 PNGs (red, green, blue), so each check can tell WHICH image arrived where.
const PROMPT_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const APPEND_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNg+M8AAAICAQB7CYF4AAAAAElFTkSuQmCC";
const RESUME_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC";
/** Every base64 image block anywhere in `x` (a message list, a tool_result's content), by its data. */
function imagesIn(x: unknown): string[] {
  const out: string[] = [];
  const walk = (v: unknown): void => {
    if (Array.isArray(v)) return v.forEach(walk);
    if (!v || typeof v !== "object") return;
    const o = v as Block;
    if (o.type === "image" && o.source?.type === "base64" && o.source.data) out.push(o.source.data);
    Object.values(o).forEach(walk);
  };
  walk(x);
  return out;
}

const t0 = Date.now();
const log = (m: string) => console.log(`[+${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);
const fail = (m: string): never => {
  console.error(`FAIL: ${m}`);
  process.exit(1);
};
const check = (cond: unknown, m: string) => (cond ? log(`ok  ${m}`) : fail(m));

type Block = { type: string; text?: string; content?: unknown; tool_use_id?: string; source?: { type?: string; media_type?: string; data?: string } };
type Msg = { role: string; content: string | Block[] };
const seen: Array<{ path: string; system?: string; messages?: Msg[]; tools?: Array<{ name: string }>; apiKey?: string | null }> = [];

const ev = (type: string, data: unknown) => `event: ${type}\ndata: ${JSON.stringify({ type, ...(data as object) })}\n\n`;
function stream(id: string, blocks: Array<{ type: "text"; text: string } | { type: "tool_use"; id: string; name: string; input: unknown }>, stop: string): Response {
  let out = ev("message_start", { message: { id, type: "message", role: "assistant", model: "claude-fake", content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 30, output_tokens: 1 } } });
  blocks.forEach((b, index) => {
    if (b.type === "text") {
      out += ev("content_block_start", { index, content_block: { type: "text", text: "" } });
      out += ev("content_block_delta", { index, delta: { type: "text_delta", text: b.text } });
    } else {
      out += ev("content_block_start", { index, content_block: { type: "tool_use", id: b.id, name: b.name, input: {} } });
      out += ev("content_block_delta", { index, delta: { type: "input_json_delta", partial_json: JSON.stringify(b.input) } });
    }
    out += ev("content_block_stop", { index });
  });
  out += ev("message_delta", { delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: 9 } });
  out += ev("message_stop", {});
  return new Response(out, { headers: { "content-type": "text/event-stream" } });
}

const fake = Bun.serve({
  port: 0,
  async fetch(req) {
    const url = new URL(req.url);
    const body = req.method === "POST" ? ((await req.json().catch(() => ({}))) as { system?: unknown; messages?: Msg[]; tools?: Array<{ name: string }>; stream?: boolean }) : {};
    seen.push({ path: url.pathname, system: JSON.stringify(body.system ?? ""), messages: body.messages, tools: body.tools, apiKey: req.headers.get("x-api-key") });
    if (url.pathname.endsWith("/count_tokens")) return Response.json({ input_tokens: 10 });
    if (!url.pathname.endsWith("/v1/messages")) return Response.json({});
    const last = body.messages?.[body.messages.length - 1];
    const blocks = Array.isArray(last?.content) ? last!.content : [];
    const result = blocks.find((b) => b.type === "tool_result");
    const hasTool = body.tools?.some((t) => t.name === "mcp__ws__echo");
    if (JSON.stringify(body.messages ?? []).includes("CREDIT ERROR")) {
      return Response.json({ type: "error", error: { type: "invalid_request_error", message: "Your credit balance is too low to access the Anthropic API." } }, { status: 400 });
    }
    if (!hasTool) return stream("msg_side", [{ type: "text", text: "ok" }], "end_turn"); // the CLI's own side requests
    if (JSON.stringify(last).includes("SECOND TURN")) return stream("msg_3", [{ type: "text", text: "Seen again." }], "end_turn");
    if (JSON.stringify(body.messages).includes("once more")) return stream("msg_3", [{ type: "text", text: "Again." }], "end_turn"); // the system-prompt resumes below
    if (result) return stream("msg_2", [{ type: "text", text: `Done: ${JSON.stringify(result.content)}` }], "end_turn");
    return stream("msg_1", [{ type: "text", text: "Echoing." }, { type: "tool_use", id: "toolu_1", name: "mcp__ws__echo", input: { text: "hello" } }], "tool_use");
  },
});

process.env.ANTHROPIC_BASE_URL = `http://localhost:${fake.port}`;
process.env.ANTHROPIC_API_KEY = "sk-ant-fake";

const order: string[] = [];
const echo: ToolBundle = {
  serverName: "ws",
  tools: [{ name: "echo", description: "Echo the text back.", inputSchema: { text: z.string() }, handler: async (a: { text: string }) => (order.push("handler"), ok(`echo: ${a.text}`)) }],
};

const backend = new ClaudeBackend({
  defaultModel: "claude-sonnet-4-5",
  boundary: {
    recordStart: (c) => void order.push(`start:${c.recordedName}`),
    recordEnd: () => void order.push("end"),
    append: (): ToolContent[] => (
      order.push("append"), [{ type: "text", text: "[inbox] a message for you" }, { type: "image", data: APPEND_PNG, mimeType: "image/png" }]
    ),
  },
});

let sid = "";
const tokens: string[] = [];
const tools: string[] = [];
const ac = new AbortController();
const timer = setTimeout(() => ac.abort(), 60_000);
try {
  const r = await backend.run(
    {
      prompt: [
        { type: "text", text: "please echo hello" },
        { type: "image", data: PROMPT_PNG, mimeType: "image/png" },
      ],
      systemPrompt: "You are a test.",
      tools: [echo],
    },
    { onToken: (t) => tokens.push(t), onTool: (n) => tools.push(n), onSessionId: (id) => (sid = id), signal: ac.signal },
  );
  log(`turn ended: ${JSON.stringify(r.text)}`);
  check(sid && r.backendSessionId === sid, `the session id came early and matches the result (${sid})`);
  check(tools.includes("mcp__ws__echo"), "the tool call was reported");
  check(order.join(",") === "start:echo,handler,end,append", `the boundary ran in order: ${order.join(",")}`);
  const second = seen.filter((s) => s.path.endsWith("/v1/messages") && s.tools?.some((t) => t.name === "mcp__ws__echo")).at(-1)!;
  const sent = JSON.stringify(second.messages);
  check(sent.includes("echo: hello") && sent.includes("[inbox] a message for you"), "the tool result and the appended text reached the model");
  const first = second.messages![0]!;
  check(first.role === "user" && imagesIn(first).includes(PROMPT_PNG) && JSON.stringify(first).includes("please echo hello"), "the prompt's image reached the model as an image block, with its text, in the opening user message");
  const toolResult = (second.messages!.at(-1)!.content as Block[]).find((b) => b.type === "tool_result")!;
  const resultBlocks = JSON.stringify(toolResult.content);
  check(
    imagesIn(toolResult).includes(APPEND_PNG) && resultBlocks.indexOf("echo: hello") < resultBlocks.indexOf("[inbox]"),
    "the appended image reached the model as an image block inside the tool result, after the tool's own text",
  );
  check(r.text.startsWith("Done:"), "the final text is the last response's");
  check(tokens.join("").includes("Echoing."), "prose streamed through onToken");
  check(r.usage.outputTokens > 0, `usage came back (${JSON.stringify(r.usage)})`);
  const toolNames = new Set(seen.flatMap((s) => (s.tools ?? []).map((t) => t.name)));
  check(![...toolNames].some((n) => ["Bash", "Read", "Write", "Edit"].includes(n)), `no built-in tool was exposed (${[...toolNames].join(", ")})`);

  // A resume whose prompt carries an image: the new user message must hold it as an image block, and the
  // earlier images must still be in the resumed history (Claude Code keeps them in its session file).
  const before = seen.length;
  const r2 = await backend.continueRun(
    { backendSessionId: sid, prompt: [{ type: "text", text: "SECOND TURN: and this one?" }, { type: "image", data: RESUME_PNG, mimeType: "image/png" }], tools: [echo] },
    { signal: ac.signal },
  );
  log(`resumed turn ended: ${JSON.stringify(r2.text)}`);
  check(r2.backendSessionId === sid && r2.text === "Seen again.", "the resume continued the same session");
  const resumed = seen.slice(before).filter((s) => s.path.endsWith("/v1/messages") && s.tools?.some((t) => t.name === "mcp__ws__echo")).at(-1)!;
  const lastUser = resumed.messages!.at(-1)!;
  check(lastUser.role === "user" && imagesIn(lastUser).includes(RESUME_PNG), "on a resume, the prompt's image reached the model as an image block in the new user message");
  const kept = imagesIn(resumed.messages!.slice(0, -1));
  check(kept.includes(PROMPT_PNG) && kept.includes(APPEND_PNG), `on a resume, the earlier turn's images are still in the history sent (${kept.length} image block(s))`);

  // With no tools there is no in-process MCP server, so the SDK closes the CLI's input right after the
  // one message instead of waiting for the result: the turn must still run.
  const bare = await new ClaudeBackend({ defaultModel: "claude-sonnet-4-5" }).run(
    { prompt: [{ type: "text", text: "no tools here" }, { type: "image", data: PROMPT_PNG, mimeType: "image/png" }], tools: [] },
    { signal: ac.signal },
  );
  check(bare.text === "ok", `a prompt of parts runs with no tools too (${JSON.stringify(bare.text)})`);

  // A resume keeps the conversation's recorded system prompt by default; with liveSystemPrompt the
  // turn's own prompt replaces it, and the conversation's history is still there.
  const lastMain = () => seen.filter((s) => s.path.endsWith("/v1/messages") && s.tools?.some((t) => t.name === "mcp__ws__echo")).at(-1)!;
  await backend.continueRun({ backendSessionId: sid, prompt: "echo once more", systemPrompt: "You are an edited test.", tools: [echo] }, { signal: ac.signal });
  check(lastMain().system!.includes("You are a test.") && !lastMain().system!.includes("edited"), "by default a resume keeps the system prompt the conversation began with");
  const live = new ClaudeBackend({ defaultModel: "claude-sonnet-4-5", liveSystemPrompt: true });
  await live.continueRun({ backendSessionId: sid, prompt: "and once more", systemPrompt: "You are an edited test.", tools: [echo] }, { signal: ac.signal });
  const liveResumed = lastMain();
  check(liveResumed.system!.includes("You are an edited test.") && !liveResumed.system!.includes("You are a test."), "with liveSystemPrompt a resume sends the turn's new system prompt");
  check(JSON.stringify(liveResumed.messages).includes("please echo hello"), "and the resumed conversation keeps its history");

  // A login that is only a settings file: an empty config dir, and an apiKeyHelper in the file passed as
  // `settings`. With no ANTHROPIC_API_KEY in the environment, the helper's key is the only one there is.
  const dir = mkdtempSync(join(tmpdir(), "claude-settings-"));
  writeFileSync(join(dir, "key"), "sk-ant-from-helper\n");
  writeFileSync(join(dir, "settings.json"), JSON.stringify({ apiKeyHelper: `cat ${JSON.stringify(join(dir, "key"))}` }));
  const envKey = process.env.ANTHROPIC_API_KEY;
  delete process.env.ANTHROPIC_API_KEY;
  try {
    const mark = seen.length;
    const viaHelper = await new ClaudeBackend({ defaultModel: "claude-sonnet-4-5", configDir: dir, settings: join(dir, "settings.json") }).run({ prompt: "no tools here", tools: [] }, { signal: ac.signal });
    const keys = [...new Set(seen.slice(mark).filter((s) => s.path.endsWith("/v1/messages")).map((s) => s.apiKey))];
    check(viaHelper.text === "ok" && keys.length === 1 && keys[0] === "sk-ant-from-helper", `a turn authenticated with the settings file's apiKeyHelper (keys sent: ${keys.join(", ")})`);

    // An API error mid-turn: Claude Code reports it as a result, the SDK throws its own error for it, and
    // what the turn cost still rides on that error.
    const failing = await new ClaudeBackend({ defaultModel: "claude-sonnet-4-5", configDir: dir, settings: join(dir, "settings.json") })
      .run({ prompt: "CREDIT ERROR", tools: [] }, { signal: ac.signal })
      .then(() => null, (e: unknown) => e);
    check(failing instanceof Error && /credit balance is too low/i.test(failing.message), `an API error fails the turn with its message (${failing instanceof Error ? failing.message : String(failing)})`);
    check(usageOfFailedTurn(failing) != null, `and the failed turn's usage is on the error (${JSON.stringify(usageOfFailedTurn(failing))})`);
  } finally {
    if (envKey === undefined) delete process.env.ANTHROPIC_API_KEY;
    else process.env.ANTHROPIC_API_KEY = envKey;
  }
  log("PASS");
} catch (e) {
  fail(`the turn threw: ${e instanceof Error ? e.message : String(e)}\nrequests: ${seen.map((s) => s.path).join(" ")}`);
} finally {
  clearTimeout(timer);
  fake.stop(true);
}
