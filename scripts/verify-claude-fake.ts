/**
 * The Claude driver end to end through the REAL Claude Code CLI, against a FAKE Anthropic Messages API on
 * localhost: no login, no quota. What the unit tests cannot see is the subprocess itself: that the
 * options this package passes are ones the CLI accepts, that a tool call reaches an in-process MCP tool
 * through the boundary, that the boundary's appended block reaches the model with the tool's result, and
 * that the session id, the prose and the usage come back.
 *
 *   bun scripts/verify-claude-fake.ts
 */
import { z } from "zod";
import { ClaudeBackend } from "../src/claude/index.ts";
import { ok, type ToolBundle } from "../src/tool.ts";

const t0 = Date.now();
const log = (m: string) => console.log(`[+${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);
const fail = (m: string): never => {
  console.error(`FAIL: ${m}`);
  process.exit(1);
};
const check = (cond: unknown, m: string) => (cond ? log(`ok  ${m}`) : fail(m));

type Msg = { role: string; content: string | Array<{ type: string; text?: string; content?: unknown; tool_use_id?: string }> };
const seen: Array<{ path: string; messages?: Msg[]; tools?: Array<{ name: string }> }> = [];

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
    const body = req.method === "POST" ? ((await req.json().catch(() => ({}))) as { messages?: Msg[]; tools?: Array<{ name: string }>; stream?: boolean }) : {};
    seen.push({ path: url.pathname, messages: body.messages, tools: body.tools });
    if (url.pathname.endsWith("/count_tokens")) return Response.json({ input_tokens: 10 });
    if (!url.pathname.endsWith("/v1/messages")) return Response.json({});
    const last = body.messages?.[body.messages.length - 1];
    const blocks = Array.isArray(last?.content) ? last!.content : [];
    const result = blocks.find((b) => b.type === "tool_result");
    const hasTool = body.tools?.some((t) => t.name === "mcp__ws__echo");
    if (!hasTool) return stream("msg_side", [{ type: "text", text: "ok" }], "end_turn"); // the CLI's own side requests
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
    append: () => (order.push("append"), "[inbox] a message for you"),
  },
});

let sid = "";
const tokens: string[] = [];
const tools: string[] = [];
const ac = new AbortController();
const timer = setTimeout(() => ac.abort(), 60_000);
try {
  const r = await backend.run(
    { prompt: "please echo hello", systemPrompt: "You are a test.", tools: [echo] },
    { onToken: (t) => tokens.push(t), onTool: (n) => tools.push(n), onSessionId: (id) => (sid = id), signal: ac.signal },
  );
  log(`turn ended: ${JSON.stringify(r.text)}`);
  check(sid && r.backendSessionId === sid, `the session id came early and matches the result (${sid})`);
  check(tools.includes("mcp__ws__echo"), "the tool call was reported");
  check(order.join(",") === "start:echo,handler,end,append", `the boundary ran in order: ${order.join(",")}`);
  const second = seen.filter((s) => s.path.endsWith("/v1/messages") && s.tools?.some((t) => t.name === "mcp__ws__echo")).at(-1)!;
  const sent = JSON.stringify(second.messages);
  check(sent.includes("echo: hello") && sent.includes("[inbox] a message for you"), "the tool result and the appended block reached the model");
  check(r.text.startsWith("Done:"), "the final text is the last response's");
  check(tokens.join("").includes("Echoing."), "prose streamed through onToken");
  check(r.usage.outputTokens > 0, `usage came back (${JSON.stringify(r.usage)})`);
  const toolNames = new Set(seen.flatMap((s) => (s.tools ?? []).map((t) => t.name)));
  check(![...toolNames].some((n) => ["Bash", "Read", "Write", "Edit"].includes(n)), `no built-in tool was exposed (${[...toolNames].join(", ")})`);
  log("PASS");
} catch (e) {
  fail(`the turn threw: ${e instanceof Error ? e.message : String(e)}\nrequests: ${seen.map((s) => s.path).join(" ")}`);
} finally {
  clearTimeout(timer);
  fake.stop(true);
}
