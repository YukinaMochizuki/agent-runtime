/**
 * `agent-runtime/convert` end to end, through the REAL Claude Code CLI against a FAKE Anthropic Messages API
 * and the OpenAI-compatible driver against a FAKE chat-completions endpoint: no login, no quota. A host
 * carrying a conversation between the two drivers does exactly this, and only the subprocess can say whether
 * a session written from outside is one the CLI resumes:
 *
 *   T1  on Claude: a prompt with an image, a tool call whose result carries an image, an answer with thinking
 *   T2  on the OpenAI-compatible driver, on T1 carried over whole (`claudeSessionMessages`)
 *   T3  on Claude again, on its OWN session with only T2 appended (`appendToClaudeSession`)
 *   T4  on the OpenAI-compatible driver again, with only T3 carried (read `after` the position T2 ended at)
 *   T5  on Claude, on a NEW session written from the whole OpenAI-compatible history
 *
 * Checked on the wire: T2's request holds T1 (text, both images, the call paired with its result, the tool
 * name in the OpenAI form, no reasoning); T3's request holds T1 as Claude wrote it (its signed thinking too)
 * followed by T2 (its tool call in Claude's form, its result, its answer) and T3's prompt.
 *
 *   bun scripts/verify-convert-fake.ts
 */
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ModelMessage } from "ai";
import { z } from "zod";
import { ClaudeBackend } from "../src/claude/index.ts";
import { appendToClaudeSession, claudeSessionMessages, findClaudeSession } from "../src/convert/index.ts";
import { OpenAICompatBackend } from "../src/openai-compat/index.ts";
import type { ToolBundle } from "../src/tool.ts";

const PROMPT_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const TOOL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNg+M8AAAICAQB7CYF4AAAAAElFTkSuQmCC";

const t0 = Date.now();
const log = (m: string) => console.log(`[+${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);
const fail = (m: string): never => {
  console.error(`FAIL: ${m}`);
  process.exit(1);
};
const check = (cond: unknown, m: string) => (cond ? log(`ok  ${m}`) : fail(m));
const has = (x: unknown, s: string) => JSON.stringify(x).includes(s);

// --- fake Anthropic ---------------------------------------------------------------------------------------
type AB = { type: "text"; text: string } | { type: "thinking"; thinking: string; signature: string } | { type: "tool_use"; id: string; name: string; input: unknown };
const ev = (type: string, data: unknown) => `event: ${type}\ndata: ${JSON.stringify({ type, ...(data as object) })}\n\n`;
// The reply is recorded under the model the request named: Claude Code drops thinking recorded under another.
function anthropic(id: string, blocks: AB[], stop: string, model = "claude-fake"): Response {
  let out = ev("message_start", { message: { id, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 30, output_tokens: 1 } } });
  blocks.forEach((b, index) => {
    if (b.type === "text") {
      out += ev("content_block_start", { index, content_block: { type: "text", text: "" } });
      out += ev("content_block_delta", { index, delta: { type: "text_delta", text: b.text } });
    } else if (b.type === "thinking") {
      out += ev("content_block_start", { index, content_block: { type: "thinking", thinking: "", signature: "" } });
      out += ev("content_block_delta", { index, delta: { type: "thinking_delta", thinking: b.thinking } });
      out += ev("content_block_delta", { index, delta: { type: "signature_delta", signature: b.signature } });
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
type Msg = { role: string; content: string | Array<Record<string, unknown>> };
const claudeSeen: Array<{ messages: Msg[] }> = [];
const claudeApi = Bun.serve({
  port: 0,
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname.endsWith("/count_tokens")) return Response.json({ input_tokens: 10 });
    if (!url.pathname.endsWith("/v1/messages")) return Response.json({});
    const body = (await req.json()) as { model: string; messages: Msg[]; tools?: Array<{ name: string }> };
    // Side requests (titles, summaries) declare none of the host's tools.
    if (!body.tools?.some((t) => t.name === "mcp__shuvi__look")) return anthropic("msg_side", [{ type: "text", text: "ok" }], "end_turn");
    claudeSeen.push(body);
    const last = JSON.stringify(body.messages.at(-1));
    if (last.includes("T1-PROMPT")) return anthropic("msg_t1a", [{ type: "thinking", thinking: "T1-THINKING", signature: "SIG-T1" }, { type: "tool_use", id: "toolu_T1", name: "mcp__shuvi__look", input: { what: "T1" } }], "tool_use", body.model);
    if (last.includes("toolu_T1")) return anthropic("msg_t1b", [{ type: "text", text: "T1-ANSWER" }], "end_turn", body.model);
    return anthropic("msg_t3", [{ type: "text", text: "T3-ANSWER" }], "end_turn", body.model);
  },
});
process.env.ANTHROPIC_BASE_URL = `http://localhost:${claudeApi.port}`;
process.env.ANTHROPIC_API_KEY = "sk-ant-fake";

// --- fake OpenAI-compatible ---------------------------------------------------------------------------------
const chunk = (delta: unknown, finish: string | null = null) => ({ id: "c", object: "chat.completion.chunk", created: 1, model: "fake-1", choices: [{ index: 0, delta, finish_reason: finish }] });
const sse = (chunks: unknown[]) => new Response(chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") + "data: [DONE]\n\n", { headers: { "content-type": "text/event-stream" } });
const openaiSeen: Array<{ messages: Msg[] }> = [];
const openaiApi = Bun.serve({
  port: 0,
  async fetch(req) {
    const body = (await req.json()) as { messages: Msg[] };
    openaiSeen.push(body);
    const last = body.messages.at(-1)!;
    if (last.role === "tool") return sse([chunk({ role: "assistant", content: "T2-ANSWER" }), chunk({}, "stop")]);
    return sse([chunk({ role: "assistant", reasoning_content: "T2-REASONING" }), chunk({ tool_calls: [{ index: 0, id: "functions.shuvi__look:0", type: "function", function: { name: "shuvi__look", arguments: '{"what":"T2"}' } }] }), chunk({}, "tool_calls")]);
  },
});

const look: ToolBundle = {
  serverName: "shuvi",
  tools: [{ name: "look", description: "Look at something.", inputSchema: { what: z.string() }, handler: async (a: { what: string }) => ({ content: [{ type: "text", text: `looked at ${a.what}` }, ...(a.what === "T1" ? [{ type: "image" as const, data: TOOL_PNG, mimeType: "image/png" }] : [])] }) }],
};
const configDir = mkdtempSync(join(tmpdir(), "verify-convert-"));
const claude = new ClaudeBackend({ configDir, defaultModel: "claude-sonnet-4-5", liveSystemPrompt: true });
const systemPrompt = "You are the verify-convert assistant.";

try {
  // T1 on Claude.
  const t1 = await claude.run({ prompt: [{ type: "text", text: "T1-PROMPT look at this" }, { type: "image", data: PROMPT_PNG, mimeType: "image/png" }], systemPrompt, tools: [look] }, {});
  check(t1.text.includes("T1-ANSWER"), "T1 ran on Claude");
  check(findClaudeSession(t1.backendSessionId, configDir), "T1's session file is where convert looks for it");

  // T1 carried to the OpenAI-compatible driver.
  const carried = claudeSessionMessages(t1.backendSessionId, { configDir });
  const claudeEnd = carried.end;
  const store = new Map<string, ModelMessage[]>([["oa", carried.messages]]);
  const oa = new OpenAICompatBackend({
    provider: { name: "fake", baseURL: `http://localhost:${openaiApi.port}/v1`, apiKey: "k" },
    store: { load: async (id) => [...(store.get(id) ?? [])], append: async (id, m) => void store.set(id, [...(store.get(id) ?? []), ...m]) },
    defaultModel: "fake-1",
    vision: true,
    boundary: {},
  });
  const t2 = await oa.continueRun({ backendSessionId: "oa", prompt: "T2-PROMPT and now", systemPrompt, tools: [look] }, {});
  check(t2.text === "T2-ANSWER", "T2 ran on the OpenAI-compatible driver");
  const r2 = openaiSeen[0]!;
  check(has(r2.messages, "T1-PROMPT") && has(r2.messages, "T1-ANSWER"), "T2's request holds T1's prompt and answer");
  check(has(r2.messages, PROMPT_PNG) && has(r2.messages, TOOL_PNG), "T2's request holds both of T1's images");
  check(has(r2.messages, '"name":"shuvi__look"') && has(r2.messages, "toolu_T1") && has(r2.messages, "looked at T1"), "T1's call reaches it in the OpenAI form, paired with its result");
  check(!has(r2.messages, "T1-THINKING") && !has(r2.messages, "[Image: source"), "no reasoning and no Claude Code path lines are carried");

  // Only T2 carried back onto Claude's own session.
  const delta = store.get("oa")!.slice(carried.messages.length);
  check(delta.length >= 3 && has(delta[0], "T2-PROMPT"), "the part Claude lacks is exactly T2");
  const appended = appendToClaudeSession(t1.backendSessionId, delta, { configDir });
  check(appended.sessionId === t1.backendSessionId && appended.end !== claudeEnd, "T2 appended onto T1's session");
  const before = claudeSeen.length;
  const t3 = await claude.continueRun({ backendSessionId: t1.backendSessionId, prompt: "T3-PROMPT back on claude", systemPrompt, tools: [look] }, {});
  check(t3.text.includes("T3-ANSWER") && t3.backendSessionId === t1.backendSessionId, "T3 resumed the same session on Claude");
  const r3 = claudeSeen.slice(before).at(-1)!.messages;
  // Each marker searched for after the one before it: the same tool name appears in T1 and in T2.
  const wire = JSON.stringify(r3);
  let at = 0;
  const missing = ["T1-PROMPT", "toolu_T1", "looked at T1", "T1-ANSWER", "T2-PROMPT", '"id":"functions_shuvi__look_0","name":"mcp__shuvi__look"', "looked at T2", "T2-ANSWER", "T3-PROMPT"].filter((s) => {
    const i = wire.indexOf(s, at);
    if (i < 0) return true;
    at = i + s.length;
    return false;
  });
  check(!missing.length, `T3's request holds T1 (native), then T2 (carried), then T3's prompt, in order${missing.length ? `; missing in order: ${missing.join(", ")}` : ""}`);
  check(has(r3, "SIG-T1"), "T1's own signed thinking is still sent: Claude's native part was never converted");
  check(!has(r3, "T2-REASONING"), "T2's reasoning is not carried");

  // Only T3 carried back to the OpenAI-compatible conversation: read after where T2's append ended.
  const t3part = claudeSessionMessages(t1.backendSessionId, { configDir, after: appended.end });
  check(t3part.afterFound && t3part.messages.length === 2 && has(t3part.messages[0], "T3-PROMPT"), "after the position T2 ended at, the session holds exactly T3");
  store.set("oa", [...store.get("oa")!, ...t3part.messages]);
  const before4 = openaiSeen.length;
  await oa.continueRun({ backendSessionId: "oa", prompt: "T4-PROMPT once more", systemPrompt, tools: [look] }, {});
  const r4 = JSON.stringify(openaiSeen[before4]!.messages);
  const count = (s: string) => r4.split(s).length - 1;
  check(count("T1-PROMPT") === 1 && count("T2-PROMPT") === 1 && count("T3-PROMPT") === 1 && r4.indexOf("T3-PROMPT") > r4.indexOf("T2-ANSWER"), "T4's request holds T1, T2 and T3 once each, in order");

  // A new Claude session written from the whole OpenAI-compatible history, resumed by the CLI.
  const fresh = appendToClaudeSession(null, store.get("oa")!, { configDir });
  const before5 = claudeSeen.length;
  const t5 = await claude.continueRun({ backendSessionId: fresh.sessionId, prompt: "T5-PROMPT on a new session", systemPrompt, tools: [look] }, {});
  check(t5.backendSessionId === fresh.sessionId, "the CLI resumed a session written from outside");
  const r5 = JSON.stringify(claudeSeen.slice(before5).at(-1)!.messages);
  check(["T1-PROMPT", "T2-PROMPT", "T3-PROMPT", "T4-PROMPT", "T5-PROMPT"].every((s, k, all) => r5.includes(s) && (k === 0 || r5.indexOf(s) > r5.indexOf(all[k - 1]!))), "its request holds T1 to T4, then T5's prompt");
  check(has(claudeSeen.slice(before5).at(-1)!.messages, TOOL_PNG), "T1's tool image is back inside a tool_result");
  log("all good");
} finally {
  claudeApi.stop(true);
  openaiApi.stop(true);
}
process.exit(0);
