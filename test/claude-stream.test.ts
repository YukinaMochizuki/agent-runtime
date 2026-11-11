import { describe, expect, test } from "bun:test";
import type { CompactionEvent } from "../src/adapter.ts";
import { ClaudeStreamTranslator, compactionOf } from "../src/claude/stream.ts";

/**
 * The shapes below are what the Agent SDK actually emits (observed against a mock endpoint): a streamed
 * response is stream events plus one `assistant` message per content block under the message_start id;
 * Claude Code's non-streaming fallback is ONLY the per-block `assistant` messages, no stream events.
 */
type Out = ["token", string] | ["tool", string, unknown];

function run(messages: { type: string; event?: unknown; message?: unknown }[]): Out[] {
  const out: Out[] = [];
  const t = new ClaudeStreamTranslator({ onToken: (s) => out.push(["token", s]), onTool: (n, i) => out.push(["tool", n, i]) });
  for (const m of messages) t.handle(m);
  return out;
}

/** The same, keeping what `run` drops: each token's kind, and the drafts. */
function runKinds(messages: { type: string; event?: unknown; message?: unknown }[]): string[] {
  const out: string[] = [];
  const t = new ClaudeStreamTranslator({
    onToken: (s, thinking) => out.push(`${thinking ? "think" : "text"}:${s}`),
    onTool: (n) => out.push(`tool:${n}`),
    onToolDraft: (n) => out.push(`draft:${n}`),
  });
  for (const m of messages) t.handle(m);
  return out;
}

const se = (event: unknown) => ({ type: "stream_event", event });
const asst = (id: string, ...content: unknown[]) => ({ type: "assistant", message: { id, content } });
const text = (t: string) => ({ type: "text", text: t });
const tool = (id: string, name: string, input: unknown) => ({ type: "tool_use", id, name, input });

function streamed(id: string, prose: string, call?: { id: string; name: string; input: unknown }) {
  const ev: { type: string; event?: unknown; message?: unknown }[] = [
    se({ type: "message_start", message: { id } }),
    se({ type: "content_block_start", index: 0, content_block: text("") }),
    se({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: prose } }),
    asst(id, text(prose)),
    se({ type: "content_block_stop", index: 0 }),
  ];
  if (call) {
    ev.push(
      se({ type: "content_block_start", index: 1, content_block: { type: "tool_use", id: call.id, name: call.name, input: {} } }),
      se({ type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: JSON.stringify(call.input) } }),
      asst(id, tool(call.id, call.name, call.input)),
      se({ type: "content_block_stop", index: 1 }),
    );
  }
  ev.push(se({ type: "message_stop" }));
  return ev;
}

describe("ClaudeStreamTranslator", () => {
  test("a streamed response goes out once, from the stream; its assistant messages add nothing", () => {
    expect(run(streamed("msg_a", "Here is the table.", { id: "t1", name: "mcp__ws__ask", input: { q: 1 } }))).toEqual([
      ["token", "Here is the table."],
      ["tool", "mcp__ws__ask", { q: 1 }],
    ]);
  });

  test("the non-streaming fallback's prose reaches onToken, before the tool call it leads up to", () => {
    // The bug: only stream events were read, so this prose never reached the transcript and the ask
    // after it appeared with nothing above it.
    const out = run([asst("msg_fb", text("| A | B |\n|---|---|\n| 1 | 2 |")), asst("msg_fb", tool("t1", "mcp__ws__ask", { q: 1 }))]);
    expect(out).toEqual([
      ["token", "| A | B |\n|---|---|\n| 1 | 2 |"],
      ["tool", "mcp__ws__ask", { q: 1 }],
    ]);
  });

  test("a fallback after a partial stream does not repeat what already streamed", () => {
    const out = run([
      se({ type: "message_start", message: { id: "msg_a" } }),
      se({ type: "content_block_start", index: 0, content_block: text("") }),
      se({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "First half, " } }),
      // the stream dies here; the retry carries the whole response under a new id
      asst("msg_b", text("First half, second half.")),
      asst("msg_b", tool("t2", "mcp__ws__ask", {})),
    ]);
    expect(out).toEqual([
      ["token", "First half, "],
      ["token", "second half."],
      ["tool", "mcp__ws__ask", {}],
    ]);
  });

  test("a retry that answers differently is emitted whole rather than dropped", () => {
    const out = run([
      se({ type: "message_start", message: { id: "msg_a" } }),
      se({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Draft one" } }),
      asst("msg_b", text("Something else entirely.")),
    ]);
    expect(out).toEqual([
      ["token", "Draft one"],
      ["token", "\n\n"],
      ["token", "Something else entirely."],
    ]);
  });

  test("a repeated fallback message is replayed once", () => {
    const out = run([asst("msg_fb", text("Once.")), asst("msg_fb", text("Once."))]);
    expect(out).toEqual([["token", "Once."]]);
  });

  test("Claude Code's own synthetic error message is not replayed as agent prose", () => {
    // Shape observed from the CLI after a connection drop: a made-up assistant message under a random id.
    const out = run([
      ...streamed("msg_a", "Before."),
      { type: "assistant", is_api_error_message: true, message: { id: "44d1f4f2-9dd1", model: "<synthetic>", content: [text("API Error: Connection lost mid-response.")] } } as never,
      { type: "assistant", message: { id: "5a0c", model: "<synthetic>", content: [text("Prompt is too long")] } },
      ...streamed("msg_b", "After."),
    ]);
    // Two responses with no tool call between them are two paragraphs, not one run-on line.
    expect(out).toEqual([
      ["token", "Before."],
      ["token", "\n\n"],
      ["token", "After."],
    ]);
  });

  test("a fallback's thinking summary is prose, tagged as thinking; an empty (omitted) one is not", () => {
    const out = runKinds([
      asst("msg_fb", { type: "thinking", thinking: "" }),
      asst("msg_fb", { type: "thinking", thinking: "Checked the logs." }),
      asst("msg_fb", text("Answer.")),
    ]);
    // No blank line between them: the switch from thinking to the reply starts a new transcript row.
    expect(out).toEqual(["think:Checked the logs.", "text:Answer."]);
  });

  test("a fallback after partly streamed thinking still shows the progress update that never streamed", () => {
    // The reviewer's probe: one thinking delta streamed, the stream died, and the fallback carries the
    // whole reasoning summary, a progress update, text and the ask. Only what already showed is skipped.
    const out = run([
      se({ type: "message_start", message: { id: "m1" } }),
      se({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Reasoning about" } }),
      asst("m2", { type: "thinking", thinking: "Reasoning about it fully." }),
      asst("m2", { type: "thinking", thinking: "PROGRESS: found three LoRAs." }),
      asst("m2", text("Here is the table.")),
      asst("m2", tool("t1", "mcp__ws__ask", {})),
    ]);
    // Two thinking blocks are two paragraphs of one thinking row; the reply after them is its own row.
    expect(out.map((o) => (o[0] === "tool" ? `<${o[1]}>` : o[1])).join("")).toBe(
      "Reasoning about it fully.\n\nPROGRESS: found three LoRAs.Here is the table.<mcp__ws__ask>",
    );
  });

  test("a streamed progress update (thinking summary) before a tool call reaches onToken before the call", () => {
    // The shape Opus 5.5 produces after a tool result: an empty reasoning block, then a progress-update
    // block carrying its summary, then the tool call. At the default `omitted` display both are empty and
    // the human sees nothing; the session asks for `summarized`, and the summary must become prose.
    const out = run([
      se({ type: "message_start", message: { id: "msg_p" } }),
      se({ type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } }),
      se({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "" } }),
      se({ type: "content_block_stop", index: 0 }),
      se({ type: "content_block_start", index: 1, content_block: { type: "thinking", thinking: "" } }),
      se({ type: "content_block_delta", index: 1, delta: { type: "thinking_delta", thinking: "Found three LoRAs; asking which to use." } }),
      se({ type: "content_block_stop", index: 1 }),
      se({ type: "content_block_start", index: 2, content_block: { type: "tool_use", id: "t1", name: "mcp__ws__ask", input: {} } }),
      se({ type: "content_block_delta", index: 2, delta: { type: "input_json_delta", partial_json: "{}" } }),
      se({ type: "content_block_stop", index: 2 }),
      se({ type: "message_stop" }),
    ]);
    expect(out).toEqual([
      ["token", "Found three LoRAs; asking which to use."],
      ["tool", "mcp__ws__ask", {}],
    ]);
  });

  test("a thinking summary and the text after it are told apart by kind; deltas of one block are not split", () => {
    const out = runKinds([
      se({ type: "message_start", message: { id: "msg_s" } }),
      se({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Weighing " } }),
      se({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "two options." } }),
      se({ type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "Here is " } }),
      se({ type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "the answer." } }),
      se({ type: "message_stop" }),
    ]);
    expect(out).toEqual(["think:Weighing ", "think:two options.", "text:Here is ", "text:the answer."]);
  });

  test("two thinking blocks in a row stay one kind, a paragraph apart", () => {
    const out = runKinds([
      se({ type: "message_start", message: { id: "msg_t" } }),
      se({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "First." } }),
      se({ type: "content_block_delta", index: 1, delta: { type: "thinking_delta", thinking: "Second." } }),
      se({ type: "message_stop" }),
    ]);
    expect(out).toEqual(["think:First.", "think:\n\n", "think:Second."]);
  });

  test("a tool_use block opening is a draft; closing it is the call", () => {
    const out = runKinds(streamed("msg_d", "Writing it.", { id: "t1", name: "mcp__ws__write_box", input: { path: "a" } }));
    expect(out).toEqual(["text:Writing it.", "draft:mcp__ws__write_box", "tool:mcp__ws__write_box"]);
  });

  test("no separator at the start of a prose stretch after a tool call", () => {
    const out = run([
      ...streamed("msg_a", "Before.", { id: "t1", name: "x", input: {} }),
      se({ type: "message_start", message: { id: "msg_b" } }),
      se({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Next step." } }),
      se({ type: "message_stop" }),
    ]);
    expect(out).toEqual([
      ["token", "Before."],
      ["tool", "x", {}],
      ["token", "Next step."],
    ]);
  });

  test("a later streamed response after a fallback is handled normally", () => {
    const out = run([asst("msg_fb", text("Fallback.")), ...streamed("msg_c", "Streamed.")]);
    expect(out).toEqual([
      ["token", "Fallback."],
      ["token", "\n\n"],
      ["token", "Streamed."],
    ]);
  });
});

describe("compaction", () => {
  const boundary = (meta: unknown) => ({ type: "system", subtype: "compact_boundary", compact_metadata: meta });

  test("a compact_boundary in the stream is forwarded at its place, between the prose before and after it", () => {
    const out: string[] = [];
    const events: CompactionEvent[] = [];
    const t = new ClaudeStreamTranslator({
      onToken: (s) => out.push(`text:${s}`),
      onTool: (n) => out.push(`tool:${n}`),
      onCompaction: (e) => (events.push(e), out.push(`compaction:${e.trigger}:${e.tokensBefore}`)),
    });
    const script = [
      ...streamed("m1", "Before the summary.", { id: "tu1", name: "look", input: {} }),
      boundary({ trigger: "auto", pre_tokens: 167_000, post_tokens: 12_000 }),
      { type: "system", subtype: "init" }, // other system messages are not compactions
      ...streamed("m2", "After it."),
    ];
    for (const m of script) t.handle(m as never);
    expect(out).toEqual(["text:Before the summary.", "tool:look", "compaction:auto:167000", "text:After it."]);
    expect(events).toEqual([{ trigger: "auto", tokensBefore: 167_000, tokensAfter: 12_000 }]);
  });

  test("its fields are guarded: a manual one, one without post_tokens, and a malformed one", () => {
    expect(compactionOf(boundary({ trigger: "manual", pre_tokens: 90_000 }))).toEqual({ trigger: "manual", tokensBefore: 90_000 });
    expect(compactionOf(boundary(undefined))).toEqual({ trigger: "auto", tokensBefore: 0 });
    expect(compactionOf(boundary({ trigger: "weird", pre_tokens: "lots" }))).toEqual({ trigger: "auto", tokensBefore: 0 });
    expect(compactionOf({ type: "system", subtype: "init" })).toBeNull();
    expect(compactionOf({ type: "assistant" })).toBeNull();
  });

  test("a host that does not listen is unaffected (backward compatible)", () => {
    expect(run([...streamed("m1", "Hello."), boundary({ trigger: "auto", pre_tokens: 1 }), ...streamed("m2", "Again.")])).toEqual([
      ["token", "Hello."],
      ["token", "\n\n"],
      ["token", "Again."],
    ]);
  });
});
