import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ModelMessage } from "ai";
import {
  appendClaudeSession,
  appendToClaudeSession,
  claudeProjectDir,
  claudeSessionMessages,
  claudeToModelMessages,
  CONVERTED_MODEL,
  findClaudeSession,
  modelMessagesToClaude,
  readClaudeSession,
  type ClaudeMessage,
} from "../src/convert/index.ts";
import { TOOL_IMAGES_PREFIX } from "../src/openai-compat/compaction.ts";
import { withoutImages } from "../src/openai-compat/backend.ts";

const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const img = (data = PNG) => ({ type: "image", source: { type: "base64", media_type: "image/png", data } });

describe("claudeToModelMessages", () => {
  test("text, a prompt image, a tool call with its result and images; thinking and Claude Code's own lines dropped", () => {
    const claude: ClaudeMessage[] = [
      { role: "user", content: [{ type: "text", text: "look" }, img(), { type: "text", text: "[Image: source: /tmp/x/1.png]" }] },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "hmm", signature: "SIG" },
          { type: "text", text: "Calling." },
          { type: "tool_use", id: "toolu_1", name: "mcp__shuvi__echo", input: { text: "hi" } },
        ],
      },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", content: [{ type: "text", text: "echo: hi" }, img("B"), { type: "text", text: "[Image: source: /tmp/r.png]" }] }] },
      { role: "assistant", content: [{ type: "thinking", thinking: "only thinking", signature: "S2" }] },
      { role: "user", content: "<command-name>/compact</command-name>" },
      { role: "assistant", content: [{ type: "text", text: "Done." }] },
    ];
    expect(claudeToModelMessages(claude)).toEqual([
      { role: "user", content: [{ type: "text", text: "look" }, { type: "file", data: PNG, mediaType: "image/png" }] },
      { role: "assistant", content: [{ type: "text", text: "Calling." }, { type: "tool-call", toolCallId: "toolu_1", toolName: "shuvi__echo", input: { text: "hi" } }] },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "toolu_1", toolName: "shuvi__echo", output: { type: "text", value: "echo: hi\n[image 1 (image/png): sent in the next message]" } }] },
      { role: "user", content: [{ type: "text", text: `${TOOL_IMAGES_PREFIX} (1):` }, { type: "file", data: "B", mediaType: "image/png" }] },
      { role: "assistant", content: [{ type: "text", text: "Done." }] },
    ]);
  });

  test("a text document carries as its text, a PDF does not; Claude Code's stop lines do not; an orphan result is dropped", () => {
    const pdf = { type: "document", source: { type: "base64", media_type: "application/pdf", data: "JVBERi0=" } };
    const note = { type: "document", source: { type: "text", media_type: "text/plain", data: "line 1\n\nline 3" } };
    const out = claudeToModelMessages([
      { role: "user", content: [{ type: "text", text: "read" }, pdf, note] },
      { role: "assistant", content: [{ type: "tool_use", id: "r", name: "mcp__shuvi__read", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "r", content: [pdf, note] }, { type: "tool_result", tool_use_id: "nowhere", content: "x" }] },
      { role: "user", content: [{ type: "text", text: "[Request interrupted by user]" }] },
      { role: "user", content: "[Request interrupted by user for tool use]" },
    ]);
    expect(out).toEqual([
      { role: "user", content: [{ type: "text", text: "read" }, { type: "text", text: "line 1\n\nline 3" }] },
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "r", toolName: "shuvi__read", input: {} }] },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "r", toolName: "shuvi__read", output: { type: "text", value: "line 1\n\nline 3" } }] },
    ]);
  });

  test("an error result stays an error", () => {
    const out = claudeToModelMessages([
      { role: "assistant", content: [{ type: "tool_use", id: "t", name: "mcp__web__web_fetch", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t", content: "boom", is_error: true }] },
    ]);
    expect(out[1]).toEqual({ role: "tool", content: [{ type: "tool-result", toolCallId: "t", toolName: "web__web_fetch", output: { type: "error-text", value: "boom" } }] });
  });
});

describe("modelMessagesToClaude", () => {
  test("a tool's images go back into its result; reasoning dropped; foreign ids mapped onto Anthropic's alphabet", () => {
    const neutral: ModelMessage[] = [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "thinking elsewhere" },
          { type: "text", text: "Let me look." },
          { type: "tool-call", toolCallId: "functions.shuvi__echo:0", toolName: "shuvi__echo", input: { text: "x" } },
        ],
      },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "functions.shuvi__echo:0", toolName: "shuvi__echo", output: { type: "text", value: "echo: x\n[image 1 (image/png): sent in the next message]" } }] },
      { role: "user", content: [{ type: "text", text: `${TOOL_IMAGES_PREFIX} (1):` }, { type: "file", data: PNG, mediaType: "image/png" }] },
      { role: "assistant", content: [{ type: "text", text: "Seen." }] },
    ];
    expect(modelMessagesToClaude(neutral)).toEqual([
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "assistant", content: [{ type: "text", text: "Let me look." }, { type: "tool_use", id: "functions_shuvi__echo_0", name: "mcp__shuvi__echo", input: { text: "x" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "functions_shuvi__echo_0", content: [{ type: "text", text: "echo: x" }, img()] }] },
      { role: "assistant", content: [{ type: "text", text: "Seen." }] },
    ]);
  });

  test("a call nothing answered gets an error result; consecutive user messages merge", () => {
    const out = modelMessagesToClaude([
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "c1", toolName: "shuvi__say", input: {} }] },
      { role: "user", content: "next" },
      { role: "user", content: "and more" },
    ]);
    expect(out).toEqual([
      { role: "assistant", content: [{ type: "tool_use", id: "c1", name: "mcp__shuvi__say", input: {} }] },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "c1", content: [{ type: "text", text: "No result: the turn ended before this call returned." }], is_error: true },
          { type: "text", text: "next" },
          { type: "text", text: "and more" },
        ],
      },
    ]);
  });

  test("a tool's text is carried as written, blank lines and trailing newline included, around its images", () => {
    const value = "line 1\n\nline 3\n[image 1 (image/png): sent in the next message]\n\ntail\n";
    const out = modelMessagesToClaude([
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "t", toolName: "shuvi__x", input: {} }] },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "t", toolName: "shuvi__x", output: { type: "text", value } }] },
      { role: "user", content: [{ type: "text", text: `${TOOL_IMAGES_PREFIX} (1):` }, { type: "file", data: PNG, mediaType: "image/png" }] },
      { role: "user", content: [{ type: "text", text: "  " }, { type: "text", text: "two\n\nparagraphs" }] },
    ]);
    expect(out[1]!.content).toEqual([
      { type: "tool_result", tool_use_id: "t", content: [{ type: "text", text: "line 1\n\nline 3" }, img(), { type: "text", text: "\ntail\n" }] },
      { type: "text", text: "two\n\nparagraphs" },
    ]);
    expect(claudeToModelMessages(out)[1]).toEqual({ role: "tool", content: [{ type: "tool-result", toolCallId: "t", toolName: "shuvi__x", output: { type: "text", value } }] });
  });

  test("a tool message that answers only some calls: the others get an error result beside it", () => {
    const out = modelMessagesToClaude([
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "a", toolName: "shuvi__x", input: {} }, { type: "tool-call", toolCallId: "b", toolName: "shuvi__y", input: {} }] },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "a", toolName: "shuvi__x", output: { type: "text", value: "ok" } }] },
      { role: "user", content: "next" },
    ]);
    expect(out[1]).toEqual({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "a", content: [{ type: "text", text: "ok" }] },
        { type: "tool_result", tool_use_id: "b", content: [{ type: "text", text: "No result: the turn ended before this call returned." }], is_error: true },
        { type: "text", text: "next" },
      ],
    });
  });

  test("every output type: content parts as blocks, a denied call as an error with its reason, an empty result with no content", () => {
    const call = (id: string) => ({ type: "tool-call" as const, toolCallId: id, toolName: "shuvi__x", input: {} });
    const out = modelMessagesToClaude([
      { role: "assistant", content: [call("c"), call("d"), call("e"), call("j")] },
      {
        role: "tool",
        content: [
          { type: "tool-result", toolCallId: "c", toolName: "shuvi__x", output: { type: "content", value: [{ type: "text", text: "see" }, { type: "image-data", data: PNG, mediaType: "image/png" }, { type: "image-url", url: "https://x/i.png" }] } },
          { type: "tool-result", toolCallId: "d", toolName: "shuvi__x", output: { type: "execution-denied", reason: "not allowed" } },
          { type: "tool-result", toolCallId: "e", toolName: "shuvi__x", output: { type: "text", value: "" } },
          { type: "tool-result", toolCallId: "j", toolName: "shuvi__x", output: { type: "json", value: { n: 1 } } },
        ],
      },
    ]);
    expect(out[1]!.content).toEqual([
      { type: "tool_result", tool_use_id: "c", content: [{ type: "text", text: "see" }, img(), { type: "image", source: { type: "url", url: "https://x/i.png" } }] },
      { type: "tool_result", tool_use_id: "d", content: [{ type: "text", text: "Not run: not allowed" }], is_error: true },
      { type: "tool_result", tool_use_id: "e" },
      { type: "tool_result", tool_use_id: "j", content: [{ type: "text", text: '{"n":1}' }] },
    ]);
  });

  test("files: a URL string is a URL, a data URL its bytes, a PDF a document, a text file a text document; empty text and what Claude cannot hold are left out", () => {
    const out = modelMessagesToClaude([
      {
        role: "user",
        content: [
          { type: "text", text: "" },
          { type: "image", image: "https://x/a.png" },
          { type: "file", data: `data:image/png;base64,${PNG}`, mediaType: "image/png" },
          { type: "file", data: "JVBERi0=", mediaType: "application/pdf" },
          { type: "file", data: Buffer.from("hello").toString("base64"), mediaType: "text/plain" },
          { type: "file", data: "AAAA", mediaType: "application/zip" },
        ],
      },
    ]);
    expect(out).toEqual([
      {
        role: "user",
        content: [
          { type: "image", source: { type: "url", url: "https://x/a.png" } },
          img(),
          { type: "document", source: { type: "base64", media_type: "application/pdf", data: "JVBERi0=" } },
          { type: "document", source: { type: "text", media_type: "text/plain", data: "hello" } },
        ],
      },
    ]);
    expect(JSON.stringify(out)).not.toContain("carried");
  });

  test("round trip: what the OpenAI-compatible driver keeps comes back unchanged (reasoning aside)", () => {
    const neutral: ModelMessage[] = [
      { role: "user", content: [{ type: "text", text: "q" }, { type: "file", data: PNG, mediaType: "image/png" }] },
      { role: "assistant", content: [{ type: "text", text: "a" }, { type: "tool-call", toolCallId: "call_1", toolName: "shuvi__echo", input: { text: "y" } }] },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "call_1", toolName: "shuvi__echo", output: { type: "text", value: "echo: y\n[image 1 (image/png): sent in the next message]" } }] },
      { role: "user", content: [{ type: "text", text: `${TOOL_IMAGES_PREFIX} (1):` }, { type: "file", data: "B", mediaType: "image/png" }] },
      { role: "assistant", content: [{ type: "text", text: "done" }] },
    ];
    expect(claudeToModelMessages(modelMessagesToClaude(neutral))).toEqual(neutral);
  });
});

describe("Claude session files", () => {
  const cfg = () => mkdtempSync(join(tmpdir(), "convert-"));

  test("a new session, then more appended: read back in order, positions resume after each part", () => {
    const configDir = cfg();
    const first = appendToClaudeSession(null, [{ role: "user", content: "one" }, { role: "assistant", content: [{ type: "text", text: "1" }] }], { configDir, cwd: "/srv/shuvi" });
    expect(findClaudeSession(first.sessionId, configDir)).toBe(join(claudeProjectDir("/srv/shuvi", configDir), `${first.sessionId}.jsonl`));
    const second = appendToClaudeSession(first.sessionId, [{ role: "user", content: "two" }, { role: "assistant", content: [{ type: "text", text: "2" }] }], { configDir });
    expect(second.sessionId).toBe(first.sessionId);
    expect(claudeSessionMessages(first.sessionId, { configDir }).messages).toEqual([
      { role: "user", content: [{ type: "text", text: "one" }] },
      { role: "assistant", content: [{ type: "text", text: "1" }] },
      { role: "user", content: [{ type: "text", text: "two" }] },
      { role: "assistant", content: [{ type: "text", text: "2" }] },
    ]);
    const after = claudeSessionMessages(first.sessionId, { configDir, after: first.end });
    expect(after.afterFound).toBe(true);
    expect(after.end).toBe(second.end);
    expect(after.messages.map((m) => m.role)).toEqual(["user", "assistant"]);
    // Every entry is chained, timestamped, and the assistant ones are not labelled as a Claude model.
    const lines = readFileSync(findClaudeSession(first.sessionId, configDir)!, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    lines.forEach((e, i) => {
      expect(e.parentUuid).toBe(i ? lines[i - 1].uuid : null);
      expect(typeof e.timestamp).toBe("string");
    });
    expect(lines.filter((e) => e.type === "assistant").every((e) => e.message.model === CONVERTED_MODEL)).toBe(true);
  });

  test("reads as the CLI writes: blocks merged by message, meta entries and side entries skipped, the chain from the latest message", () => {
    const configDir = cfg();
    const dir = claudeProjectDir("/w", configDir);
    mkdirSync(dir, { recursive: true });
    const e = (o: object) => JSON.stringify({ isSidechain: false, timestamp: "2026-11-13T00:00:00.000Z", ...o });
    writeFileSync(
      join(dir, "s1.jsonl"),
      [
        JSON.stringify({ type: "queue-operation" }),
        e({ type: "user", uuid: "u1", parentUuid: null, message: { role: "user", content: [{ type: "text", text: "hello" }] } }),
        e({ type: "attachment", uuid: "a1", parentUuid: "u1" }),
        e({ type: "user", uuid: "m1", parentUuid: "a1", isMeta: true, message: { role: "user", content: [{ type: "text", text: "[Image: source: /tmp/1.png]" }] } }),
        e({ type: "assistant", uuid: "b1", parentUuid: "m1", message: { id: "msg_1", role: "assistant", content: [{ type: "thinking", thinking: "t", signature: "s" }] } }),
        e({ type: "assistant", uuid: "b2", parentUuid: "b1", message: { id: "msg_1", role: "assistant", content: [{ type: "text", text: "hi there" }] } }),
        e({ type: "user", uuid: "x1", parentUuid: "b2", isSidechain: true, message: { role: "user", content: "a subagent's" } }),
        "{ truncated",
      ].join("\n") + "\n",
    );
    const read = readClaudeSession("s1", { configDir });
    expect(read.messages).toEqual([
      { role: "user", content: [{ type: "text", text: "hello" }] },
      { role: "assistant", content: [{ type: "thinking", thinking: "t", signature: "s" }, { type: "text", text: "hi there" }] },
    ]);
    expect(read.end).toBe("b2");
    // Appending chains onto the latest conversation entry, not the side entry or the torn line.
    appendClaudeSession("s1", [{ role: "user", content: "more" }], { configDir });
    const last = JSON.parse(readFileSync(join(dir, "s1.jsonl"), "utf8").trim().split("\n").at(-1)!);
    expect(last.parentUuid).toBe("b2");
  });

  test("after a compaction the chain starts at its summary; a position compacted away returns all of it", () => {
    const configDir = cfg();
    const dir = claudeProjectDir("/w", configDir);
    mkdirSync(dir, { recursive: true });
    const e = (o: object) => JSON.stringify({ isSidechain: false, timestamp: "2026-11-13T00:00:00.000Z", ...o });
    writeFileSync(
      join(dir, "s2.jsonl"),
      [
        e({ type: "user", uuid: "u1", parentUuid: null, message: { role: "user", content: "old question" } }),
        e({ type: "assistant", uuid: "a1", parentUuid: "u1", message: { id: "m1", role: "assistant", content: [{ type: "text", text: "old answer" }] } }),
        e({ type: "system", subtype: "compact_boundary", uuid: "cb", parentUuid: null, logicalParentUuid: "a1" }),
        e({ type: "user", uuid: "sum", parentUuid: "cb", isCompactSummary: true, message: { role: "user", content: "This session is being continued… SUMMARY" } }),
        e({ type: "user", uuid: "q2", parentUuid: "sum", message: { role: "user", content: "new question" } }),
      ].join("\n") + "\n",
    );
    const read = claudeSessionMessages("s2", { configDir, after: "a1" });
    expect(read.afterFound).toBe(false);
    expect(read.messages).toEqual([
      { role: "user", content: "This session is being continued… SUMMARY" },
      { role: "user", content: "new question" },
    ]);
  });

  test("a position between a tool call and its result reads from the call", () => {
    const configDir = cfg();
    const dir = claudeProjectDir("/w", configDir);
    mkdirSync(dir, { recursive: true });
    const e = (o: object) => JSON.stringify({ isSidechain: false, timestamp: "2026-11-13T00:00:00.000Z", ...o });
    writeFileSync(
      join(dir, "s3.jsonl"),
      [
        e({ type: "user", uuid: "u1", parentUuid: null, message: { role: "user", content: "go" } }),
        e({ type: "assistant", uuid: "a1", parentUuid: "u1", message: { id: "m1", role: "assistant", content: [{ type: "text", text: "calling" }] } }),
        e({ type: "assistant", uuid: "a2", parentUuid: "a1", message: { id: "m1", role: "assistant", content: [{ type: "tool_use", id: "t", name: "mcp__shuvi__x", input: {} }] } }),
        e({ type: "user", uuid: "r1", parentUuid: "a2", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t", content: "done" }] } }),
        e({ type: "assistant", uuid: "a3", parentUuid: "r1", message: { id: "m2", role: "assistant", content: [{ type: "text", text: "finished" }] } }),
      ].join("\n") + "\n",
    );
    const read = claudeSessionMessages("s3", { configDir, after: "a2" });
    expect(read.afterFound).toBe(false); // it holds the call, which came before `after`
    expect(read.messages.map((m) => m.role)).toEqual(["assistant", "tool", "assistant"]);
    expect(read.messages[0]).toEqual({ role: "assistant", content: [{ type: "text", text: "calling" }, { type: "tool-call", toolCallId: "t", toolName: "shuvi__x", input: {} }] });
  });

  test("appended timestamps come after every one already in the file", () => {
    const configDir = cfg();
    const dir = claudeProjectDir("/w", configDir);
    mkdirSync(dir, { recursive: true });
    const future = "2099-01-01T00:00:00.000Z";
    writeFileSync(join(dir, "s4.jsonl"), JSON.stringify({ type: "user", uuid: "u1", parentUuid: null, isSidechain: false, timestamp: future, message: { role: "user", content: "hi" } }) + "\n");
    appendClaudeSession("s4", [{ role: "assistant", content: [{ type: "text", text: "yo" }] }], { configDir });
    const last = JSON.parse(readFileSync(join(dir, "s4.jsonl"), "utf8").trim().split("\n").at(-1)!);
    expect(Date.parse(last.timestamp)).toBeGreaterThan(Date.parse(future));
  });

  test("appending to a session that does not exist is an error, not a new session", () => {
    expect(() => appendClaudeSession("missing", [{ role: "user", content: "x" }], { configDir: cfg() })).toThrow(/no Claude session missing/);
  });
});

test("withoutImages: images in history become notes in what is sent", () => {
  const history: ModelMessage[] = [
    { role: "user", content: [{ type: "text", text: "see" }, { type: "file", data: PNG, mediaType: "image/png" }] },
    { role: "assistant", content: [{ type: "text", text: "ok" }] },
  ];
  expect(withoutImages(history)).toEqual([
    { role: "user", content: [{ type: "text", text: "see" }, { type: "text", text: "[an image (image/png) was attached; this model does not take images, so it was not sent]" }] },
    history[1]!,
  ]);
  expect(history[0]!.content).toHaveLength(2); // the stored history is not touched
});
