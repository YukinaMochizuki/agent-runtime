import { describe, expect, test } from "bun:test";
import { claudePrompt, claudeQueryOptions, claudeSubprocessEnv, resultErrorMessage } from "../src/claude/sdk.ts";

/** The query options ARE the behaviour: each flag here is a lesson a host would otherwise relearn. */
describe("claudeQueryOptions", () => {
  test("hides every built-in tool by default, and reads no settings files", () => {
    const o = claudeQueryOptions({ prompt: "hi" });
    expect(o.tools).toEqual([]);
    expect(o.settingSources).toEqual([]);
    expect(o.permissionMode).toBe("dontAsk");
  });

  test("exposes only the built-in tools a host names", () => {
    expect(claudeQueryOptions({ prompt: "hi", builtinTools: ["WebSearch"] }).tools).toEqual(["WebSearch"]);
  });

  test("asks Claude Code for summarized thinking, without forcing a thinking type", () => {
    const o = claudeQueryOptions({ prompt: "hi", onToken: () => {} });
    expect(o.extraArgs).toEqual({ "thinking-display": "summarized" });
    // A `thinking` option would also pass --thinking and override each model's default type.
    expect(o.thinking).toBeUndefined();
    expect(o.includePartialMessages).toBe(true);
  });

  test("leaves the model to the SDK when none is named, and passes resume through", () => {
    const o = claudeQueryOptions({ prompt: "hi", resumeSessionId: "s1" });
    expect(o.model).toBeUndefined();
    expect(o.resume).toBe("s1");
  });

  test("a system prompt is recorded by default, and rendered fresh with systemPromptSnapshot: false", () => {
    expect(claudeQueryOptions({ prompt: "hi", systemPrompt: "S" }).systemPrompt).toBe("S");
    expect(claudeQueryOptions({ prompt: "hi", systemPrompt: "S", systemPromptSnapshot: false }).systemPrompt).toEqual({ type: "custom", prompt: "S", snapshot: false });
  });
});

describe("claudePrompt", () => {
  test("a string goes as the string", () => {
    expect(claudePrompt("hi")).toBe("hi");
  });

  test("parts go as ONE user message of content blocks, an image as a base64 source, in order", async () => {
    const p = claudePrompt([
      { type: "text", text: "look" },
      { type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" },
      { type: "text", text: "what is it?" },
    ]);
    expect(typeof p).not.toBe("string");
    const msgs = [];
    for await (const m of p as AsyncIterable<unknown>) msgs.push(m);
    expect(msgs).toEqual([
      {
        type: "user",
        parent_tool_use_id: null,
        message: {
          role: "user",
          content: [
            { type: "text", text: "look" },
            { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
            { type: "text", text: "what is it?" },
          ],
        },
      },
    ]);
  });
});

test("a failed result's message carries the CLI's errors when it gave some, and is unchanged when it did not", () => {
  expect(resultErrorMessage({ subtype: "error_during_execution", errors: ["No conversation found with session ID: s1"] })).toBe(
    'Claude Agent SDK run ended with "error_during_execution": No conversation found with session ID: s1',
  );
  expect(resultErrorMessage({ subtype: "error_during_execution", errors: ["a", "b"] })).toBe('Claude Agent SDK run ended with "error_during_execution": a; b');
  expect(resultErrorMessage({ subtype: "error_max_turns", errors: [] })).toBe('Claude Agent SDK run ended with "error_max_turns"');
  expect(resultErrorMessage({ subtype: "error_max_turns" })).toBe('Claude Agent SDK run ended with "error_max_turns"');
});

// The SDK's `env` replaces the CLI child's environment wholesale, so the switch that keeps the claude.ai
// account's connectors out of every session must ride along WITH the inherited env.
test("claude subprocess env turns off claude.ai connectors and keeps the inherited environment", () => {
  const env = claudeSubprocessEnv();
  expect(env.ENABLE_CLAUDEAI_MCP_SERVERS).toBe("false");
  expect(env.PATH).toBe(process.env.PATH);
  expect(env.HOME).toBe(process.env.HOME);
  expect(env.CLAUDE_CONFIG_DIR).toBe(process.env.CLAUDE_CONFIG_DIR);
  expect(claudeSubprocessEnv("/tmp/acct").CLAUDE_CONFIG_DIR).toBe("/tmp/acct");
});
