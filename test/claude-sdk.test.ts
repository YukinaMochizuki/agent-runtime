import { describe, expect, test } from "bun:test";
import { LostSessionError, usageOfFailedTurn, withTurnUsage } from "../src/adapter.ts";
import { claudePrompt, claudeQueryOptions, claudeSubprocessEnv, resultError, resultErrorMessage } from "../src/claude/sdk.ts";

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

test("a resume the CLI cannot find is a LostSessionError; every other failed result is a plain Error", () => {
  const lost = { subtype: "error_during_execution", errors: ["No conversation found with session ID: s1"] };
  const e = resultError(lost, "s1");
  expect(e).toBeInstanceOf(LostSessionError);
  expect((e as LostSessionError).backendSessionId).toBe("s1");
  expect(e.message).toBe(resultErrorMessage(lost));
  // Not resuming: the text cannot be about our session.
  expect(resultError(lost)).not.toBeInstanceOf(LostSessionError);
  // Another failure on a resume (an outage, an expired login) must not read as lost.
  const other = resultError({ subtype: "error_during_execution", errors: ["API Error: 529 overloaded"] }, "s1");
  expect(other).not.toBeInstanceOf(LostSessionError);
  expect(other.message).toBe('Claude Agent SDK run ended with "error_during_execution": API Error: 529 overloaded');
  // An error that only quotes the phrase, or names another session, is not this session lost.
  const quoted = { subtype: "error_during_execution", errors: ['API Error: 400 {"message":"No conversation found with session ID: s1"}'] };
  expect(resultError(quoted, "s1")).not.toBeInstanceOf(LostSessionError);
  expect(resultError(lost, "s2")).not.toBeInstanceOf(LostSessionError);
  // Among several errors, the CLI's own one still counts.
  expect(resultError({ subtype: "error_during_execution", errors: ["warning", "No conversation found with session ID: s1"] }, "s1")).toBeInstanceOf(LostSessionError);
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

// A login that authenticates through a setting (an apiKeyHelper) is a settings FILE: with no setting
// sources read, the flag-settings layer is the only way it reaches Claude Code (verify-claude-fake.ts runs
// it through the real CLI).
test("a settings file is passed to the SDK as flag settings, and only when given", () => {
  expect(claudeQueryOptions({ prompt: "hi", settings: "/acct/settings.json" }).settings).toBe("/acct/settings.json");
  expect(claudeQueryOptions({ prompt: "hi", settings: "/acct/settings.json" }).settingSources).toEqual([]);
  expect("settings" in claudeQueryOptions({ prompt: "hi" })).toBe(false);
});

test("a failed turn's usage rides on its error, and an error without one reads as null", () => {
  const usage = { model: "m", costUsd: 0.42, inputTokens: 1, outputTokens: 2, cacheReadTokens: 3, cacheCreationTokens: 4 };
  const err = withTurnUsage(resultError({ subtype: "error_during_execution", errors: ["boom"] }), usage);
  expect(usageOfFailedTurn(err)).toEqual(usage);
  expect(err.message).toBe(resultError({ subtype: "error_during_execution", errors: ["boom"] }).message);
  expect(JSON.stringify(err)).not.toContain("0.42"); // not enumerable: it does not leak into a serialised error
  expect(usageOfFailedTurn(new Error("plain"))).toBeNull();
  expect(usageOfFailedTurn(null)).toBeNull();
});

