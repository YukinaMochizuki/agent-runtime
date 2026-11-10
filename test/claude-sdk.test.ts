import { describe, expect, test } from "bun:test";
import { claudeQueryOptions, claudeSubprocessEnv } from "../src/claude/sdk.ts";

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
