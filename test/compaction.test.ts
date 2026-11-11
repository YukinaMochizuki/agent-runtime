import { describe, expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { unansweredCalls } from "../src/openai-compat/backend.ts";
import { withCacheBreakpoints } from "../src/openai-compat/cache.ts";
import { malformedHistory } from "../src/openai-compat/backend.ts";
import { defaultCompactor, estimateTokens, prune, pruneAllButLastStep, splitRecent, SUMMARY_PREFIX, TOOL_IMAGES_PREFIX, transcriptOf, turnStarts } from "../src/openai-compat/compaction.ts";

const user = (text: string): ModelMessage => ({ role: "user", content: text });
const said = (text: string): ModelMessage => ({ role: "assistant", content: [{ type: "text", text }] });
const call = (id: string, tool = "bash"): ModelMessage => ({ role: "assistant", content: [{ type: "tool-call", toolCallId: id, toolName: tool, input: { cmd: id } }] });
const result = (id: string, value: string, tool = "bash"): ModelMessage => ({ role: "tool", content: [{ type: "tool-result", toolCallId: id, toolName: tool, output: { type: "text", value } }] });
const images: ModelMessage = { role: "user", content: [{ type: "text", text: `${TOOL_IMAGES_PREFIX} (1):` }, { type: "file", data: "AAAA".repeat(5000), mediaType: "image/png" }] };

/** One turn of an agent at work: the request, `steps` tool calls with long outputs, an answer. */
function turn(ask: string, steps: number, prefix: string): ModelMessage[] {
  const out: ModelMessage[] = [user(ask)];
  for (let i = 0; i < steps; i++) out.push(call(`${prefix}${i}`), result(`${prefix}${i}`, `output ${prefix}${i} `.repeat(200)));
  out.push(said(`done with ${ask}`));
  return out;
}

describe("where history may be cut", () => {
  test("a tool's image follow-up belongs to its turn, not a new one", () => {
    expect(turnStarts([user("a"), call("1"), result("1", "x"), images, said("ok"), user("b")])).toEqual([0, 5]);
  });

  test("the last N steps are kept exactly, and the cut never separates a call from its result", () => {
    const h = [...turn("first", 6, "a"), ...turn("second", 6, "b")];
    const { older, recent, opening } = splitRecent(h, 4);
    expect(recent[0]!.role).toBe("assistant");
    expect(recent.filter((m) => m.role === "assistant")).toHaveLength(4);
    expect(unansweredCalls(older)).toEqual([]);
    expect(unansweredCalls([...older, ...recent])).toEqual([]);
    expect(opening).toEqual(user("second")); // the cut fell inside the second turn
  });

  test("a step that opens its turn takes the turn's user message with it", () => {
    const h = [...turn("first", 2, "a"), user("second"), call("b0"), result("b0", "x"), said("ok")];
    const { recent, opening } = splitRecent(h, 2);
    expect(recent[0]).toEqual(user("second"));
    expect(opening).toBeNull();
  });

  test("a request right after a stopped turn's tool results is still a turn: the current request is the one kept word for word", async () => {
    // A stopped (or repaired) turn ends on tool results, so the next request follows a tool message.
    const h: ModelMessage[] = [
      user("u".repeat(40_000)),
      said("ok"),
      user("old request"),
      call("s1"),
      result("s1", "stopped here"),
      user("THE CURRENT REQUEST"),
      ...[1, 2, 3, 4].flatMap((i) => [call(`c${i}`), result(`c${i}`, "r")]),
    ];
    expect(splitRecent(h, 3).opening).toEqual(user("THE CURRENT REQUEST"));
    const next = await defaultCompactor({ messages: h, usedTokens: estimateTokens(h), contextWindow: estimateTokens(h), target: 0.1, keepRecentSteps: 3, summarize: async () => "summary" });
    expect(next![1]).toEqual(user("THE CURRENT REQUEST"));
    expect(JSON.stringify(next)).not.toContain("old request");
  });

  test("a prompt carrying a photo opens a turn (whichever part comes first); as the kept opening, its image is pruned and its text kept", async () => {
    const photo: ModelMessage = { role: "user", content: [{ type: "text", text: "THE PHOTO REQUEST" }, { type: "file", data: "BBBB".repeat(5000), mediaType: "image/jpeg" }] };
    const imageFirst: ModelMessage = { role: "user", content: [{ type: "file", data: "CCCC", mediaType: "image/jpeg" }, { type: "text", text: "caption" }] };
    expect(turnStarts([user("a"), said("ok"), photo, call("1"), result("1", "x"), images, said("ok"), imageFirst])).toEqual([0, 2, 7]);
    expect(estimateTokens([photo])).toBeLessThan(2_000); // an image at its flat rate, not its base64 length
    const h: ModelMessage[] = [user("u".repeat(40_000)), said("ok"), photo, ...[1, 2, 3, 4].flatMap((i) => [call(`c${i}`), result(`c${i}`, "r")])];
    expect(splitRecent(h, 3).opening).toEqual(photo);
    const next = await defaultCompactor({ messages: h, usedTokens: estimateTokens(h), contextWindow: estimateTokens(h), target: 0.1, keepRecentSteps: 3, summarize: async () => "summary" });
    expect(JSON.stringify(next![1])).toContain("THE PHOTO REQUEST");
    expect(JSON.stringify(next![1])).toContain("an earlier image, omitted");
    expect(JSON.stringify(next)).not.toContain("BBBB");
  });

  test("Chinese text is not undercounted fourfold", () => {
    expect(estimateTokens([user("中".repeat(1000))])).toBeGreaterThanOrEqual(900);
    expect(estimateTokens([user("a".repeat(1000))])).toBeLessThanOrEqual(260);
  });

  test("with no more steps than are kept, nothing may be touched", () => {
    const h = turn("only", 3, "a");
    expect(splitRecent(h, 8).older).toEqual([]);
  });
});

describe("the two stages", () => {
  test("pruning shortens long tool outputs and images in place; calls and short outputs stay", () => {
    const h = [user("go"), call("1"), result("1", "x".repeat(5000)), call("2"), result("2", "short"), images];
    const p = prune(h);
    expect(unansweredCalls(p)).toEqual([]);
    expect(JSON.stringify(p[2])).toContain("omitted to save context");
    expect(JSON.stringify(p[2]).length).toBeLessThan(600);
    expect(p[4]).toEqual(h[4]!);
    expect(JSON.stringify(p[5])).not.toContain("AAAA");
    expect(p[1]).toEqual(h[1]!);
  });

  test("when pruning is enough, no summary is asked for", async () => {
    const h = [...turn("first", 10, "a"), ...turn("second", 10, "b")];
    let asked = 0;
    const next = await defaultCompactor({ messages: h, usedTokens: estimateTokens(h), contextWindow: estimateTokens(h), target: 0.5, keepRecentSteps: 4, summarize: async () => (asked++, "x") });
    expect(asked).toBe(0);
    expect(next).not.toBeNull();
    expect(estimateTokens(next!)).toBeLessThan(estimateTokens(h) * 0.5);
    expect(unansweredCalls(next!)).toEqual([]);
  });

  test("when it is not, the older steps become one summary, then the request being worked on, then the recent steps", async () => {
    // Long user messages: what pruning cannot shorten.
    const h = [user("a".repeat(40_000)), said("ok"), user("b".repeat(40_000)), said("ok"), ...turn("current request", 10, "c")];
    let transcript = "";
    const next = await defaultCompactor({ messages: h, usedTokens: estimateTokens(h), contextWindow: estimateTokens(h), target: 0.3, keepRecentSteps: 4, summarize: async (t) => ((transcript = t), "the summary") });
    expect(transcript).toContain("USER: aaaa");
    expect(next![0]).toEqual(user(`${SUMMARY_PREFIX}\n\nthe summary`));
    expect(next![1]).toEqual(user("current request")); // word for word
    expect(next!.slice(2).filter((m) => m.role === "assistant")).toHaveLength(4);
    expect(unansweredCalls(next!)).toEqual([]);
    expect(estimateTokens(next!)).toBeLessThan(estimateTokens(h) * 0.3);
  });

  test("a few huge steps, all recent: their tool output is pruned too, all but the latest step's", async () => {
    const h = [user("read the logs"), ...[1, 2, 3, 4, 5].flatMap((i) => [call(`c${i}`), result(`c${i}`, "L".repeat(160_000))])];
    const next = await defaultCompactor({ messages: h, usedTokens: estimateTokens(h), contextWindow: 128_000, target: 0.5, keepRecentSteps: 8, summarize: async () => "" });
    expect(next).not.toBeNull();
    expect(estimateTokens(next!)).toBeLessThan(estimateTokens(h) / 3);
    expect(next!.at(-1)).toEqual(h.at(-1)!); // the latest step's result, untouched
    expect(malformedHistory(next!)).toBeNull();
    expect(pruneAllButLastStep([user("x")])).toEqual([user("x")]);
  });

  test("a summary that could not be had leaves what pruning got", async () => {
    const h = [user("a".repeat(40_000)), said("ok"), ...turn("current", 6, "c")];
    const next = await defaultCompactor({ messages: h, usedTokens: estimateTokens(h), contextWindow: estimateTokens(h), target: 0.1, keepRecentSteps: 2, summarize: async () => "" });
    expect(next).not.toBeNull();
    expect(JSON.stringify(next)).not.toContain(SUMMARY_PREFIX);
    expect(malformedHistory(next!)).toBeNull();
  });

  test("malformedHistory refuses orphan results, results not right after their call, and unanswered calls", () => {
    expect(malformedHistory([user("a"), call("1"), result("1", "x")])).toBeNull();
    expect(malformedHistory([user("a"), result("1", "x"), user("b")])).toMatch(/no earlier message calls/);
    expect(malformedHistory([call("1"), user("between"), result("1", "x")])).toMatch(/does not follow its call directly/);
    expect(malformedHistory([user("a"), call("1")])).toMatch(/without a result/);
  });

  test("the transcript a summary is written from names the calls and their results", () => {
    const t = transcriptOf([user("deploy it"), call("1", "bash_box"), result("1", "exit 0", "bash_box")]);
    expect(t).toContain("USER: deploy it");
    expect(t).toContain('ASSISTANT called bash_box({"cmd":"1"})');
    expect(t).toContain("TOOL bash_box: exit 0");
  });
});

describe("cache breakpoints on the wire", () => {
  test("the system prompt, the summary and the last message are marked, on a content part", () => {
    const body = {
      model: "m",
      messages: [
        { role: "system", content: "be terse" },
        { role: "user", content: `${SUMMARY_PREFIX}\n\nwhat happened` },
        { role: "user", content: "go on" },
        { role: "assistant", content: null, tool_calls: [{ id: "c1" }] },
        { role: "tool", tool_call_id: "c1", content: "result" },
      ],
    };
    const out = withCacheBreakpoints(body) as { messages: Array<{ content: unknown }> };
    const marked = out.messages.map((m) => JSON.stringify(m.content).includes("cache_control"));
    expect(marked).toEqual([true, true, false, false, true]);
    expect(out.messages[0]!.content).toEqual([{ type: "text", text: "be terse", cache_control: { type: "ephemeral" } }]);
    expect(body.messages[0]!.content).toBe("be terse"); // the original is not mutated
  });

  test("a message that already has parts gets the marker on its last part", () => {
    const out = withCacheBreakpoints({ messages: [{ role: "user", content: [{ type: "text", text: "a" }, { type: "image_url", image_url: { url: "x" } }] }] }) as {
      messages: Array<{ content: Array<Record<string, unknown>> }>;
    };
    expect(out.messages[0]!.content[0]!.cache_control).toBeUndefined();
    expect(out.messages[0]!.content[1]!.cache_control).toEqual({ type: "ephemeral" });
  });
});
