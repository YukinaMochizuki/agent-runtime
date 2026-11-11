import { describe, expect, test } from "bun:test";
import type { LanguageModelV4StreamPart } from "@ai-sdk/provider";
import type { ModelMessage } from "ai";
import { MockLanguageModelV4, convertArrayToReadableStream } from "ai/test";
import { z } from "zod";
import type { ToolBoundary } from "../src/boundary.ts";
import { OpenAICompatBackend, toToolOutput, unansweredCalls, wireToolName, type ConversationStore } from "../src/openai-compat/backend.ts";
import { SUMMARY_PREFIX } from "../src/openai-compat/compaction.ts";
import { ok, type ToolBundle } from "../src/tool.ts";

const usage = { inputTokens: { total: 120, noCache: 100, cacheRead: 20, cacheWrite: 0 }, outputTokens: { total: 7, text: 7, reasoning: undefined } };
const finish = (unified: "stop" | "tool-calls"): LanguageModelV4StreamPart => ({ type: "finish", finishReason: { unified, raw: undefined }, usage });

/** One model call that says `text`. */
const says = (text: string, reasoning?: string): LanguageModelV4StreamPart[] => [
  ...(reasoning
    ? ([
        { type: "reasoning-start", id: "r" },
        { type: "reasoning-delta", id: "r", delta: reasoning },
        { type: "reasoning-end", id: "r" },
      ] as LanguageModelV4StreamPart[])
    : []),
  { type: "text-start", id: "t" },
  { type: "text-delta", id: "t", delta: text },
  { type: "text-end", id: "t" },
  finish("stop"),
];

/** One model call that calls `toolName` with `input`. */
const calls = (toolName: string, input: unknown, id = "call_1"): LanguageModelV4StreamPart[] => [
  { type: "tool-input-start", id, toolName },
  { type: "tool-input-end", id },
  { type: "tool-call", toolCallId: id, toolName, input: JSON.stringify(input) },
  finish("tool-calls"),
];

function scripted(...steps: LanguageModelV4StreamPart[][]): MockLanguageModelV4 {
  return new MockLanguageModelV4({ doStream: steps.map((chunks) => ({ stream: convertArrayToReadableStream(chunks) })) });
}

function memoryStore(): ConversationStore & { all: Map<string, ModelMessage[]>; compactions: number } {
  const all = new Map<string, ModelMessage[]>();
  const store = {
    all,
    compactions: 0,
    load: async (id: string) => [...(all.get(id) ?? [])],
    append: async (id: string, msgs: ModelMessage[]) => void all.set(id, [...(all.get(id) ?? []), ...msgs]),
    compact: async (id: string, msgs: ModelMessage[]) => {
      store.compactions++;
      all.set(id, [...msgs]);
    },
  };
  return store;
}

const echo: ToolBundle = {
  serverName: "ws",
  tools: [{ name: "echo", description: "echo it", inputSchema: { text: z.string() }, handler: async (a: { text: string }) => ok(`echo: ${a.text}`) }],
};

/** The tool results stored for one call id, across every tool message. */
function resultsFor(store: { all: Map<string, ModelMessage[]> }, callId: string): unknown[] {
  return store.all
    .get("conv-1")!
    .flatMap((m) => (m.role === "tool" ? m.content : []))
    .filter((p) => p.type === "tool-result" && p.toolCallId === callId);
}

function backend(model: MockLanguageModelV4, store = memoryStore(), boundary?: ToolBoundary) {
  return new OpenAICompatBackend({ provider: { name: "local", baseURL: "http://x" }, store, defaultModel: "m1", boundary, newId: () => "conv-1", model: () => model });
}

describe("OpenAICompatBackend", () => {
  test("a turn with no tool call streams its prose, tags reasoning, stores the exchange, and reports the last call's usage", async () => {
    const store = memoryStore();
    const tokens: string[] = [];
    let sid = "";
    const r = await backend(scripted(says("Hello.", "thinking it over")), store).run(
      { prompt: "hi", systemPrompt: "Be brief.", tools: [] },
      { onToken: (t, thinking) => tokens.push(`${thinking ? "think" : "text"}:${t}`), onSessionId: (id) => (sid = id) },
    );
    expect(r.text).toBe("Hello.");
    expect(r.backendSessionId).toBe("conv-1");
    expect(sid).toBe("conv-1");
    expect(tokens).toEqual(["think:thinking it over", "text:Hello."]);
    expect(r.usage).toEqual({ model: "m1", costUsd: 0, inputTokens: 100, outputTokens: 7, cacheReadTokens: 20, cacheCreationTokens: 0 });
    const kept = store.all.get("conv-1")!;
    expect(kept.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(kept.some((m) => m.role === "system")).toBe(false); // the system prompt is re-sent, never stored
  });

  test("a tool call runs through the boundary, and what the boundary appends reaches the model's next call", async () => {
    const model = scripted(calls(wireToolName("ws", "echo"), { text: "x" }), says("done"));
    const log: string[] = [];
    const boundary: ToolBoundary = {
      recordStart: (c) => void log.push(`start:${c.recordedName}:${JSON.stringify(c.args)}`),
      recordEnd: (c) => void log.push(`end:${c.recordedName}`),
      append: () => "[inbox] a message for you",
    };
    const seen: string[] = [];
    const r = await backend(model, memoryStore(), boundary).run(
      { prompt: "use it", tools: [echo] },
      { onTool: (name) => seen.push(`tool:${name}`), onToolDraft: (name) => seen.push(`draft:${name}`) },
    );
    expect(r.text).toBe("done");
    expect(seen).toEqual(["draft:ws__echo", "tool:ws__echo"]);
    expect(log).toEqual(['start:echo:{"text":"x"}', "end:echo"]);
    // The second model call carries the tool result, with the appended block, after the call.
    const prompt = JSON.stringify(model.doStreamCalls[1]!.prompt);
    expect(prompt).toContain("echo: x");
    expect(prompt).toContain("[inbox] a message for you");
  });

  test("a continue resumes the stored conversation", async () => {
    const store = memoryStore();
    await backend(scripted(says("first")), store).run({ prompt: "one", tools: [] }, {});
    const model = scripted(says("second"));
    const r = await backend(model, store).continueRun({ prompt: "two", tools: [], backendSessionId: "conv-1" }, {});
    expect(r.text).toBe("second");
    const prompt = JSON.stringify(model.doStreamCalls[0]!.prompt);
    expect(prompt).toContain("one");
    expect(prompt).toContain("first");
    expect(prompt).toContain("two");
    expect(store.all.get("conv-1")!.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
  });

  test("a stop before the next call ends the turn; the tool that ran is still recorded", async () => {
    const ac = new AbortController();
    const stopping: ToolBundle = {
      serverName: "ws",
      tools: [{ name: "stop", description: "", inputSchema: {}, handler: async () => (ac.abort(), ok("stopped")) }],
    };
    const store = memoryStore();
    await expect(backend(scripted(calls("ws__stop", {}), says("never")), store).run({ prompt: "go", tools: [stopping] }, { signal: ac.signal })).rejects.toThrow();
    expect(store.all.get("conv-1")!.map((m) => m.role)).toEqual(["user", "assistant", "tool"]);
  });

  test("a call to a tool that does not exist is answered once, with an error, and nothing runs", async () => {
    const model = scripted(calls("ws__nope", {}), says("ok"));
    const store = memoryStore();
    const r = await backend(model, store).run({ prompt: "go", tools: [echo] }, {});
    expect(r.text).toBe("ok");
    expect(unansweredCalls(store.all.get("conv-1")!)).toEqual([]);
    expect(resultsFor(store, "call_1")).toHaveLength(1);
  });

  test("input that fails the tool's schema never reaches the handler, and the call has exactly one result", async () => {
    let ran = 0;
    const strict: ToolBundle = {
      serverName: "ws",
      tools: [{ name: "echo", description: "", inputSchema: { text: z.string() }, handler: async () => (ran++, ok("ran")) }],
    };
    const store = memoryStore();
    await backend(scripted(calls("ws__echo", { nope: 1 }), says("ok")), store).run({ prompt: "go", tools: [strict] }, {});
    expect(ran).toBe(0);
    expect(resultsFor(store, "call_1")).toHaveLength(1);
  });

  test("a tool that throws answers as an error and the turn goes on, as a throwing MCP handler does on Claude", async () => {
    const throwing: ToolBundle = {
      serverName: "ws",
      tools: [{ name: "boom", description: "", inputSchema: {}, handler: async () => { throw new Error("disk full"); } }],
    };
    const model = scripted(calls("ws__boom", {}), says("recovered"));
    const store = memoryStore();
    const r = await backend(model, store).run({ prompt: "go", tools: [throwing] }, {});
    expect(r.text).toBe("recovered");
    expect(JSON.stringify(model.doStreamCalls[1]!.prompt)).toContain("disk full");
    expect(unansweredCalls(store.all.get("conv-1")!)).toEqual([]);
  });

  test("a stop WHILE a tool runs answers that call and every later one, ends the turn, and the next turn resumes", async () => {
    const ac = new AbortController();
    let secondRan = false;
    const tools: ToolBundle = {
      serverName: "ws",
      tools: [
        // Like `ask` or a long command: the stop rejects the call it is waiting in.
        { name: "wait", description: "", inputSchema: {}, handler: async () => { ac.abort(); throw new DOMException("aborted", "AbortError"); } },
        { name: "after", description: "", inputSchema: {}, handler: async () => ((secondRan = true), ok("x")) },
      ],
    };
    const twoCalls: LanguageModelV4StreamPart[] = [
      { type: "tool-call", toolCallId: "c1", toolName: "ws__wait", input: "{}" },
      { type: "tool-call", toolCallId: "c2", toolName: "ws__after", input: "{}" },
      finish("tool-calls"),
    ];
    const store = memoryStore();
    await expect(backend(scripted(twoCalls), store).run({ prompt: "go", tools: [tools] }, { signal: ac.signal })).rejects.toThrow();
    expect(secondRan).toBe(false);
    const kept = store.all.get("conv-1")!;
    expect(kept.map((m) => m.role)).toEqual(["user", "assistant", "tool"]);
    expect(unansweredCalls(kept)).toEqual([]);
    expect(JSON.stringify(kept[2])).toContain("Interrupted");
    expect(JSON.stringify(kept[2])).toContain("Not run: the turn was stopped");
    // The conversation goes on.
    const next = scripted(says("carrying on"));
    expect((await backend(next, store).continueRun({ prompt: "continue", tools: [tools], backendSessionId: "conv-1" }, {})).text).toBe("carrying on");
  });

  test("a continue answers calls a process that died mid-tool left open, then goes on", async () => {
    const store = memoryStore();
    await store.append("conv-1", [
      { role: "user", content: "go" },
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "lost", toolName: "ws__echo", input: { text: "x" } }] },
    ]);
    const model = scripted(says("back"));
    const r = await backend(model, store).continueRun({ prompt: "where were we?", tools: [echo], backendSessionId: "conv-1" }, {});
    expect(r.text).toBe("back");
    expect(JSON.stringify(model.doStreamCalls[0]!.prompt)).toContain("the turn ended before this call returned");
    expect(unansweredCalls(store.all.get("conv-1")!)).toEqual([]);
  });

  test("an image from a tool goes to a model that sees as a user message after the tool results; not to one that does not", async () => {
    const shooter: ToolBundle = {
      serverName: "ws",
      tools: [{ name: "shot", description: "", inputSchema: {}, handler: async () => ({ content: [{ type: "text", text: "here" }, { type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" }] }) }],
    };
    const run = async (vision: boolean) => {
      const store = memoryStore();
      const b = new OpenAICompatBackend({ provider: { name: "local", baseURL: "http://x" }, store, defaultModel: "m1", vision, newId: () => "conv-1", model: () => scripted(calls("ws__shot", {}), says("seen")) });
      await b.run({ prompt: "look", tools: [shooter] }, {});
      return store.all.get("conv-1")!;
    };
    const seeing = await run(true);
    expect(seeing.map((m) => m.role)).toEqual(["user", "assistant", "tool", "user", "assistant"]);
    expect(JSON.stringify(seeing[2])).not.toContain("iVBORw0KGgo=");
    expect(JSON.stringify(seeing[3])).toContain("iVBORw0KGgo=");
    const blind = await run(false);
    expect(blind.map((m) => m.role)).toEqual(["user", "assistant", "tool", "assistant"]);
    expect(JSON.stringify(blind)).not.toContain("iVBORw0KGgo=");
  });

  test("a prompt of parts opens the turn as one user message, images as file parts; a model that does not see gets a note", async () => {
    const prompt = [
      { type: "text" as const, text: "[photo] look" },
      { type: "image" as const, data: "iVBORw0KGgo=", mimeType: "image/jpeg" },
    ];
    const run = async (vision: boolean) => {
      const store = memoryStore();
      const model = scripted(says("a fox"));
      await new OpenAICompatBackend({ provider: { name: "local", baseURL: "http://x" }, store, defaultModel: "m1", vision, newId: () => "conv-1", model: () => model }).run({ prompt, tools: [] }, {});
      return { stored: store.all.get("conv-1")!, sent: JSON.stringify(model.doStreamCalls[0]!.prompt) };
    };
    const seeing = await run(true);
    expect(seeing.stored[0]).toEqual({
      role: "user",
      content: [
        { type: "text", text: "[photo] look" },
        { type: "file", data: "iVBORw0KGgo=", mediaType: "image/jpeg" },
      ],
    });
    expect(seeing.sent).toContain("iVBORw0KGgo=");
    const blind = await run(false);
    expect(JSON.stringify(blind.stored)).not.toContain("iVBORw0KGgo=");
    expect(blind.sent).toContain("does not take images");
    expect(blind.sent).toContain("[photo] look");
  });

  test("a string prompt is stored as a string, as before", async () => {
    const store = memoryStore();
    await backend(scripted(says("ok")), store).run({ prompt: "hi", tools: [] }, {});
    expect(store.all.get("conv-1")![0]).toEqual({ role: "user", content: "hi" });
  });

  test("blocks the boundary appends (text and an image) follow the tool's own result: text in it, the image in the user message after", async () => {
    const store = memoryStore();
    const boundary: ToolBoundary = { append: () => [{ type: "text", text: "[inbox] a photo arrived" }, { type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" }] };
    const model = scripted(calls("ws__echo", { text: "hi" }), says("seen"));
    await new OpenAICompatBackend({ provider: { name: "local", baseURL: "http://x" }, store, defaultModel: "m1", vision: true, boundary, newId: () => "conv-1", model: () => model }).run({ prompt: "go", tools: [echo] }, {});
    const h = store.all.get("conv-1")!;
    expect(h.map((m) => m.role)).toEqual(["user", "assistant", "tool", "user", "assistant"]);
    const result = JSON.stringify(h[2]);
    expect(result.indexOf("echo: hi")).toBeLessThan(result.indexOf("[inbox] a photo arrived"));
    expect(result).not.toContain("iVBORw0KGgo=");
    expect(JSON.stringify(h[3])).toContain("iVBORw0KGgo=");
    expect(JSON.stringify(model.doStreamCalls[1]!.prompt)).toContain("iVBORw0KGgo=");
  });

  test("effort is sent only at a level the endpoint has a name for", async () => {
    const high = scripted(says("a"));
    await backend(high).run({ prompt: "x", tools: [], effort: "high" }, {});
    expect(JSON.stringify(high.doStreamCalls[0]!.providerOptions)).toContain('"reasoningEffort":"high"');
    const max = scripted(says("a"));
    await backend(max).run({ prompt: "x", tools: [], effort: "max" }, {});
    expect(JSON.stringify(max.doStreamCalls[0]!.providerOptions ?? {})).not.toContain("reasoningEffort");
  });

  test("a model error fails the turn", async () => {
    const model = scripted([{ type: "error", error: new Error("429 rate limited") }]);
    await expect(backend(model).run({ prompt: "x", tools: [] }, {})).rejects.toThrow("429 rate limited");
  });

  test("no model named and no default is an error before any call", async () => {
    const b = new OpenAICompatBackend({ provider: { name: "p", baseURL: "http://x" }, store: memoryStore(), model: () => scripted(says("x")) });
    await expect(b.run({ prompt: "x", tools: [] }, {})).rejects.toThrow(/no model named/);
  });
});

describe("compaction in the loop", () => {
  const big = "output line ".repeat(400); // ~1.2k tokens a result
  const bigTool: ToolBundle = { serverName: "ws", tools: [{ name: "cat", description: "", inputSchema: {}, handler: async () => ok(big) }] };
  /** A model that calls `cat` `n` times, then answers. */
  const steps = (n: number, answer: string) => [...Array.from({ length: n }, (_, i) => calls("ws__cat", {}, `c${answer}${i}`)), says(answer)];
  /**
   * Like `scripted`, but each call reports the prompt it was sent as its input usage (about four characters
   * a token), as a real endpoint would: the loop trusts what the endpoint measured.
   */
  const measuring = (...script: LanguageModelV4StreamPart[][]): MockLanguageModelV4 => {
    let i = 0;
    return new MockLanguageModelV4({
      doStream: async (options) => {
        const tokens = Math.ceil(JSON.stringify(options.prompt).length / 4);
        const chunks = script[i++]!.map((c) =>
          c.type === "finish" ? { ...c, usage: { inputTokens: { total: tokens, noCache: tokens, cacheRead: 0, cacheWrite: 0 }, outputTokens: { total: 10, text: 10, reasoning: undefined } } } : c,
        );
        return { stream: convertArrayToReadableStream(chunks) };
      },
    });
  };
  const compacting = (store: ReturnType<typeof memoryStore>, model: MockLanguageModelV4, contextWindow: number, extra: object = {}) =>
    new OpenAICompatBackend({
      provider: { name: "local", baseURL: "http://x" },
      store,
      defaultModel: "m1",
      newId: () => "conv-1",
      model: () => model,
      compaction: { contextWindow, keepRecentSteps: 2 },
      ...extra,
    });

  test("under the threshold nothing is rewritten: every request's prefix is the previous request, unchanged", async () => {
    const store = memoryStore();
    const model = measuring(...steps(4, "done"));
    await compacting(store, model, 1_000_000).run({ prompt: "read it all", systemPrompt: "sys", tools: [bigTool] }, {});
    expect(store.compactions).toBe(0);
    for (let i = 1; i < model.doStreamCalls.length; i++) {
      const before = model.doStreamCalls[i - 1]!.prompt;
      const now = model.doStreamCalls[i]!.prompt;
      expect(JSON.stringify(now.slice(0, before.length))).toBe(JSON.stringify(before)); // the cache's whole condition
    }
  });

  test("past the threshold the history is compacted once, stored, reported, and the next turn starts from it", async () => {
    const store = memoryStore();
    const reports: unknown[] = [];
    // ~8 results of ~1.2k tokens against a 10k window: the threshold (7.5k) is crossed mid-turn.
    const model = measuring(...steps(8, "first done"));
    await compacting(store, model, 10_000, { onCompaction: (r: unknown) => reports.push(r) }).run({ prompt: "read it all", tools: [bigTool] }, {});
    expect(store.compactions).toBe(1);
    expect(reports).toHaveLength(1);
    expect(reports[0]).toMatchObject({ summarized: false }); // pruning old outputs was enough
    const kept = store.all.get("conv-1")!;
    expect(unansweredCalls(kept)).toEqual([]);
    expect(JSON.stringify(kept)).toContain("omitted to save context");
    // The next turn loads the compacted history: no compaction again just for starting.
    const next = measuring(says("second done"));
    await compacting(store, next, 10_000).continueRun({ prompt: "and now?", tools: [bigTool], backendSessionId: "conv-1" }, {});
    expect(store.compactions).toBe(1);
    expect(JSON.stringify(next.doStreamCalls[0]!.prompt)).toContain("omitted to save context");
  });

  test("when pruning is not enough, the summary model writes the summary", async () => {
    const store = memoryStore();
    // History whose bulk is what pruning cannot shorten: long user messages.
    await store.append("conv-1", [
      { role: "user", content: "u".repeat(30_000) },
      { role: "assistant", content: [{ type: "text", text: "ok" }] },
      { role: "user", content: "v".repeat(30_000) },
      { role: "assistant", content: [{ type: "text", text: "ok" }] },
    ]);
    const summarizer = new MockLanguageModelV4({
      doGenerate: { content: [{ type: "text", text: "what came before" }], finishReason: { unified: "stop", raw: undefined }, usage, warnings: [] },
    });
    const main = measuring(...steps(2, "done"));
    const b = new OpenAICompatBackend({
      provider: { name: "local", baseURL: "http://x" },
      store,
      defaultModel: "m1",
      model: (id) => (id === "cheap" ? summarizer : main),
      compaction: { contextWindow: 16_000, keepRecentSteps: 1, summaryModel: "cheap" },
    });
    await b.continueRun({ prompt: "go on", tools: [bigTool], backendSessionId: "conv-1" }, {});
    expect(summarizer.doGenerateCalls).toHaveLength(1);
    const kept = store.all.get("conv-1")!;
    expect(kept[0]).toEqual({ role: "user", content: `${SUMMARY_PREFIX}\n\nwhat came before` });
    expect(JSON.stringify(kept)).not.toContain("uuuuuuuuuu");
    expect(unansweredCalls(kept)).toEqual([]);
  });

  test("an endpoint that reports no usage still compacts, on the estimate", async () => {
    const store = memoryStore();
    const silent = new MockLanguageModelV4({
      doStream: steps(8, "done").map((chunks) => ({
        stream: convertArrayToReadableStream(
          chunks.map((c) => (c.type === "finish" ? { ...c, usage: { inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined }, outputTokens: { total: undefined, text: undefined, reasoning: undefined } } } : c)),
        ),
      })),
    });
    await compacting(store, silent, 10_000).run({ prompt: "read", tools: [bigTool] }, {});
    expect(store.compactions).toBe(1);
  });

  test("a store without `compact`, or compaction off, never compacts", async () => {
    const store = memoryStore();
    const model = measuring(...steps(8, "done"));
    await compacting(store, model, 10_000, { compaction: false }).run({ prompt: "read", tools: [bigTool] }, {});
    expect(store.compactions).toBe(0);
    const { compact: _dropped, ...plain } = memoryStore();
    const noCompact = { ...plain, compactions: 0 } as unknown as ReturnType<typeof memoryStore>;
    await compacting(noCompact, measuring(...steps(8, "done")), 10_000).run({ prompt: "read", tools: [bigTool] }, {});
    expect(noCompact.all.get("conv-1")!.length).toBeGreaterThan(8); // ran, appended, never compacted
  });

  test("a chat-only session compacts on what the previous turn MEASURED, not only on the estimate", async () => {
    // Short turns, no tools: each turn has a single step, so its first check is the only one.
    const store = memoryStore();
    for (let i = 0; i < 4; i++) {
      await store.append("conv-1", [
        { role: "user", content: `question ${i} ${"q".repeat(3000)}` },
        { role: "assistant", content: [{ type: "text", text: `answer ${i}` }] },
      ]);
    }
    // One model that both answers (stream) and writes the summary (generate).
    const answering = () =>
      new MockLanguageModelV4({
        doStream: { stream: convertArrayToReadableStream(says("ok")) },
        doGenerate: { content: [{ type: "text", text: "the earlier questions" }], finishReason: { unified: "stop", raw: undefined }, usage, warnings: [] },
      });
    const run = (contextTokens?: number) =>
      compacting(store, answering(), 10_000, { compaction: { contextWindow: 10_000, keepRecentSteps: 1 } }).continueRun(
        { prompt: "next", tools: [], backendSessionId: "conv-1", ...(contextTokens ? { contextTokens } : {}) },
        {},
      );
    await run(); // the estimate alone (~3.2k) is under the threshold
    expect(store.compactions).toBe(0);
    await run(9_000); // what the endpoint measured last turn is over it
    expect(store.compactions).toBe(1);
  });

  test("a summary call that fails falls back to the pruned history; the turn runs, and the report says why", async () => {
    const store = memoryStore();
    await store.append("conv-1", [
      { role: "user", content: "u".repeat(30_000) },
      { role: "assistant", content: [{ type: "text", text: "ok" }] },
      { role: "user", content: "v".repeat(30_000) },
      { role: "assistant", content: [{ type: "text", text: "ok" }] },
    ]);
    const broken = new MockLanguageModelV4({
      doGenerate: async () => {
        throw new Error("summary endpoint 500");
      },
    });
    const main = measuring(says("still answered"));
    const reports: Array<{ summaryError?: string }> = [];
    const r = await new OpenAICompatBackend({
      provider: { name: "local", baseURL: "http://x" },
      store,
      defaultModel: "m1",
      model: (id) => (id === "cheap" ? broken : main),
      compaction: { contextWindow: 16_000, keepRecentSteps: 1, summaryModel: "cheap" },
      onCompaction: (rep) => reports.push(rep),
    }).continueRun({ prompt: "go on", tools: [], backendSessionId: "conv-1" }, {});
    expect(r.text).toBe("still answered");
    expect(main.doStreamCalls).toHaveLength(1);
    expect(reports[0]?.summaryError).toContain("summary endpoint 500");
  });

  test("a failed attempt is not retried at every step of the turn", async () => {
    const store = memoryStore();
    let summaries = 0;
    const broken = new MockLanguageModelV4({
      doGenerate: async () => {
        summaries++;
        throw new Error("down");
      },
    });
    await store.append("conv-1", [{ role: "user", content: "u".repeat(60_000) }, { role: "assistant", content: [{ type: "text", text: "ok" }] }]);
    const main = measuring(...steps(4, "done"));
    await new OpenAICompatBackend({
      provider: { name: "local", baseURL: "http://x" },
      store,
      defaultModel: "m1",
      model: (id) => (id === "cheap" ? broken : main),
      compaction: { contextWindow: 16_000, keepRecentSteps: 8, summaryModel: "cheap" },
    }).continueRun({ prompt: "go", tools: [bigTool], backendSessionId: "conv-1" }, {});
    expect(main.doStreamCalls).toHaveLength(5);
    expect(summaries).toBeLessThanOrEqual(1);
  });

  test("a host compactor's result that an endpoint would refuse is not stored, and the turn goes on", async () => {
    const store = memoryStore();
    const reports: Array<{ rejected?: string }> = [];
    const orphan: ModelMessage[] = [{ role: "tool", content: [{ type: "tool-result", toolCallId: "ghost", toolName: "t", output: { type: "text", value: "x" } }] }];
    const r = await compacting(store, measuring(...steps(8, "done")), 10_000, {
      compaction: { contextWindow: 10_000, keepRecentSteps: 2, compactor: async () => orphan },
      onCompaction: (rep: { rejected?: string }) => reports.push(rep),
    }).run({ prompt: "read", tools: [bigTool] }, {});
    expect(r.text).toBe("done");
    expect(store.compactions).toBe(0);
    expect(reports[0]?.rejected).toMatch(/no earlier message calls/);
  });

  test("across turns, and after a reload from the store, each request's prefix is the last one's, unchanged", async () => {
    const store = memoryStore();
    const first = measuring(...steps(2, "one"));
    await compacting(store, first, 1_000_000).run({ prompt: "first", systemPrompt: "sys", tools: [bigTool] }, {});
    const second = measuring(...steps(2, "two"));
    await compacting(store, second, 1_000_000).continueRun({ prompt: "second", systemPrompt: "sys", tools: [bigTool], backendSessionId: "conv-1" }, {});
    const last = first.doStreamCalls.at(-1)!.prompt;
    const next = second.doStreamCalls[0]!.prompt;
    expect(JSON.stringify(next.slice(0, last.length))).toBe(JSON.stringify(last));
  });
});

describe("tool results as the model sees them", () => {
  test("text and errors are text; an image is named in the text and handed back only for a model that sees", () => {
    expect(toToolOutput(ok("hi"))).toEqual({ output: { type: "text", value: "hi" }, images: [] });
    expect(toToolOutput({ content: [{ type: "text", text: "bad" }], isError: true }).output).toEqual({ type: "error-text", value: "bad" });
    const shot = { content: [{ type: "text" as const, text: "see" }, { type: "image" as const, data: "AAAA", mimeType: "image/png" }] };
    const seen = toToolOutput(shot, true);
    expect(seen.images).toEqual([{ data: "AAAA", mediaType: "image/png" }]);
    expect(seen.output).toEqual({ type: "text", value: "see\n[image 1 (image/png): sent in the next message]" });
    const blind = toToolOutput(shot, false);
    expect(blind.images).toEqual([]);
    expect(JSON.stringify(blind.output)).not.toContain("AAAA"); // never base64 as text
    expect(JSON.stringify(blind.output)).toContain("does not take images");
  });

  test("unansweredCalls finds the calls no tool message answers", () => {
    const history: ModelMessage[] = [
      { role: "user", content: "go" },
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "a", toolName: "t", input: {} }, { type: "tool-call", toolCallId: "b", toolName: "t", input: {} }] },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "a", toolName: "t", output: { type: "text", value: "ok" } }] },
    ];
    expect(unansweredCalls(history)).toEqual([{ toolCallId: "b", toolName: "t" }]);
  });

  test("tool names are `<server>__<tool>`, cleaned to what an OpenAI function name allows", () => {
    expect(wireToolName("ws", "bash_box")).toBe("ws__bash_box");
    expect(wireToolName("local.chrome", "click")).toBe("local_chrome__click");
    expect(wireToolName("s", "x".repeat(100)).length).toBe(64);
  });
});
