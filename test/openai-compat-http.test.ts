import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { z } from "zod";
import { OpenAICompatBackend, type ConversationStore } from "../src/openai-compat/backend.ts";
import { listOpenAICompatModels } from "../src/openai-compat/probe.ts";
import { ok, type ToolBundle } from "../src/tool.ts";

/**
 * The driver against a FAKE OpenAI-compatible HTTP server, through the real `@ai-sdk/openai-compatible`
 * provider: what the mock-model tests cannot reach is the wire itself (the request body, SSE chunks, a
 * streamed tool call's arguments, the tool result going back as a `tool` message). The fake answers the
 * first request with a tool call and every later one with text that quotes what it was sent.
 */

type ChatRequest = { model: string; messages: Array<{ role: string; content?: unknown; tool_call_id?: string; tool_calls?: unknown[] }>; tools?: Array<{ function: { name: string } }>; reasoning_effort?: string; stream?: boolean };

const requests: ChatRequest[] = [];
let server: ReturnType<typeof Bun.serve>;

const sse = (chunks: unknown[]): Response =>
  new Response(chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") + "data: [DONE]\n\n", { headers: { "content-type": "text/event-stream" } });

const chunk = (delta: unknown, finish: string | null = null, usage?: unknown) => ({
  id: "chatcmpl-1",
  object: "chat.completion.chunk",
  created: 1,
  model: "fake-1",
  choices: [{ index: 0, delta, finish_reason: finish }],
  ...(usage ? { usage } : {}),
});

beforeAll(() => {
  server = Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === "/v1/models") {
        if (req.headers.get("authorization") !== "Bearer sk-fake") return new Response("no", { status: 401 });
        return Response.json({ object: "list", data: [{ id: "fake-1" }, { id: "fake-2" }] });
      }
      if (url.pathname !== "/v1/chat/completions") return new Response("not found", { status: 404 });
      const body = (await req.json()) as ChatRequest;
      requests.push(body);
      const last = body.messages[body.messages.length - 1]!;
      if (last.role === "user" && typeof last.content === "string" && body.tools?.some((t) => t.function.name === "ws__shot")) {
        return sse([
          chunk({ role: "assistant", content: null, tool_calls: [{ index: 0, id: "call_s", type: "function", function: { name: "ws__shot", arguments: "{}" } }] }),
          chunk({}, "tool_calls"),
        ]);
      }
      if (last.role === "user" && body.tools?.some((t) => t.function.name === "ws__echo")) {
        // A tool call whose arguments arrive in two pieces, as real endpoints stream them.
        return sse([
          chunk({ role: "assistant", content: null, tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "ws__echo", arguments: '{"text":' } }] }),
          chunk({ tool_calls: [{ index: 0, function: { arguments: '"hello"}' } }] }),
          chunk({}, "tool_calls", { prompt_tokens: 40, completion_tokens: 5, total_tokens: 45 }),
        ]);
      }
      const said = last.role === "tool" ? `saw: ${JSON.stringify(last.content)}` : "plain answer";
      return sse([chunk({ role: "assistant", content: said.slice(0, 5) }), chunk({ content: said.slice(5) }), chunk({}, "stop", { prompt_tokens: 60, completion_tokens: 7, total_tokens: 67 })]);
    },
  });
});

afterAll(() => server.stop(true));

function memoryStore(): ConversationStore {
  const all = new Map<string, ModelMessage[]>();
  return { load: async (id) => [...(all.get(id) ?? [])], append: async (id, msgs) => void all.set(id, [...(all.get(id) ?? []), ...msgs]) };
}

const echo: ToolBundle = {
  serverName: "ws",
  tools: [{ name: "echo", description: "echo it", inputSchema: { text: z.string() }, handler: async (a: { text: string }) => ok(`echo: ${a.text}`) }],
};

describe("OpenAICompatBackend over HTTP", () => {
  test("a streamed tool call runs, its result goes back as a tool message, and the answer streams", async () => {
    requests.length = 0;
    const provider = { name: "fake", baseURL: `http://localhost:${server.port}/v1`, apiKey: "sk-fake" };
    const b = new OpenAICompatBackend({ provider, store: memoryStore(), defaultModel: "fake-1", boundary: { append: () => "[inbox] hi" } });
    const tokens: string[] = [];
    const tools: Array<[string, unknown]> = [];
    const r = await b.run({ prompt: "please echo", systemPrompt: "You are terse.", tools: [echo], effort: "medium" }, { onToken: (t) => tokens.push(t), onTool: (n, i) => tools.push([n, i]) });

    expect(tools).toEqual([["ws__echo", { text: "hello" }]]);
    expect(requests).toHaveLength(2);
    // First request: system prompt, the user's message, the tool declared, the effort sent, streaming on.
    expect(requests[0]!.messages.map((m) => m.role)).toEqual(["system", "user"]);
    expect(requests[0]!.tools!.map((t) => t.function.name)).toEqual(["ws__echo"]);
    expect(requests[0]!.reasoning_effort).toBe("medium");
    expect(requests[0]!.stream).toBe(true);
    // Second request: the assistant's call and the tool's result (with the appended block) went back.
    const second = requests[1]!.messages;
    expect(second.map((m) => m.role)).toEqual(["system", "user", "assistant", "tool"]);
    expect(second[3]!.tool_call_id).toBe("call_1");
    expect(JSON.stringify(second[3]!.content)).toContain("echo: hello");
    expect(JSON.stringify(second[3]!.content)).toContain("[inbox] hi");
    // The answer streamed in pieces and is the turn's text; usage is the LAST request's.
    expect(tokens.length).toBeGreaterThan(1);
    expect(r.text).toStartWith("saw: ");
    expect(r.usage.inputTokens).toBe(60);
    expect(r.usage.outputTokens).toBe(7);
  });

  test("an image a tool returns reaches a model that sees as image_url in a user message; the tool message is text", async () => {
    requests.length = 0;
    const shot: ToolBundle = {
      serverName: "ws",
      tools: [{ name: "shot", description: "take one", inputSchema: {}, handler: async () => ({ content: [{ type: "text", text: "here" }, { type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" }] }) }],
    };
    const provider = { name: "fake", baseURL: `http://localhost:${server.port}/v1`, apiKey: "sk-fake" };
    await new OpenAICompatBackend({ provider, store: memoryStore(), defaultModel: "fake-1", vision: true }).run({ prompt: "look", tools: [shot] }, {});
    const sent = requests[1]!.messages;
    expect(sent.map((m) => m.role)).toEqual(["user", "assistant", "tool", "user"]);
    expect(typeof sent[2]!.content).toBe("string");
    expect(sent[2]!.content as string).not.toContain("iVBORw0KGgo=");
    const parts = sent[3]!.content as Array<{ type: string; image_url?: { url: string } }>;
    expect(parts.some((p) => p.type === "image_url" && p.image_url!.url.startsWith("data:image/png;base64,iVBORw0KGgo="))).toBe(true);
  });

  test("with cache breakpoints on, the system prompt and the last message carry cache_control on the wire; off, nothing does", async () => {
    const provider = { name: "fake", baseURL: `http://localhost:${server.port}/v1`, apiKey: "sk-fake" };
    requests.length = 0;
    await new OpenAICompatBackend({ provider: { ...provider, cacheBreakpoints: true }, store: memoryStore(), defaultModel: "fake-1" }).run({ prompt: "hello", systemPrompt: "Be terse.", tools: [] }, {});
    const marked = requests[0]!.messages;
    expect(marked[0]!.content).toEqual([{ type: "text", text: "Be terse.", cache_control: { type: "ephemeral" } }]);
    expect(JSON.stringify(marked[marked.length - 1]!.content)).toContain("cache_control");
    requests.length = 0;
    await new OpenAICompatBackend({ provider, store: memoryStore(), defaultModel: "fake-1" }).run({ prompt: "hello", systemPrompt: "Be terse.", tools: [] }, {});
    expect(JSON.stringify(requests[0]!.messages)).not.toContain("cache_control");
  });

  test("an endpoint's error status fails the turn", async () => {
    const b = new OpenAICompatBackend({ provider: { name: "fake", baseURL: `http://localhost:${server.port}/nope` }, store: memoryStore(), defaultModel: "fake-1" });
    await expect(b.run({ prompt: "x", tools: [] }, {})).rejects.toThrow();
  });

  test("the models an endpoint lists, with its key sent", async () => {
    const provider = { name: "fake", baseURL: `http://localhost:${server.port}/v1`, apiKey: "sk-fake" };
    expect(await listOpenAICompatModels(provider)).toEqual([
      { id: "fake/fake-1", label: "fake-1", provider: "fake", model: "fake-1", reported: {} },
      { id: "fake/fake-2", label: "fake-2", provider: "fake", model: "fake-2", reported: {} },
    ]);
    await expect(listOpenAICompatModels({ ...provider, apiKey: "wrong" })).rejects.toThrow(/401/);
  });
});
