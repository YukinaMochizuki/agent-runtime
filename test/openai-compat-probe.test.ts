import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { DEFAULT_PROBE_MAX_OUTPUT_TOKENS, listOpenAICompatModels, probeModel, reportedModel } from "../src/openai-compat/probe.ts";

/**
 * `/models` and the probe against FAKE OpenAI-compatible HTTP servers, through the real AI SDK provider, one
 * per behaviour an endpoint really shows: a vLLM that calls tools, an OpenRouter with rich model metadata, a
 * model that answers in text instead of calling the tool, a vLLM that refuses `tool_choice: auto`, an
 * endpoint that 401s, and one that is slow (a cold start).
 */

type ChatRequest = { model: string; messages: Array<{ role: string; content?: unknown }>; tools?: Array<{ function: { name: string } }>; stream?: boolean };

const requests: Array<{ path: string; body: ChatRequest; auth: string | null; headers: Headers }> = [];
let server: ReturnType<typeof Bun.serve>;
let base = "";
let chatHits = 0; // every chat request, refused ones included
let lastMaxTokens: number | undefined;

const chunk = (delta: unknown, finish: string | null = null) => ({ id: "c1", object: "chat.completion.chunk", created: 1, model: "m", choices: [{ index: 0, delta, finish_reason: finish }] });
const sse = (chunks: unknown[]): Response => new Response(chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") + "data: [DONE]\n\n", { headers: { "content-type": "text/event-stream" } });
const toolCallStream = () =>
  sse([chunk({ role: "assistant", content: null, tool_calls: [{ index: 0, id: "call_p", type: "function", function: { name: "ping", arguments: "{}" } }] }), chunk({}, "tool_calls")]);
const textStream = (t: string) => sse([chunk({ role: "assistant", content: t }), chunk({}, "stop")]);
const completion = (t: string) => Response.json({ id: "c2", object: "chat.completion", created: 1, model: "m", choices: [{ index: 0, message: { role: "assistant", content: t }, finish_reason: "stop" }] });
const hasImage = (b: ChatRequest) => JSON.stringify(b.messages).includes("image_url");

beforeAll(() => {
  server = Bun.serve({
    port: 0,
    idleTimeout: 30,
    async fetch(req) {
      const url = new URL(req.url);
      const kind = url.pathname.split("/")[1]; // /<kind>/v1/<rest>
      const rest = url.pathname.split("/").slice(3).join("/");
      if (rest === "chat/completions") chatHits++;
      if (kind === "auth" || (kind === "vllm" && req.headers.get("authorization") !== "Bearer sk-vllm")) return Response.json({ error: { message: "Invalid API key", type: "auth" } }, { status: 401 });
      if (rest === "models") {
        if (kind === "vllm") return Response.json({ object: "list", data: [{ id: "Qwen/Qwen3-32B", object: "model", created: 1, owned_by: "vllm", root: "Qwen/Qwen3-32B", parent: null, max_model_len: 32768 }] });
        if (kind === "or")
          return Response.json({
            data: [
              {
                id: "openai/gpt-5",
                name: "OpenAI: GPT-5",
                context_length: 400000,
                architecture: { modality: "text+image->text", input_modalities: ["text", "image", "file"], output_modalities: ["text"] },
                top_provider: { context_length: 400000, max_completion_tokens: 128000 },
                supported_parameters: ["max_tokens", "tools", "tool_choice", "reasoning", "include_reasoning"],
              },
              { id: "meta/tiny", name: "Tiny", context_length: 8192, architecture: { input_modalities: ["text"] }, supported_parameters: ["max_tokens", "temperature"] },
              { id: "bare" },
              { name: "no id" },
            ],
          });
        return Response.json({ data: [] });
      }
      if (rest !== "chat/completions") return new Response("not found", { status: 404 });
      const body = (await req.json()) as ChatRequest;
      requests.push({ path: url.pathname, body, auth: req.headers.get("authorization"), headers: req.headers });
      if (kind === "slow") await Bun.sleep(400);
      // OpenRouter, a model with no route that takes tools or pictures: 404s worded exactly as it words them.
      if (kind === "ornoroute") {
        if (hasImage(body)) return Response.json({ error: { message: "No endpoints found that support image input", code: 404 } }, { status: 404 });
        return Response.json({ error: { message: "No endpoints found that support tool use. To learn more about provider routing, visit: https://openrouter.ai/docs/provider-routing", code: 404 } }, { status: 404 });
      }
      if (hasImage(body)) {
        if (body.model === "meta/tiny") return Response.json({ error: { message: "this model does not support image input", code: 400 } }, { status: 400 });
        return completion("Orange");
      }
      // A reasoning model that thinks through its whole budget: reasoning deltas only, then `length`.
      if (kind === "thinker") {
        lastMaxTokens = (body as { max_tokens?: number }).max_tokens;
        return sse([chunk({ role: "assistant", content: null, reasoning_content: "Let me think about which tool to call. " }), chunk({ reasoning_content: "The user wants ping, so" }), chunk({}, "length")]);
      }
      if (kind === "missing") return Response.json({ error: { message: `The model \`${body.model}\` does not exist.` } }, { status: 404 });
      if (kind === "noauto")
        return Response.json({ object: "error", message: '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set', type: "BadRequestError", param: null, code: 400 }, { status: 400 });
      if (kind === "text") return textStream('<tool_call>{"name": "ping", "arguments": {}}</tool_call>');
      return toolCallStream();
    },
  });
  base = `http://localhost:${server.port}`;
});

afterAll(() => server.stop(true));

describe("listOpenAICompatModels", () => {
  test("vLLM: the context window from max_model_len, everything else unknown", async () => {
    const models = await listOpenAICompatModels({ name: "modal", baseURL: `${base}/vllm/v1`, apiKey: "sk-vllm" });
    expect(models).toEqual([{ id: "modal/Qwen/Qwen3-32B", label: "Qwen/Qwen3-32B", provider: "modal", model: "Qwen/Qwen3-32B", reported: { contextWindow: 32768 } }]);
    // Unknown is absent, never false.
    expect("vision" in models[0]!.reported).toBe(false);
    expect("tools" in models[0]!.reported).toBe(false);
  });

  test("OpenRouter: context, vision, tools and reasoning as reported; a list without an entry reads as no", async () => {
    const models = await listOpenAICompatModels({ name: "or", baseURL: `${base}/or/v1/` });
    expect(models.map((m) => m.id)).toEqual(["or/openai/gpt-5", "or/meta/tiny", "or/bare"]);
    expect(models[0]!.reported).toEqual({
      contextWindow: 400000,
      maxOutputTokens: 128000,
      inputModalities: ["text", "image", "file"],
      vision: true,
      tools: true,
      reasoning: true,
      name: "OpenAI: GPT-5",
    });
    expect(models[1]!.reported).toEqual({ contextWindow: 8192, inputModalities: ["text"], vision: false, tools: false, reasoning: false, name: "Tiny" });
    expect(models[2]!.reported).toEqual({});
  });

  test("a 401 throws with the status", async () => {
    await expect(listOpenAICompatModels({ name: "x", baseURL: `${base}/auth/v1` })).rejects.toThrow(/401/);
  });

  test("reportedModel ignores values of the wrong type", () => {
    expect(reportedModel({ id: "m", max_model_len: "big", context_length: -1, architecture: { input_modalities: "image" }, supported_parameters: [1] })).toEqual({});
    expect(reportedModel({ id: "m", context_window: 131072 })).toEqual({ contextWindow: 131072 });
  });
});

describe("probeModel", () => {
  test("a model that calls the tool: ok, timed, one streamed request offering ping, the key and headers sent", async () => {
    requests.length = 0;
    const p = await probeModel({ name: "modal", baseURL: `${base}/vllm/v1`, apiKey: "sk-vllm", headers: { "Modal-Key": "mk" } }, "Qwen/Qwen3-32B");
    expect(p.toolCall).toBe("ok");
    expect(p.error).toBeUndefined();
    expect(p.latencyMs).toBeGreaterThanOrEqual(0);
    expect(p.firstTokenMs).toBeDefined();
    expect(p.finishReason).toBe("tool-calls");
    expect(requests).toHaveLength(1);
    expect(requests[0]!.body.stream).toBe(true);
    expect(requests[0]!.body.model).toBe("Qwen/Qwen3-32B");
    expect(requests[0]!.body.tools!.map((t) => t.function.name)).toEqual(["ping"]);
    expect(requests[0]!.auth).toBe("Bearer sk-vllm");
    expect(requests[0]!.headers.get("modal-key")).toBe("mk");
  });

  test("a model that answers in text instead: text-instead, with what it wrote and the vLLM hint", async () => {
    const p = await probeModel({ name: "t", baseURL: `${base}/text/v1` }, "m");
    expect(p.toolCall).toBe("text-instead");
    expect(p.text).toContain("<tool_call>");
    expect(p.hint).toContain("--enable-auto-tool-choice");
    expect(p.error).toBeUndefined();
  });

  test("a vLLM that refuses tool_choice auto: an error of kind tools-unsupported, with the hint", async () => {
    const p = await probeModel({ name: "n", baseURL: `${base}/noauto/v1` }, "m");
    expect(p.toolCall).toBe("error");
    expect(p.error!.kind).toBe("tools-unsupported");
    expect(p.error!.status).toBe(400);
    expect(p.error!.message).toContain("enable-auto-tool-choice");
    expect(p.hint).toContain("--tool-call-parser");
  });

  test("a 401: an auth error, once (no retries), and no vision probe after it", async () => {
    chatHits = 0;
    const p = await probeModel({ name: "a", baseURL: `${base}/auth/v1`, apiKey: "sk-secret-value" }, "m", { vision: true });
    expect(p.toolCall).toBe("error");
    expect(p.error!.kind).toBe("auth");
    expect(p.error!.status).toBe(401);
    expect(p.error!.message).not.toContain("sk-secret-value");
    expect(p.vision).toBeUndefined();
    expect(chatHits).toBe(1);
  });

  test("an unknown model: not-found", async () => {
    const p = await probeModel({ name: "x", baseURL: `${base}/missing/v1` }, "nope");
    expect(p.error!.kind).toBe("not-found");
  });

  test("nothing listening: unreachable", async () => {
    const dead = Bun.serve({ port: 0, fetch: () => new Response() });
    const port = dead.port;
    dead.stop(true);
    const p = await probeModel({ name: "x", baseURL: `http://localhost:${port}/v1` }, "m", { timeoutMs: 5000 });
    expect(p.toolCall).toBe("error");
    expect(p.error!.kind).toBe("unreachable");
  });

  test("a slow endpoint inside the timeout: ok, and the latency shows the wait", async () => {
    const p = await probeModel({ name: "s", baseURL: `${base}/slow/v1` }, "m", { timeoutMs: 5000 });
    expect(p.toolCall).toBe("ok");
    expect(p.latencyMs).toBeGreaterThanOrEqual(350);
  });

  test("a slow endpoint past the timeout: timeout", async () => {
    const p = await probeModel({ name: "s", baseURL: `${base}/slow/v1` }, "m", { timeoutMs: 100 });
    expect(p.toolCall).toBe("error");
    expect(p.error!.kind).toBe("timeout");
    expect(p.latencyMs).toBeLessThan(350);
  });

  test("the caller's signal: aborted, not timeout", async () => {
    const ctl = new AbortController();
    setTimeout(() => ctl.abort(), 50);
    const p = await probeModel({ name: "s", baseURL: `${base}/slow/v1` }, "m", { signal: ctl.signal, timeoutMs: 5000 });
    expect(p.error!.kind).toBe("aborted");
    const already = new AbortController();
    already.abort();
    expect((await probeModel({ name: "s", baseURL: `${base}/vllm/v1`, apiKey: "sk-vllm" }, "m", { signal: already.signal })).error!.kind).toBe("aborted");
  });

  test("vision: a picture accepted, a picture refused; only acceptance is claimed", async () => {
    requests.length = 0;
    const yes = await probeModel({ name: "or", baseURL: `${base}/or/v1` }, "openai/gpt-5", { vision: true });
    expect(yes.toolCall).toBe("ok");
    expect(yes.vision!.result).toBe("accepted");
    expect(requests).toHaveLength(2);
    expect(JSON.stringify(requests[1]!.body.messages)).toContain("data:image/png;base64,");
    const no = await probeModel({ name: "or", baseURL: `${base}/or/v1` }, "meta/tiny", { vision: true });
    expect(no.vision!.result).toBe("rejected");
    expect(no.vision!.error!.kind).toBe("rejected");
    expect(no.vision!.error!.message).toContain("image");
  });

  test("OpenRouter with no tool-capable route: tools-unsupported with a routing hint; its image 404 is a rejected picture", async () => {
    const p = await probeModel({ name: "or", baseURL: `${base}/ornoroute/v1` }, "some/model", { vision: true });
    expect(p.toolCall).toBe("error");
    expect(p.error!.kind).toBe("tools-unsupported");
    expect(p.error!.status).toBe(404);
    expect(p.error!.message).toContain("support tool use");
    expect(p.hint).toContain("No route serving this model supports tool calls");
    expect(p.hint).not.toContain("vLLM");
    // The model was reached (it was the tool that was refused), so the picture is tried too.
    expect(p.vision!.result).toBe("rejected");
    expect(p.vision!.error!.kind).toBe("rejected");
    expect(p.vision!.error!.message).toContain("support image input");
  });

  test("a 404 for an unknown model stays not-found, and its picture is an error, not a rejection", async () => {
    const p = await probeModel({ name: "x", baseURL: `${base}/missing/v1` }, "llama-3.2-11b-vision", { vision: true });
    expect(p.error!.kind).toBe("not-found");
    expect(p.hint).toBeUndefined();
    expect(p.vision).toBeUndefined(); // not reached: no picture after it
  });

  test("a reasoning model that spends its whole budget thinking: out-of-tokens, not text-instead; the default budget has room to think", async () => {
    const p = await probeModel({ name: "r", baseURL: `${base}/thinker/v1` }, "deep/thinker");
    expect(p.toolCall).toBe("error");
    expect(p.error!.kind).toBe("out-of-tokens");
    expect(p.finishReason).toBe("length");
    expect(p.text).toBeUndefined();
    expect(p.hint).toContain("output budget");
    expect(p.hint).not.toContain("--enable-auto-tool-choice");
    expect(lastMaxTokens).toBe(DEFAULT_PROBE_MAX_OUTPUT_TOKENS);
    expect(DEFAULT_PROBE_MAX_OUTPUT_TOKENS).toBeGreaterThanOrEqual(8192);
    const small = await probeModel({ name: "r", baseURL: `${base}/thinker/v1` }, "deep/thinker", { maxOutputTokens: 512 });
    expect(lastMaxTokens).toBe(512);
    expect(small.error!.kind).toBe("out-of-tokens");
  });

  test("without vision asked for, no picture is sent", async () => {
    requests.length = 0;
    const p = await probeModel({ name: "or", baseURL: `${base}/or/v1` }, "openai/gpt-5");
    expect(p.vision).toBeUndefined();
    expect(requests).toHaveLength(1);
  });
});
