import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { APICallError, generateText, isStepCount, streamText, tool, type LanguageModel } from "ai";
import { z } from "zod";
import type { OpenAICompatProvider } from "./backend.ts";

/**
 * What an OpenAI-compatible endpoint says about its models, and what one small request shows it can do.
 *
 * Two sources, kept apart because they are not equally trustworthy:
 *
 *  - REPORTED (`listOpenAICompatModels`): what `GET /models` says. Endpoints say very different amounts:
 *    vLLM reports a context window (`max_model_len`) and nothing else; OpenRouter reports a context window,
 *    input modalities and the request parameters a model takes. Whatever an endpoint does not say stays
 *    `undefined`, never `false`: "not reported" is not "no". A field IS false when the endpoint reports the
 *    list it would appear in and it is not there (OpenRouter's `supported_parameters` without `tools`).
 *  - PROBED (`probeModel`): one minimal streamed request offering a single tool and asking for a call.
 *    It is the only way to learn whether tool calls actually work end to end, which is what an agent loop
 *    needs and what a served model most often gets wrong (a vLLM started without
 *    `--enable-auto-tool-choice`, or with the wrong `--tool-call-parser`). The optional vision probe sends a
 *    small picture and reports whether the endpoint ACCEPTED it. A caller may take acceptance as support
 *    (workstream does): the endpoints this is used with (vLLM, OpenRouter) refuse a picture for a model that
 *    cannot take one. An endpoint that silently drops images instead reads as accepting them, and needs the
 *    caller's vision override set to no.
 *
 * The probe goes through the same AI SDK provider the driver uses (backend.ts), so what it shows is what a
 * turn would see. It never retries: a cold start is waited out by a long timeout (minutes, for a scaled-to-
 * zero GPU), not by repeating a request that is already queued behind the boot.
 */

/** What `/models` said about one model. Every field is optional: absent means the endpoint did not say. */
export interface ReportedModel {
  /** Its context window in tokens (vLLM `max_model_len`, OpenRouter `context_length`, ...). */
  contextWindow?: number;
  /** The most it will write in one response, when the endpoint says (OpenRouter `top_provider`). */
  maxOutputTokens?: number;
  /** What it takes as input (OpenRouter `architecture.input_modalities`: "text", "image", "file", ...). */
  inputModalities?: string[];
  /** Takes images: derived from `inputModalities`. */
  vision?: boolean;
  /** Accepts `tools` (OpenRouter `supported_parameters`). */
  tools?: boolean;
  /** Accepts a reasoning control (OpenRouter `supported_parameters`: `reasoning` / `reasoning_effort`). */
  reasoning?: boolean;
  /** A display name, when the endpoint has one besides the id (OpenRouter `name`). */
  name?: string;
}

/** One model an endpoint lists. */
export interface ListedModel {
  /** `<provider>/<model>`: the id a picker shows. */
  id: string;
  /** The model's own id, as the endpoint lists it. */
  label: string;
  provider: string;
  /** The model's own id (the same as `label`), to send as the request's `model`. */
  model: string;
  reported: ReportedModel;
}

const positive = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) && v > 0 ? Math.floor(v) : undefined);
const strings = (v: unknown): string[] | undefined => (Array.isArray(v) && v.every((s) => typeof s === "string") ? (v as string[]) : undefined);

/** One `/models` entry, in the shapes the common endpoints use, as a `ReportedModel`. */
export function reportedModel(entry: Record<string, unknown>): ReportedModel {
  const top = (entry.top_provider ?? {}) as Record<string, unknown>;
  const arch = (entry.architecture ?? {}) as Record<string, unknown>;
  const contextWindow = positive(entry.max_model_len) ?? positive(entry.context_length) ?? positive(entry.context_window) ?? positive(entry.max_context_length) ?? positive(top.context_length);
  const maxOutputTokens = positive(top.max_completion_tokens) ?? positive(entry.max_completion_tokens) ?? positive(entry.max_output_tokens);
  const inputModalities = strings(arch.input_modalities) ?? strings(entry.input_modalities);
  const params = strings(entry.supported_parameters);
  const out: ReportedModel = {};
  if (contextWindow !== undefined) out.contextWindow = contextWindow;
  if (maxOutputTokens !== undefined) out.maxOutputTokens = maxOutputTokens;
  if (inputModalities) {
    out.inputModalities = inputModalities;
    out.vision = inputModalities.includes("image");
  }
  if (params) {
    out.tools = params.includes("tools");
    out.reasoning = params.includes("reasoning") || params.includes("reasoning_effort");
  }
  if (typeof entry.name === "string" && entry.name && entry.name !== entry.id) out.name = entry.name;
  return out;
}

const requestHeaders = (provider: OpenAICompatProvider): Record<string, string> => ({
  ...(provider.apiKey ? { authorization: `Bearer ${provider.apiKey}` } : {}),
  ...(provider.headers ?? {}),
});

/**
 * The models an endpoint lists (`GET <baseURL>/models`, the OpenAI shape `{data: [{id, ...}]}`), each with
 * what the endpoint reported about it. A provider that does not answer throws, saying its status; the
 * caller decides whether that is an error.
 */
export async function listOpenAICompatModels(provider: OpenAICompatProvider, fetchImpl: typeof fetch = fetch, signal?: AbortSignal): Promise<ListedModel[]> {
  const res = await fetchImpl(`${provider.baseURL.replace(/\/$/, "")}/models`, { headers: requestHeaders(provider), ...(signal ? { signal } : {}) });
  if (!res.ok) throw new Error(`${provider.name}: GET /models answered ${res.status}`);
  const body = (await res.json()) as { data?: unknown };
  const data = Array.isArray(body.data) ? (body.data as Array<Record<string, unknown>>) : [];
  return data.flatMap((m) =>
    m && typeof m === "object" && typeof m.id === "string" ? [{ id: `${provider.name}/${m.id}`, label: m.id, provider: provider.name, model: m.id, reported: reportedModel(m) }] : [],
  );
}

/** Why a probe failed, in the categories a human acts on differently. */
export type ProbeErrorKind =
  /** 401 / 403: the key (or a credential header) is wrong or missing. */
  | "auth"
  /** 404: no such model (or no such path: a base URL without `/v1`). */
  | "not-found"
  /**
   * The endpoint refused the tool itself: a vLLM without `--enable-auto-tool-choice`, or an OpenRouter model
   * with no route that serves tools (a 404 "No endpoints found that support tool use").
   */
  | "tools-unsupported"
  /**
   * The model used its whole output budget before answering (`finish_reason: length`, no text and no tool
   * call): a reasoning model still thinking. Says nothing about its tool calls either way.
   */
  | "out-of-tokens"
  /** The endpoint refused the picture (vision probe only). */
  | "rejected"
  /** No answer within the timeout. */
  | "timeout"
  /** The caller's signal stopped it. */
  | "aborted"
  /** Nothing answered at all (refused connection, DNS, TLS). */
  | "unreachable"
  /** Any other error status. */
  | "http"
  /** Anything else (a malformed stream, ...). */
  | "other";

export interface ProbeError {
  kind: ProbeErrorKind;
  message: string;
  /** The HTTP status, when there was one. */
  status?: number;
}

export interface VisionProbe {
  /** "accepted": the endpoint answered a request carrying a picture (how far that goes: the header above). */
  result: "accepted" | "rejected" | "error";
  latencyMs: number;
  error?: ProbeError;
}

export interface ModelProbe {
  /**
   * "ok": a structured tool call came back. "text-instead": the model answered, but in text, with no tool
   * call (a served model whose tool-call parser is off or wrong). "error": the request failed (`error`).
   */
  toolCall: "ok" | "text-instead" | "error";
  /** From sending the request to the end of the response (a cold start included). */
  latencyMs: number;
  /** From sending the request to its first streamed part, when one arrived. */
  firstTokenMs?: number;
  error?: ProbeError;
  /** A likely cause and fix, for a human: set for "text-instead" and for a refused tool. */
  hint?: string;
  /** The start of what the model wrote instead of calling the tool ("text-instead" only). */
  text?: string;
  /** Why the response ended (`stop`, `tool-calls`, `length`, ...), when it did. */
  finishReason?: string;
  /** Only when asked for (`opts.vision`), and only when the tool probe reached the model. */
  vision?: VisionProbe;
}

export interface ProbeOptions {
  /**
   * How long to wait for the whole probe (each request), in ms. Default 5 minutes: an endpoint that scales
   * to zero (Modal) boots a GPU and loads weights before it answers the first request.
   */
  timeoutMs?: number;
  /** Stops the probe; it then reports `error.kind: "aborted"`. */
  signal?: AbortSignal;
  /** Also send a small picture, and report whether the endpoint accepted it. */
  vision?: boolean;
  /**
   * A cap on the tool probe's output. Default `DEFAULT_PROBE_MAX_OUTPUT_TOKENS`: a reasoning model thinks
   * before it calls, and a budget it spends entirely on that reads as `out-of-tokens`, not as an answer.
   */
  maxOutputTokens?: number;
  /** Test seam: the fetch the requests go through. */
  fetch?: typeof fetch;
  /** Test seam: the model for an id (default: the provider's chat model). */
  model?: (modelId: string) => LanguageModel;
}

export const DEFAULT_PROBE_TIMEOUT_MS = 5 * 60_000;

/** The tool probe's default output budget: room for a reasoning model's preamble before it calls the tool. */
export const DEFAULT_PROBE_MAX_OUTPUT_TOKENS = 8192;

/** The name of the probe's one tool. */
export const PROBE_TOOL = "ping";

const PROBE_INSTRUCTIONS = `You are being tested for tool use. Call the ${PROBE_TOOL} tool exactly once, with no arguments. Do not write anything else.`;
const TEXT_INSTEAD_HINT =
  "The model answered in text instead of calling the tool. A served model needs its tool-call parser on: on vLLM, start it with --enable-auto-tool-choice and the --tool-call-parser for this model family (hermes, llama3_json, mistral, ...).";
const TOOLS_REFUSED_HINT = "The endpoint refused the tool. On vLLM, start the model with --enable-auto-tool-choice and a --tool-call-parser for its family.";
const NO_TOOL_ROUTE_HINT =
  "No route serving this model supports tool calls (on OpenRouter: none of its providers takes tools for it). Pick another model, or a provider route that supports tools.";
const OUT_OF_TOKENS_HINT =
  "The model spent its whole output budget before answering, most likely reasoning first. Nothing is known about its tool calls yet: probe it again with a larger output budget, or with a lower reasoning effort.";

/** A refused tool, as an endpoint words it: vLLM's tool_choice/parser errors, OpenRouter's "support tool use". */
const TOOLS_REFUSED = /tool[-_ ]choice|tool[-_ ]call[-_ ]parser|tools? (are|is) not supported|does not support tools|support tool use|support tools/i;
/** A refused picture, as an endpoint words it (OpenRouter: 404 "No endpoints found that support image input"). */
const IMAGE_REFUSED = /support image|image input|does not support images?/i;

/** A 32x32 PNG: big enough for vision processors that reject tiny images (Qwen-VL wants 28px a side). */
const PROBE_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR42u3NsQkAAAgDsJ7u5/YKByGQPdnJqQgEAoFAIBAIBIIvQQHoo/xM3BB6yAAAAABJRU5ErkJggg==";

const clip = (s: string, n = 300): string => (s.length > n ? `${s.slice(0, n)}…` : s);

/** An error as a `ProbeError`, telling a timeout from the caller's stop by which signal fired. */
function classify(e: unknown, timedOut: () => boolean, stopped: () => boolean): ProbeError {
  if (stopped()) return { kind: "aborted", message: "the probe was stopped" };
  if (timedOut()) return { kind: "timeout", message: "no answer within the timeout" };
  const api = findApiError(e);
  if (api) {
    const status = api.statusCode;
    const body = api.responseBody ? clip(api.responseBody.replace(/\s+/g, " ").trim()) : "";
    const message = body && !api.message.includes(body) ? `${api.message}: ${body}` : api.message;
    if (status === 401 || status === 403) return { kind: "auth", message, status };
    // Before the 404 check: OpenRouter answers a tool request it has no tool-capable route for with a 404.
    if (status !== undefined && TOOLS_REFUSED.test(`${api.message} ${body}`)) return { kind: "tools-unsupported", message, status };
    if (status === 404) return { kind: "not-found", message, status };
    if (status !== undefined) return { kind: "http", message, status };
    return { kind: "unreachable", message: api.message };
  }
  const message = e instanceof Error ? e.message : String(e);
  // fetch's own failure: nothing answered (connection refused, DNS, TLS).
  if (e instanceof TypeError || /ECONNREFUSED|ENOTFOUND|fetch failed|Unable to connect|ConnectionRefused/i.test(message)) return { kind: "unreachable", message };
  return { kind: "other", message };
}

function findApiError(e: unknown): APICallError | null {
  for (let x: unknown = e, i = 0; x && i < 5; i++, x = (x as { cause?: unknown }).cause) {
    if (APICallError.isInstance(x)) return x;
    const last = (x as { lastError?: unknown }).lastError;
    if (last && APICallError.isInstance(last)) return last;
  }
  return null;
}

/** A signal that fires on the caller's signal or after `timeoutMs`, and which of the two it was. */
function deadline(timeoutMs: number, outer?: AbortSignal): { signal: AbortSignal; timedOut: () => boolean; stopped: () => boolean; done: () => void } {
  const ctl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctl.abort(new DOMException("probe timed out", "TimeoutError"));
  }, timeoutMs);
  const onAbort = () => ctl.abort(outer?.reason);
  if (outer?.aborted) onAbort();
  else outer?.addEventListener("abort", onAbort, { once: true });
  return {
    signal: ctl.signal,
    timedOut: () => timedOut,
    stopped: () => !timedOut && !!outer?.aborted,
    done: () => {
      clearTimeout(timer);
      outer?.removeEventListener("abort", onAbort);
    },
  };
}

/**
 * Probe one model: a single streamed request offering one tool (`ping`, no arguments) and asking for a
 * call, timed. Never throws for anything the endpoint does: a failure is the result's `error`.
 */
export async function probeModel(provider: OpenAICompatProvider, modelId: string, opts: ProbeOptions = {}): Promise<ModelProbe> {
  const chat =
    opts.model ??
    ((id: string) =>
      createOpenAICompatible({ name: provider.name, baseURL: provider.baseURL, apiKey: provider.apiKey, headers: provider.headers, ...(opts.fetch ? { fetch: opts.fetch } : {}) }).chatModel(id));
  const model = chat(modelId);
  const timeoutMs = opts.timeoutMs ?? DEFAULT_PROBE_TIMEOUT_MS;

  const d = deadline(timeoutMs, opts.signal);
  const started = performance.now();
  let firstTokenMs: number | undefined;
  let probe: ModelProbe;
  try {
    const result = streamText({
      model,
      instructions: PROBE_INSTRUCTIONS,
      prompt: `Call ${PROBE_TOOL} now.`,
      tools: { [PROBE_TOOL]: tool({ description: "Answers pong. Call it when asked to.", inputSchema: z.object({}) }) },
      stopWhen: isStepCount(1),
      maxRetries: 0,
      maxOutputTokens: opts.maxOutputTokens ?? DEFAULT_PROBE_MAX_OUTPUT_TOKENS,
      abortSignal: d.signal,
      onError: () => {},
    });
    let text = "";
    let failed: unknown = null;
    for await (const part of result.stream) {
      if (firstTokenMs === undefined && part.type !== "start" && part.type !== "start-step") firstTokenMs = Math.round(performance.now() - started);
      if (part.type === "text-delta") text += part.text;
      else if (part.type === "error") failed ??= part.error;
      else if (part.type === "abort") failed ??= new DOMException("aborted", "AbortError");
    }
    if (failed) throw failed;
    const step = await result.finalStep;
    const latencyMs = Math.round(performance.now() - started);
    const finishReason = step.finishReason;
    if (step.toolCalls.length) probe = { toolCall: "ok", latencyMs, finishReason };
    // Cut off with nothing said: the budget went on thinking. Not "answered in text": it did not answer.
    else if (finishReason === "length" && !text.trim())
      probe = { toolCall: "error", latencyMs, finishReason, error: { kind: "out-of-tokens", message: "the output budget ran out before the model answered" }, hint: OUT_OF_TOKENS_HINT };
    else probe = { toolCall: "text-instead", latencyMs, finishReason, hint: TEXT_INSTEAD_HINT, text: clip(text.trim(), 200) };
  } catch (e) {
    const error = classify(e, d.timedOut, d.stopped);
    const hint = error.kind !== "tools-unsupported" ? undefined : error.status === 404 || /no endpoints found/i.test(error.message) ? NO_TOOL_ROUTE_HINT : TOOLS_REFUSED_HINT;
    probe = { toolCall: "error", latencyMs: Math.round(performance.now() - started), error, ...(hint ? { hint } : {}) };
  } finally {
    d.done();
  }
  if (firstTokenMs !== undefined) probe.firstTokenMs = firstTokenMs;

  // Only when the model was reached: a picture sent to an endpoint that refused the key or never answered
  // would only repeat that, and wait out a second timeout.
  const reached = probe.toolCall !== "error" || probe.error?.kind === "tools-unsupported" || probe.error?.kind === "out-of-tokens" || probe.error?.kind === "http";
  if (opts.vision && reached) probe.vision = await probeVision(model, timeoutMs, opts.signal);
  return probe;
}

async function probeVision(model: LanguageModel, timeoutMs: number, signal: AbortSignal | undefined): Promise<VisionProbe> {
  const d = deadline(timeoutMs, signal);
  const started = performance.now();
  try {
    await generateText({
      model,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What colour is this picture? One word." },
            { type: "file", data: PROBE_IMAGE, mediaType: "image/png" },
          ],
        },
      ],
      maxRetries: 0,
      maxOutputTokens: 256,
      abortSignal: d.signal,
    });
    return { result: "accepted", latencyMs: Math.round(performance.now() - started) };
  } catch (e) {
    const error = classify(e, d.timedOut, d.stopped);
    // A 4xx other than a key problem is the endpoint saying no to the picture (or to image parts at all). A
    // 404 is "no such model" unless it says it is the picture (OpenRouter: "...support image input").
    const rejected =
      error.status !== undefined &&
      error.status >= 400 &&
      error.status < 500 &&
      error.kind !== "auth" &&
      (error.kind !== "not-found" || IMAGE_REFUSED.test(error.message));
    return { result: rejected ? "rejected" : "error", latencyMs: Math.round(performance.now() - started), error: rejected ? { ...error, kind: "rejected" } : error };
  } finally {
    d.done();
  }
}
