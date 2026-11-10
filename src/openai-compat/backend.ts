import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { isStepCount, streamText, tool, type LanguageModel, type ModelMessage, type ToolResultPart, type ToolSet } from "ai";
import { z } from "zod";
import type { AdapterContinueInput, AdapterRunInput, AdapterRunOptions, AdapterRunResult, BackendAdapter, TurnUsage } from "../adapter.ts";
import { runAtBoundary, type ToolBoundary } from "../boundary.ts";
import type { EffortLevel } from "../effort.ts";
import type { ToolBundle, ToolDef, ToolResult } from "../tool.ts";

/**
 * A driver for any OpenAI-compatible chat API, with the agent loop written HERE rather than handed to a
 * framework.
 *
 * The AI SDK (`ai` + `@ai-sdk/openai-compatible`) is used for exactly one thing: one model call, streamed
 * (`streamText`, one step, tools declared WITHOUT `execute`). Everything around that call is this file's:
 * running each tool call through the host's tool boundary (the same boundary.ts the Claude driver uses,
 * so a refusal, the transcript rows, and messages that arrived mid-turn behave identically on both), the
 * stop between steps, and the conversation's history. Keeping the loop is the point: what happens at a
 * tool boundary is exactly what a host most needs to control, and it is what a framework's loop hides.
 *
 * HISTORY is the host's (`ConversationStore`): unlike the Claude CLI, which keeps a session file it can
 * resume, an API call is stateless, so every message a turn produces is handed to the store as it is
 * produced. The one invariant that history must keep is that every tool call is answered before the next
 * request, or the conversation cannot go on: a turn answers every call it made before it ends (a stop and
 * a throwing tool included), and a continue answers whatever a process that DIED mid-tool left open. The
 * system prompt is NOT stored: it is sent on every turn from `input.systemPrompt`, which a host re-sends
 * every turn anyway.
 */

/** Where a conversation's messages live. A host implements it on its own store. */
export interface ConversationStore {
  /** Every message of the conversation, oldest first (empty for a new one). */
  load(conversationId: string): Promise<ModelMessage[]>;
  /** Append messages, in order, as they are produced. */
  append(conversationId: string, messages: ModelMessage[]): Promise<void>;
}

/** One OpenAI-compatible endpoint. */
export interface OpenAICompatProvider {
  /** The provider's name: the prefix of its models' ids in a picker, and its key in provider options. */
  name: string;
  baseURL: string;
  apiKey?: string;
  headers?: Record<string, string>;
}

export interface OpenAICompatBackendOptions {
  provider: OpenAICompatProvider;
  store: ConversationStore;
  /** The model when a run names none. A run with neither fails. */
  defaultModel?: string;
  /** What every tool call passes through. */
  boundary?: ToolBoundary;
  /** The name a call is recorded under. Default: the bare tool name. */
  recordAs?: (serverName: string, toolName: string) => string;
  /**
   * A cap on model calls per turn: a loop that never stops calling tools is otherwise unbounded. The
   * Claude CLI has no such cap either, so the default is high; a host that wants a budget sets one.
   */
  maxSteps?: number;
  /** A fresh conversation id (default: a UUID). */
  newId?: () => string;
  /**
   * Whether the model takes images. When it does, an image a tool returns is sent in a user message after
   * the tool results; when it does not (or it is not known), it is dropped with a note in the result.
   */
  vision?: boolean;
  /** Test seam: the model for an id (default: the provider's chat model). */
  model?: (modelId: string) => LanguageModel;
}

const DEFAULT_MAX_STEPS = 500;

/**
 * The reasoning efforts an OpenAI-compatible endpoint can be asked for (`reasoning_effort`). The higher
 * two of the runtime's vocabulary have no name there, so they are NOT sent: a level a driver cannot honour
 * is ignored, never rounded to a different one.
 */
export const OPENAI_COMPAT_EFFORTS: readonly EffortLevel[] = ["low", "medium", "high"];

/**
 * The name a tool is published under: `<server>__<tool>`, so a host that strips the server prefix (as it
 * does Claude's `mcp__<server>__<tool>`) gets the bare name back. OpenAI function names allow only
 * `[A-Za-z0-9_-]`, at most 64 characters.
 */
export function wireToolName(serverName: string, toolName: string): string {
  return `${serverName}__${toolName}`.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 64);
}

/** An image a tool returned, to be sent to the model in a user message after the tool results. */
export interface ToolImage {
  data: string; // base64
  mediaType: string;
}

/**
 * A tool's result as the model is sent it. TEXT only: an OpenAI-compatible `tool` message carries text, and
 * a provider that is handed parts there flattens them into a JSON string, so an image would reach the model
 * as base64 text (hundreds of thousands of tokens, never seen as a picture). Each image is named in the
 * text instead and returned separately, for a user message after the tool results when the model takes
 * images (`vision`), or replaced by a note saying it was dropped when it does not.
 */
export function toToolOutput(res: ToolResult, vision = false): { output: ToolResultPart["output"]; images: ToolImage[] } {
  const images: ToolImage[] = [];
  const lines = res.content.map((c) => {
    if (c.type === "text") return c.text;
    if (!vision) return `[an image (${c.mimeType}) was returned; this model does not take images, so it was not sent]`;
    images.push({ data: c.data, mediaType: c.mimeType });
    return `[image ${images.length} (${c.mimeType}): sent in the next message]`;
  });
  const value = lines.join("\n");
  return { output: res.isError ? { type: "error-text", value } : { type: "text", value }, images };
}

const errorResult = (call: { toolCallId: string; toolName: string }, text: string): ToolResultPart => ({
  type: "tool-result",
  toolCallId: call.toolCallId,
  toolName: call.toolName,
  output: { type: "error-text", value: text },
});

/**
 * Tool calls in `history` that no tool message answers. A conversation must answer every call before the
 * next request (the SDK refuses to send one that does not), and a process that dies between saving a call
 * and saving its result leaves exactly this behind.
 */
export function unansweredCalls(history: ModelMessage[]): Array<{ toolCallId: string; toolName: string }> {
  const answered = new Set<string>();
  for (const m of history) if (m.role === "tool") for (const p of m.content) if (p.type === "tool-result") answered.add(p.toolCallId);
  const out: Array<{ toolCallId: string; toolName: string }> = [];
  for (const m of history) {
    if (m.role !== "assistant" || typeof m.content === "string") continue;
    for (const p of m.content) if (p.type === "tool-call" && !answered.has(p.toolCallId)) out.push({ toolCallId: p.toolCallId, toolName: p.toolName });
  }
  return out;
}

export class OpenAICompatBackend implements BackendAdapter {
  readonly id: string = "openai-compat";
  private readonly chat: (modelId: string) => LanguageModel;

  constructor(private readonly opts: OpenAICompatBackendOptions) {
    if (opts.model) this.chat = opts.model;
    else {
      const p = createOpenAICompatible({ name: opts.provider.name, baseURL: opts.provider.baseURL, apiKey: opts.provider.apiKey, headers: opts.provider.headers, includeUsage: true });
      this.chat = (id) => p.chatModel(id);
    }
  }

  async run(input: AdapterRunInput, opts: AdapterRunOptions): Promise<AdapterRunResult> {
    const id = (this.opts.newId ?? (() => crypto.randomUUID()))();
    // Known at once, before any model call: a turn killed mid-flight is then resumable from the store.
    opts.onSessionId?.(id);
    return this.turn(id, [], input, opts);
  }

  async continueRun(input: AdapterContinueInput, opts: AdapterRunOptions): Promise<AdapterRunResult> {
    opts.onSessionId?.(input.backendSessionId);
    const history = await this.opts.store.load(input.backendSessionId);
    // A turn that ended normally, or was stopped, answered every call it made. One whose PROCESS died in
    // the middle of a tool did not; answer those now, saying so, or the conversation cannot go on.
    const dangling = unansweredCalls(history);
    if (dangling.length) {
      const repair: ModelMessage = { role: "tool", content: dangling.map((c) => errorResult(c, "No result: the turn ended before this call returned.")) };
      await this.opts.store.append(input.backendSessionId, [repair]);
      history.push(repair);
    }
    return this.turn(input.backendSessionId, history, input, opts);
  }

  private async turn(id: string, history: ModelMessage[], input: AdapterRunInput, opts: AdapterRunOptions): Promise<AdapterRunResult> {
    const modelId = input.model ?? this.opts.defaultModel;
    if (!modelId) throw new Error(`no model named for provider "${this.opts.provider.name}", and it has no default`);
    const model = this.chat(modelId);
    const { tools, defs } = this.toolSet(input.tools);
    const effort = input.effort && OPENAI_COMPAT_EFFORTS.includes(input.effort) ? input.effort : undefined;
    const signal = opts.signal;

    const messages: ModelMessage[] = [...history];
    const keep = async (msgs: ModelMessage[]): Promise<void> => {
      messages.push(...msgs);
      await this.opts.store.append(id, msgs);
    };
    await keep([{ role: "user", content: input.prompt }]);

    let usage: TurnUsage = { model: modelId, costUsd: 0, inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheCreationTokens: 0 };
    let text = "";
    const maxSteps = this.opts.maxSteps ?? DEFAULT_MAX_STEPS;
    for (let step = 0; ; step++) {
      signal?.throwIfAborted();
      if (step >= maxSteps) throw new Error(`stopped after ${maxSteps} model calls in one turn without a final answer`);
      const result = streamText({
        model,
        ...(input.systemPrompt ? { instructions: input.systemPrompt } : {}),
        messages,
        tools,
        stopWhen: isStepCount(1),
        // The error also arrives as an `error` part, which fails the turn below; the SDK's default
        // handler would only print it a second time.
        onError: () => {},
        ...(signal ? { abortSignal: signal } : {}),
        ...(effort ? { providerOptions: { [this.opts.provider.name]: { reasoningEffort: effort } } } : {}),
      });
      let stepText = "";
      let failed: unknown = null;
      for await (const part of result.stream) {
        switch (part.type) {
          case "text-delta":
            stepText += part.text;
            opts.onToken?.(part.text, false);
            break;
          case "reasoning-delta":
            opts.onToken?.(part.text, true);
            break;
          case "tool-input-start":
            opts.onToolDraft?.(part.toolName);
            break;
          case "tool-call":
            opts.onTool?.(part.toolName, part.input);
            break;
          case "error":
            failed ??= part.error;
            break;
          case "abort":
            failed ??= new DOMException("This operation was aborted", "AbortError");
            break;
        }
      }
      if (failed) throw failed instanceof Error ? failed : new Error(String(failed));
      const finalStep = await result.finalStep;
      const u = finalStep.usage;
      // The LAST call's usage is the context's size now; per-turn sums of every call would read as a
      // context far larger than it is. No cost: an OpenAI-compatible endpoint does not report one.
      usage = {
        model: modelId,
        costUsd: 0,
        inputTokens: u.inputTokenDetails?.noCacheTokens ?? u.inputTokens ?? 0,
        outputTokens: u.outputTokens ?? 0,
        cacheReadTokens: u.inputTokenDetails?.cacheReadTokens ?? 0,
        cacheCreationTokens: u.inputTokenDetails?.cacheWriteTokens ?? 0,
      };
      const response = finalStep.response.messages as ModelMessage[];
      await keep(response);
      const calls = finalStep.toolCalls;
      if (!calls.length) {
        text = stepText;
        break;
      }
      // A call the SDK already answered (input that fails the tool's schema, a tool that does not exist)
      // has its error result in the response just kept. Running it as well would hand the handler input its
      // schema never checked, and give the model two results for one call.
      const answered = new Set(response.flatMap((m) => (m.role === "tool" ? m.content.flatMap((p) => (p.type === "tool-result" ? [p.toolCallId] : [])) : [])));
      // One at a time, in the order the model wrote them: a call may depend on the one before it, and the
      // transcript then reads in the order things happened.
      //
      // EVERY call gets a result, whatever happens: a tool that throws answers as an error (as an MCP server
      // answers a throwing handler on Claude, so the turn goes on); a stop answers the call it interrupted
      // and every call after it, and only then ends the turn. Saved before the turn ends, so the next turn
      // resumes a conversation in which nothing is left unanswered.
      const results: ToolResultPart[] = [];
      const images: ToolImage[] = [];
      let stopped: unknown = null;
      for (const call of calls) {
        if (answered.has(call.toolCallId)) continue;
        if (stopped === null && signal?.aborted) stopped = signal.reason ?? new DOMException("This operation was aborted", "AbortError");
        if (stopped !== null) {
          results.push(errorResult(call, "Not run: the turn was stopped before this call."));
          continue;
        }
        const found = call.invalid ? undefined : defs.get(call.toolName);
        if (!found) {
          results.push(errorResult(call, call.invalid ? `Not run: the call's input is not valid for ${call.toolName}.` : `no tool named ${call.toolName}`));
          continue;
        }
        let res: ToolResult;
        try {
          res = await runAtBoundary(this.opts.boundary, { name: found.def.name, recordedName: found.recordedName, args: call.input, ...(signal ? { signal } : {}) }, () => found.def.handler(call.input));
        } catch (e) {
          if (signal?.aborted) {
            stopped = e;
            res = { content: [{ type: "text", text: "Interrupted: the turn was stopped while this call ran." }], isError: true };
          } else {
            res = { content: [{ type: "text", text: e instanceof Error ? e.message : String(e) }], isError: true };
          }
        }
        const { output, images: shown } = toToolOutput(res, this.opts.vision);
        images.push(...shown);
        results.push({ type: "tool-result", toolCallId: call.toolCallId, toolName: call.toolName, output });
      }
      if (results.length) await keep([{ role: "tool", content: results }]);
      if (images.length) {
        await keep([
          {
            role: "user",
            content: [
              { type: "text", text: `The ${images.length === 1 ? "image" : `${images.length} images`} returned by the tool calls above:` },
              ...images.map((i) => ({ type: "file" as const, data: i.data, mediaType: i.mediaType })),
            ],
          },
        ]);
      }
      if (stopped !== null) throw stopped;
    }
    return { text: text.trim(), backendSessionId: id, usage };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private toolSet(bundles: ToolBundle<any>[]): { tools: ToolSet; defs: Map<string, { def: ToolDef; recordedName: string }> } {
    const tools: ToolSet = {};
    const defs = new Map<string, { def: ToolDef; recordedName: string }>();
    for (const bundle of bundles) {
      for (const def of bundle.tools as ToolDef[]) {
        const wire = wireToolName(bundle.serverName, def.name);
        // No `execute`: the SDK then returns the call instead of running it, and the loop above runs it.
        tools[wire] = tool({ description: def.description, inputSchema: z.object(def.inputSchema) });
        defs.set(wire, { def, recordedName: this.opts.recordAs ? this.opts.recordAs(bundle.serverName, def.name) : def.name });
      }
    }
    return { tools, defs };
  }
}

/**
 * The models an endpoint lists (`GET <baseURL>/models`, the OpenAI shape `{data: [{id}]}`), as picker rows
 * whose ids are `<provider>/<model>`. A provider that does not answer lists nothing; the caller decides
 * whether that is an error.
 */
export async function listOpenAICompatModels(provider: OpenAICompatProvider, fetchImpl: typeof fetch = fetch): Promise<Array<{ id: string; label: string; provider: string }>> {
  const res = await fetchImpl(`${provider.baseURL.replace(/\/$/, "")}/models`, {
    headers: { ...(provider.apiKey ? { authorization: `Bearer ${provider.apiKey}` } : {}), ...(provider.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`${provider.name}: GET /models answered ${res.status}`);
  const body = (await res.json()) as { data?: Array<{ id?: unknown }> };
  return (body.data ?? []).flatMap((m) => (typeof m.id === "string" ? [{ id: `${provider.name}/${m.id}`, label: m.id, provider: provider.name }] : []));
}
