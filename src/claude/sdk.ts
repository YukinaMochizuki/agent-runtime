import { query, type Options, type SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";
import { EFFORT_LEVELS, type EffortLevel } from "../effort.ts";
import type { ModelChoice, TurnUsage } from "../adapter.ts";
import type { Prompt, PromptPart } from "../tool.ts";
import { ClaudeStreamTranslator } from "./stream.ts";

/**
 * The Claude Agent SDK, driven for a host that owns its own tools. Knows nothing about any store: it runs
 * one turn and returns the conclusion and the SDK's session id.
 *
 * WHICH LOGIN a turn runs on is a CLAUDE_CONFIG_DIR (where `claude auth login` left its credentials):
 * `configDir` here. Absent leaves the environment as it is, which is the login of the user the process
 * runs as. Picking among several logins is a host's policy, not this module's.
 */

/**
 * The environment for every Claude Code subprocess. The CLI otherwise fetches the logged-in claude.ai
 * account's connectors (Notion, Linear, …) and loads them as MCP servers in EVERY session: callable tools
 * and "needs authentication" notices the host never granted, which also bypass `tools: []`.
 * `settingSources: []` means the `disableClaudeAiConnectors` setting is never read, so the env var is the
 * only switch. The SDK's `env` REPLACES the child's environment rather than merging, hence the spread.
 */
export function claudeSubprocessEnv(configDir: string | null | undefined = null): NonNullable<Options["env"]> {
  return { ...process.env, ENABLE_CLAUDEAI_MCP_SERVERS: "false", ...(configDir ? { CLAUDE_CONFIG_DIR: configDir } : {}) };
}

/**
 * The models the SDK reports for a login. `supportedModels()` lives on the Query object, so this boots a
 * throwaway query (empty prompt, never iterated → the prompt never runs), pulls the list, and aborts.
 * Cheap enough to call once and cache at the caller.
 */
export async function listClaudeModels(configDir: string | null = null): Promise<ModelChoice[]> {
  const ac = new AbortController();
  const q = query({ prompt: "", options: { tools: [], settingSources: [], permissionMode: "dontAsk", abortController: ac, env: claudeSubprocessEnv(configDir) } });
  try {
    const models = await q.supportedModels();
    return models.map((m) => ({
      id: m.value,
      canonicalId: m.resolvedModel,
      label: m.displayName,
      description: m.description,
      // Only what ModelInfo actually carries. There is deliberately no "vision" here: the SDK reports no
      // such field, and inventing the label from a guess is the kind of claim that goes stale silently.
      capabilities: [
        ...(m.supportsEffort && m.supportedEffortLevels?.length ? [`effort ${m.supportedEffortLevels[0]}–${m.supportedEffortLevels[m.supportedEffortLevels.length - 1]}`] : []),
        ...(m.supportsAdaptiveThinking ? ["adaptive thinking"] : []),
        ...(m.supportsFastMode ? ["fast mode"] : []),
      ],
      // Narrowed to the levels this package has a name for, in OUR order rather than the SDK's.
      effortLevels: m.supportsEffort ? EFFORT_LEVELS.filter((l) => m.supportedEffortLevels?.includes(l)) : [],
    }));
  } finally {
    ac.abort(); // discard the throwaway session; the empty prompt never ran
  }
}

/** How full one limit window is, in percent (0–100), and when it resets. */
export interface UsageWindow {
  utilization: number | null;
  resetsAt: string | null;
}

/** What a login says about itself. */
export interface AccountProbe {
  email: string | null;
  plan: string | null;
  fiveHour: UsageWindow | null;
  sevenDay: UsageWindow | null;
}

/**
 * Ask a login about itself, the way `listClaudeModels` asks for models: a throwaway query whose empty
 * prompt never runs, so no model is called and no quota is spent. The plan comes from `accountInfo()`; the
 * windows from Claude Code's own /usage data, which the SDK marks experimental, so a failure there leaves
 * them null rather than failing the probe.
 */
export async function probeClaudeAccount(configDir: string | null, timeoutMs = 15_000): Promise<AccountProbe> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  const q = query({ prompt: "", options: { tools: [], settingSources: [], permissionMode: "dontAsk", abortController: ac, env: claudeSubprocessEnv(configDir) } });
  try {
    const info = await q.accountInfo();
    let windows: { five_hour?: { utilization: number | null; resets_at: string | null } | null; seven_day?: { utilization: number | null; resets_at: string | null } | null } | null = null;
    let plan = info.subscriptionType ?? null;
    try {
      const usage = await q.usage_EXPERIMENTAL_MAY_CHANGE_DO_NOT_RELY_ON_THIS_API_YET({ skipBehaviors: true });
      windows = usage.rate_limits ?? null;
      plan = plan ?? usage.subscription_type ?? null;
    } catch {
      // The usage call is experimental; the plan alone is still worth showing.
    }
    const win = (w: { utilization: number | null; resets_at: string | null } | null | undefined): UsageWindow | null =>
      w ? { utilization: w.utilization, resetsAt: w.resets_at } : null;
    return { email: info.email ?? null, plan, fiveHour: win(windows?.five_hour), sevenDay: win(windows?.seven_day) };
  } finally {
    clearTimeout(timer);
    ac.abort(); // discard the throwaway session; the empty prompt never ran
  }
}

/** One reading of how full a limit window is, from a turn's `rate_limit_event`. */
export interface RateLimitReading {
  window: "five_hour" | "seven_day";
  /** 0–100 (the event says 0–1; normalised here). */
  percent: number;
  /** Epoch ms, or null when the event did not say. */
  resetsAt: number | null;
}

export interface RunClaudeAgentInput {
  /** The login to run on (absent: the process's own environment). */
  configDir?: string | null;
  /** Text, or text and images (sent as one user message of content blocks; see `claudePrompt`). */
  prompt: Prompt;
  systemPrompt?: string;
  /** Absent: the SDK's own default. A host with a configured default passes it here. */
  model?: string;
  /** How hard the model should think. Absent leaves the model's own default (not the same as any level). */
  effort?: EffortLevel;
  /** Continue a prior conversation by its session id. */
  resumeSessionId?: string;
  /** With resumeSessionId, branch into a new session id instead of continuing in place. */
  forkSession?: boolean;
  /** SDK MCP servers to expose (see claude/mcp.ts `adaptTools`). */
  mcpServers?: Options["mcpServers"];
  /** Tools the agent may call without asking. Defaults to none (conversation-only). */
  allowedTools?: string[];
  /**
   * Claude Code's BUILT-IN tools (Bash, Read, WebSearch, …) to expose. Default: NONE, and that default is
   * the safety boundary: without `tools: []` they default to the full claude_code preset and the agent can
   * read the host's cwd or run bash, even under `dontAsk` (which only denies execution, it does not hide
   * the tools). A host that wants some names them here, and must also list them in `allowedTools`.
   */
  builtinTools?: string[];
  onToken?: (text: string, thinking?: boolean) => void;
  onTool?: (name: string, input?: unknown) => void;
  onToolDraft?: (name: string) => void;
  /** Fired once, the first time the SDK reveals its session_id (every message carries it). */
  onSessionId?: (id: string) => void;
  /** Each `rate_limit_event`: how full a window of this login is, as of this request. */
  onRateLimit?: (reading: RateLimitReading) => void;
  /** Abort the run mid-flight (user stop). */
  signal?: AbortSignal;
}

export interface RunClaudeAgentResult {
  text: string;
  sessionId: string;
  usage: TurnUsage;
}

/** The SDK options a run is started with. Exported for tests: these flags ARE the behaviour. */
export function claudeQueryOptions(input: RunClaudeAgentInput): Options {
  const options: Options = {
    ...(input.model ? { model: input.model } : {}),
    tools: input.builtinTools ?? [],
    // Auto-allow the host's MCP tools (they would otherwise be gated by dontAsk).
    allowedTools: input.allowedTools ?? [],
    permissionMode: "dontAsk",
    settingSources: [],
    // Ask for thinking SUMMARIES instead of the default empty blocks. Newer models (Opus 5.5 and its
    // generation) write their user-facing notes between tool calls as progress-update `thinking` blocks
    // rather than `text`, so at the default `omitted` display a turn's mid-way commentary never reaches
    // the host. `updates` is what the API recommends, but the bundled Claude Code rejects it, so
    // `summarized` it is: reasoning summaries come along too, and stream.ts shows both as prose. Passed
    // as a raw CLI flag rather than the SDK's `thinking` option, which would also force a thinking TYPE
    // and override each model's default.
    extraArgs: { "thinking-display": "summarized" },
    env: claudeSubprocessEnv(input.configDir ?? null),
    ...(input.onToken || input.onTool || input.onToolDraft ? { includePartialMessages: true } : {}),
    ...(input.mcpServers ? { mcpServers: input.mcpServers } : {}),
    ...(input.effort ? { effort: input.effort } : {}),
    ...(input.systemPrompt ? { systemPrompt: input.systemPrompt } : {}),
    ...(input.resumeSessionId ? { resume: input.resumeSessionId } : {}),
    ...(input.forkSession ? { forkSession: true } : {}),
  };
  return options;
}

/** Prompt parts as Messages API content blocks: an image as a base64 source. */
export function claudeContentBlocks(parts: PromptPart[]): Array<Record<string, unknown>> {
  return parts.map((p) =>
    p.type === "text" ? { type: "text", text: p.text } : { type: "image", source: { type: "base64", media_type: p.mimeType, data: p.data } },
  );
}

/**
 * The prompt as `query` takes it. A string goes as it always has. Parts go as ONE user message of content
 * blocks, through the SDK's streaming input (an iterable of user messages): a string prompt cannot carry an
 * image. The iterable ends after that message, so the turn is still a single user turn: the SDK closes the
 * CLI's input once the first result arrives (it waits for it because in-process MCP tools need the channel).
 * The same holds on a resume: the message is the next user message of the resumed session.
 */
export function claudePrompt(prompt: Prompt): string | AsyncIterable<SDKUserMessage> {
  if (typeof prompt === "string") return prompt;
  const message = {
    type: "user",
    message: { role: "user", content: claudeContentBlocks(prompt) },
    parent_tool_use_id: null,
  } as unknown as SDKUserMessage; // the block shapes are the Messages API's; the SDK types them from its own copy
  return (async function* () {
    yield message;
  })();
}

/** Why a non-success `result` ended the run: its subtype, then the CLI's own errors when it gave some. */
export function resultErrorMessage(result: { subtype: string; errors?: unknown }): string {
  const errors = Array.isArray(result.errors) ? result.errors.filter((e): e is string => typeof e === "string" && e.length > 0) : [];
  return `Claude Agent SDK run ended with "${result.subtype}"${errors.length ? `: ${errors.join("; ")}` : ""}`;
}

export async function runClaudeAgent(input: RunClaudeAgentInput): Promise<RunClaudeAgentResult> {
  const options = claudeQueryOptions(input);

  // Bridge the caller's AbortSignal to the SDK's AbortController so a stop aborts the query.
  if (input.signal) {
    const ac = new AbortController();
    if (input.signal.aborted) ac.abort();
    else input.signal.addEventListener("abort", () => ac.abort(), { once: true });
    options.abortController = ac;
  }

  let text = "";
  let sessionId = "";
  const model = input.model ?? "";
  let usage: TurnUsage = { model, costUsd: 0, inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheCreationTokens: 0 };
  // The LAST assistant message's usage = the size of the final prompt = current context occupancy. Used
  // for the token counts instead of the `result` message's usage, which aggregates the whole turn across
  // every tool round-trip (that sums cache-reads, so a heavy multi-tool turn reads as a huge, misleading
  // "context %"). Cost still comes from the result (its total_cost_usd is the turn total).
  let ctxUsage: { input: number; cacheRead: number; cacheCreation: number; output: number } | null = null;
  // Prose and tool calls, from the stream or (on Claude Code's non-streaming fallback) from the complete
  // assistant message — see stream.ts.
  const translator = new ClaudeStreamTranslator({ onToken: input.onToken, onTool: input.onTool, onToolDraft: input.onToolDraft });

  for await (const message of query({ prompt: claudePrompt(input.prompt), options })) {
    // Every message carries session_id: surface it from the FIRST one, not at `result`. A run killed
    // mid-flight (a restart) must still be resumable.
    const mid = (message as { session_id?: string }).session_id;
    if (mid && !sessionId) {
      sessionId = mid;
      input.onSessionId?.(sessionId);
    }
    translator.handle(message as { type: string; event?: unknown; message?: unknown });
    if (message.type === "rate_limit_event") {
      const info = (message as { rate_limit_info?: { rateLimitType?: string; utilization?: number; resetsAt?: number } }).rate_limit_info;
      const window = info?.rateLimitType === "five_hour" || info?.rateLimitType === "seven_day" ? info.rateLimitType : null;
      if (window && typeof info?.utilization === "number") {
        input.onRateLimit?.({ window, percent: info.utilization * 100, resetsAt: typeof info.resetsAt === "number" ? info.resetsAt * 1000 : null });
      }
    } else if (message.type === "assistant") {
      // Each assistant message carries the usage of the API request that produced it; keep the latest so
      // `result` can report context occupancy (final prompt size), not the turn sum.
      const mu = (message.message as {
        usage?: { input_tokens?: number; output_tokens?: number; cache_read_input_tokens?: number; cache_creation_input_tokens?: number };
      }).usage;
      if (mu) {
        ctxUsage = {
          input: mu.input_tokens ?? 0,
          output: mu.output_tokens ?? 0,
          cacheRead: mu.cache_read_input_tokens ?? 0,
          cacheCreation: mu.cache_creation_input_tokens ?? 0,
        };
      }
    } else if (message.type === "result") {
      sessionId = message.session_id;
      // cost/usage live on the final result message; guard each field (beta surface).
      const m = message as {
        total_cost_usd?: number;
        usage?: { input_tokens?: number; output_tokens?: number; cache_read_input_tokens?: number; cache_creation_input_tokens?: number };
      };
      const u = m.usage ?? {};
      usage = {
        model,
        costUsd: m.total_cost_usd ?? 0,
        inputTokens: ctxUsage?.input ?? u.input_tokens ?? 0,
        outputTokens: ctxUsage?.output ?? u.output_tokens ?? 0,
        cacheReadTokens: ctxUsage?.cacheRead ?? u.cache_read_input_tokens ?? 0,
        cacheCreationTokens: ctxUsage?.cacheCreation ?? u.cache_creation_input_tokens ?? 0,
      };
      if (message.subtype === "success") {
        text = message.result;
      } else {
        throw new Error(resultErrorMessage(message));
      }
    }
  }

  return { text: text.trim(), sessionId, usage };
}
