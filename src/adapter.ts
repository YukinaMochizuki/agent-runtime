import type { EffortLevel } from "./effort.ts";
import type { Prompt, ToolBundle } from "./tool.ts";

/**
 * The backend-neutral driver contract. A host talks only to a BackendAdapter, so adding or swapping a
 * driver never touches the caller. Each call is one TURN: the adapter exposes the tools, drives the model,
 * streams its output through `opts`, and returns the conclusion and usage.
 *
 * Conversation identity:
 *   - `run` starts a fresh conversation; result.backendSessionId is its id.
 *   - `continueRun` resumes an existing conversation in place.
 * Ids are NOT portable between drivers.
 */

export interface AdapterRunOptions {
  /**
   * Live token deltas. `thinking` marks the model's thinking as opposed to its reply. Every driver must
   * tag it: a switch between the two is where a transcript starts a new row.
   */
  onToken?: (text: string, thinking?: boolean) => void;
  /** Each tool invocation, with its parsed input, once its arguments are complete. */
  onTool?: (name: string, input?: unknown) => void;
  /**
   * A tool call's arguments STARTED streaming, before `onTool`. The gap is the model writing, not the
   * tool running, and for a big write it is seconds long. For a live status only.
   */
  onToolDraft?: (name: string) => void;
  /**
   * The conversation id, fired AS SOON as it is known (not at run end), so a run interrupted mid-flight
   * (a process restart) is still resumable.
   */
  onSessionId?: (backendSessionId: string) => void;
  /** Abort the in-flight turn (a user stop). Every driver must honour it. */
  signal?: AbortSignal;
  /**
   * The driver compacted the conversation during this turn: from here on the model sees a summary of what
   * came before, not the messages themselves. Every driver that compacts reports it here, in one shape, so a
   * host can mark the point in a transcript. Fired once per compaction, at the point in the turn it happened.
   */
  onCompaction?: (event: CompactionEvent) => void;
}

/** A compaction, as a driver reports it to the host (`AdapterRunOptions.onCompaction`). */
export interface CompactionEvent {
  /** `auto`: the driver's own threshold; `manual`: someone asked for it (Claude Code's `/compact`). */
  trigger: "auto" | "manual";
  /** The context before it, in tokens, as the driver measured it. */
  tokensBefore: number;
  /** The context after it, when the driver reports that. */
  tokensAfter?: number;
  /** Whether the older part was summarized by the model (true) or only pruned (false); absent when unknown. */
  summarized?: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface AdapterRunInput<B extends ToolBundle<any> = ToolBundle> {
  /**
   * The user message that opens the turn: text, or text and images in order (`PromptPart`). Images go to
   * the model as images; a driver whose model cannot take them replaces each with a short note.
   */
  prompt: Prompt;
  /** Optional per-run system prompt; absent = the driver's default. */
  systemPrompt?: string;
  /** Optional per-run model; absent = the driver's configured default. */
  model?: string;
  /** Reasoning effort for this turn; absent = the driver's own default. */
  effort?: EffortLevel;
  /** The tools to expose for this run. */
  tools: B[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface AdapterContinueInput<B extends ToolBundle<any> = ToolBundle> extends AdapterRunInput<B> {
  /** The conversation id returned by a previous run/continue. */
  backendSessionId: string;
  /**
   * The context the previous turn ENDED at, in tokens, if the host recorded it (its last usage: input,
   * cached included, plus output). A driver that manages its own context starts the turn from this
   * measurement rather than from an estimate; one that does not ignores it.
   */
  contextTokens?: number;
}

/** One turn's usage. `inputTokens`/`cache*` describe the FINAL request (= context occupancy). */
export interface TurnUsage {
  model: string;
  costUsd: number;
  inputTokens: number; // fresh (uncached) input
  outputTokens: number;
  cacheReadTokens: number; // input served from cache
  cacheCreationTokens: number; // input written to cache
}

export interface AdapterRunResult {
  /** The agent's final text for the turn. */
  text: string;
  /** Conversation id: fresh for `run`, the same as the input's for `continueRun`. */
  backendSessionId: string;
  usage: TurnUsage;
}

/** A model a driver offers, for a picker. */
export interface ModelChoice {
  id: string; // the value passed as AdapterRunInput.model
  label: string; // human-readable name for the picker
  provider?: string; // grouping hint
  description?: string;
  /**
   * Short factual labels about what this model can do. Pre-rendered strings rather than flags, because
   * drivers do not know the same things: a `vision?: boolean` would be undefined on a driver that cannot
   * tell and read as "no" at a glance. An empty list means "this driver does not report it".
   */
  capabilities?: string[];
  /** The effort levels this model actually accepts, low → high. Absent or empty = do not offer the choice. */
  effortLevels?: EffortLevel[];
  /** The canonical wire id this row resolves to (`opus` → `claude-opus-…`), when the driver reports one. */
  canonicalId?: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface BackendAdapter<B extends ToolBundle<any> = ToolBundle> {
  readonly id: string;
  run(input: AdapterRunInput<B>, opts: AdapterRunOptions): Promise<AdapterRunResult>;
  continueRun(input: AdapterContinueInput<B>, opts: AdapterRunOptions): Promise<AdapterRunResult>;
}
