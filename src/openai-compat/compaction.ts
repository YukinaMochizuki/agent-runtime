import type { ModelMessage, ToolResultPart } from "ai";

/**
 * Keeping a conversation inside the model's context window, without throwing away the prompt cache.
 *
 * The two pull against each other. A provider caches the PREFIX of a request, so the cheapest history is
 * one that only ever grows at the end; every rewrite loses the cache from the rewritten point on. Trimming
 * a little every turn (a sliding window) would therefore miss the cache on every request. So history is
 * left alone until it crosses a threshold, then compacted ONCE, by enough to buy a long stretch of
 * append-only turns again, and the compacted version is stored (`ConversationStore.compact`) so the next
 * turn starts from it instead of compacting again.
 *
 * The default compactor works in two stages, cheapest first:
 *   1. PRUNE: in the older turns, a long tool output becomes a one-line placeholder and an image becomes a
 *      note. What was said and what was called stay; what a tool printed is usually what fills a context.
 *   2. SUMMARIZE, only if pruning was not enough: the older turns become one summary written by the model,
 *      followed by the recent turns as they were.
 *
 * It keeps the last few STEPS (an assistant message and the tool results that answer it) exactly as they
 * are (unless even a summary is not enough: then their tool output is pruned too, all but the latest
 * step's), and only ever cuts before an assistant message or a user turn, so a tool call is never separated
 * from its result. Steps, not turns: one turn of an agent at work can hold hundreds of tool calls, and a
 * compactor that only cut between turns would have nothing to cut in exactly the session that needs it.
 * When the cut falls inside a turn, that turn's opening user message is kept word for word after the
 * summary: the request being worked on is the one thing a summary must not paraphrase away.
 */

export interface CompactionOptions {
  /** The model's context window, in tokens. */
  contextWindow: number;
  /** Compact when the estimated context exceeds this fraction of the window. Default 0.75. */
  threshold?: number;
  /** What compaction aims to get under, as a fraction of the window. Default 0.5. */
  target?: number;
  /** Steps at the end kept exactly as they are. Default 8. */
  keepRecentSteps?: number;
  /** Replace the default two-stage compactor. */
  compactor?: Compactor;
  /** The model the summary is written by (default: the turn's own model). */
  summaryModel?: string;
}

export interface CompactionInput {
  messages: ModelMessage[];
  /** The best estimate of the context those messages take, in tokens. */
  usedTokens: number;
  contextWindow: number;
  target: number;
  keepRecentSteps: number;
  /** Ask the model for a summary of a plain-text transcript. */
  summarize: (transcript: string) => Promise<string>;
}

/** A new history to continue from, or null to leave it as it is. */
export type Compactor = (input: CompactionInput) => Promise<ModelMessage[] | null>;

/** The opening of the message a summary is carried in: how it is recognised later (a cache breakpoint, a re-summary). */
export const SUMMARY_PREFIX = "[Summary of the earlier conversation, written to save context. The recent turns follow it unchanged.]";

/** Tokens an image is counted as: its base64 length would wildly overstate what a model bills for it. */
const IMAGE_TOKENS = 1_500;
/** A tool output longer than this (in characters) is pruned in the older turns. */
const PRUNE_OVER_CHARS = 400;

/**
 * A rough token count of a text: about four ASCII characters a token, and about one token for every other
 * character. Counting everything at four a character undercounts CJK text three to four times over, which
 * would let a conversation in Chinese overflow before the estimate ever crossed a threshold.
 */
function textTokens(s: string): number {
  let ascii = 0;
  let other = 0;
  for (const ch of s) {
    if (ch.charCodeAt(0) < 128) ascii++;
    else other++;
  }
  return ascii / 4 + other;
}

/** A rough token count for messages (see textTokens), images at a flat rate. */
export function estimateTokens(messages: ModelMessage[]): number {
  let tokens = 0;
  for (const m of messages) {
    if (typeof m.content === "string") {
      tokens += textTokens(m.content);
      continue;
    }
    for (const p of m.content as Array<Record<string, unknown>>) {
      tokens += p.type === "file" || p.type === "image" ? IMAGE_TOKENS : textTokens(JSON.stringify(p));
    }
  }
  return Math.ceil(tokens);
}

/**
 * How the user message carrying a tool's images opens (the loop writes it right after the tool results).
 * It is recognised by this, not by position: a user message after a tool message is ALSO what a new turn
 * looks like when the turn before it was stopped, failed, or repaired after its tool results.
 */
export const TOOL_IMAGES_PREFIX = "[Images returned by the tool calls above]";

export function isToolImages(m: ModelMessage): boolean {
  if (m.role !== "user" || typeof m.content === "string") return false;
  const first = m.content[0];
  return first?.type === "text" && first.text.startsWith(TOOL_IMAGES_PREFIX);
}

/** Where each turn starts: a user message, unless it is the one carrying a tool's images. */
export function turnStarts(messages: ModelMessage[]): number[] {
  const starts: number[] = [];
  messages.forEach((m, i) => {
    if (m.role === "user" && !isToolImages(m)) starts.push(i);
  });
  return starts;
}

/**
 * Split into the older part (may be compacted) and the recent part (kept exactly): the recent part starts at
 * the `keepRecentSteps`-th assistant message from the end, or at the start of that step's turn if the turn
 * begins right before it. `opening` is the user message that opened the turn the cut falls inside, when it
 * falls inside one: it is in `older`, and a summary keeps it word for word.
 */
export function splitRecent(messages: ModelMessage[], keepRecentSteps: number): { older: ModelMessage[]; recent: ModelMessage[]; opening: ModelMessage | null } {
  const assistants: number[] = [];
  messages.forEach((m, i) => m.role === "assistant" && assistants.push(i));
  if (assistants.length <= keepRecentSteps) return { older: [], recent: messages, opening: null };
  let cut = assistants[assistants.length - Math.max(1, keepRecentSteps)]!;
  const starts = turnStarts(messages);
  const turn = starts.filter((s) => s < cut).at(-1);
  // The step opens its turn: cut before the turn's user message instead, so nothing is left dangling.
  if (turn !== undefined && turn === cut - 1) cut = turn;
  const opening = turn !== undefined && turn < cut ? messages[turn]! : null;
  return { older: messages.slice(0, cut), recent: messages.slice(cut), opening };
}

/** Stage 1: the older turns with long tool outputs and images replaced by short notes. */
export function prune(older: ModelMessage[]): ModelMessage[] {
  return older.map((m): ModelMessage => {
    if (m.role === "tool") {
      return {
        ...m,
        content: m.content.map((p) => {
          if (p.type !== "tool-result") return p;
          const text = p.output.type === "text" || p.output.type === "error-text" ? p.output.value : JSON.stringify(p.output);
          if (text.length <= PRUNE_OVER_CHARS) return p;
          const note = `[earlier ${p.toolName} output, ${text.length} characters, omitted to save context. Its start: ${text.slice(0, 160).replace(/\s+/g, " ")}…]`;
          const output: ToolResultPart["output"] = p.output.type === "error-text" ? { type: "error-text", value: note } : { type: "text", value: note };
          return { ...p, output };
        }),
      };
    }
    if (m.role === "user" && typeof m.content !== "string") {
      return {
        ...m,
        content: m.content.map((p) => (p.type === "file" || p.type === "image" ? { type: "text" as const, text: "[an earlier image, omitted to save context]" } : p)),
      };
    }
    return m;
  });
}

/** The older turns as a plain-text transcript, for the summary: no tool schemas, no wire format to get right. */
export function transcriptOf(messages: ModelMessage[], maxResultChars = 2_000): string {
  const lines: string[] = [];
  const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n)}… [${s.length - n} more characters]` : s);
  for (const m of messages) {
    if (typeof m.content === "string") {
      lines.push(`${m.role.toUpperCase()}: ${m.content}`);
      continue;
    }
    for (const p of m.content as Array<Record<string, unknown>>) {
      if (p.type === "text") lines.push(`${m.role.toUpperCase()}: ${p.text as string}`);
      else if (p.type === "reasoning") continue;
      else if (p.type === "tool-call") lines.push(`ASSISTANT called ${p.toolName as string}(${clip(JSON.stringify(p.input), 500)})`);
      else if (p.type === "tool-result") {
        const out = p.output as { type: string; value?: unknown };
        const text = typeof out.value === "string" ? out.value : JSON.stringify(out.value ?? out);
        lines.push(`TOOL ${p.toolName as string}${out.type === "error-text" ? " (error)" : ""}: ${clip(text, maxResultChars)}`);
      } else if (p.type === "file" || p.type === "image") lines.push(`${m.role.toUpperCase()}: [an image]`);
    }
  }
  return lines.join("\n");
}

/** The instruction the summary is written to. */
export const SUMMARY_INSTRUCTIONS =
  "You are compacting the earlier part of a long working conversation between a user and an AI agent so the agent can keep going in a smaller context. " +
  "Write a summary the agent will read INSTEAD of that part. Keep: what the user wants and every decision they made; facts established (names, paths, ids, values, commands that worked or failed and why); " +
  "what the agent did and its results; what is still open. Drop chit-chat and anything superseded. Be concrete and complete; use short sections or bullet lists. " +
  "If the transcript starts with an earlier summary, fold it in.";

/**
 * The default: prune the older steps; if that is not enough, summarize them; if even that is not enough
 * (a few huge steps, all recent), prune the tool output of every step but the latest as a last resort.
 * A summary that cannot be had (`summarize` answered "") leaves the pruned history.
 */
export const defaultCompactor: Compactor = async (input) => {
  const { older, recent, opening } = splitRecent(input.messages, input.keepRecentSteps);
  const goal = input.contextWindow * input.target;
  // Scale the estimate to the measured size, so a rough count does not decide on its own.
  const scale = input.usedTokens > 0 ? input.usedTokens / Math.max(1, estimateTokens(input.messages)) : 1;
  const fits = (ms: ModelMessage[]) => estimateTokens(ms) * scale <= goal;
  let next: ModelMessage[] = input.messages;
  if (older.length) {
    const pruned = prune(older);
    next = [...pruned, ...recent];
    if (!fits(next)) {
      const summary = (await input.summarize(transcriptOf(pruned))).trim();
      // The request being worked on, word for word, when the cut fell inside its turn. Pruned, so an
      // image it carried does not come back at full weight.
      if (summary) next = [{ role: "user", content: `${SUMMARY_PREFIX}\n\n${summary}` }, ...(opening ? prune([opening]) : []), ...recent];
    }
  }
  if (!fits(next)) next = pruneAllButLastStep(next);
  return JSON.stringify(next) === JSON.stringify(input.messages) ? null : next;
};

/** Prune everything before the latest assistant message: the latest step is what the agent is acting on. */
export function pruneAllButLastStep(messages: ModelMessage[]): ModelMessage[] {
  let last = -1;
  messages.forEach((m, i) => m.role === "assistant" && (last = i));
  return last <= 0 ? messages : [...prune(messages.slice(0, last)), ...messages.slice(last)];
}
