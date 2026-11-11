/**
 * Turns the Agent SDK's message stream into the two callbacks the transcript is built from: `onToken`
 * for prose and `onTool` for a tool call (whose arrival is what flushes the prose before it).
 *
 * Prose is `text` AND thinking summaries. On Opus 5.5 and its generation the model's notes to the human
 * between tool calls are progress-update `thinking` blocks, not `text`; the session asks for
 * `summarized` display (claude-sdk.ts) and every non-empty summary is shown. Reasoning summaries cannot
 * be told apart from progress updates under that display, so they are shown too, deliberately.
 * Every thinking token is TAGGED (`onToken(text, true)`), so the transcript can keep it apart from the
 * reply and draw it a step quieter; that is as far as the two can be told apart.
 *
 * `onToolDraft` fires when a `tool_use` block OPENS, which is when the model starts writing the call's
 * arguments. `onTool` fires when it closes. The gap is the model writing, not the tool running, and for
 * a big `write_box` it is seconds long, so the live status says which of the two it is.
 *
 * Normally both come from `stream_event`s: text deltas, and a tool_use block closing. But Claude Code
 * falls back to a NON-STREAMING retry when a streamed request fails or yields no events, and then the
 * SDK emits only the complete `assistant` message — no stream events at all. Reading the stream alone,
 * that response's prose never reached the transcript: the `ask` after it still appeared (its handler
 * writes its own event), with the explanation it was asking about missing above it.
 *
 * So an `assistant` message whose id never streamed is replayed through the same callbacks. One that
 * did stream is ignored here, as before, because its content already went out delta by delta. The SDK
 * emits a response as one `assistant` message PER CONTENT BLOCK, all under the same id, so a replay is
 * de-duplicated by block (its text, or its tool_use id), never by marking the id done after the first.
 *
 * The fallback can follow a stream that got PART way: text already shown live, then the stream died,
 * the streamed retries ran out, and the non-streaming request carries the whole response. Replaying
 * that in full would print the streamed prefix twice, so the text an unfinished stream emitted is kept
 * and skipped when the replay starts with it. If it does not (the model answered differently the
 * second time), the replay is emitted whole: the partial is still on screen, but a duplicated sentence
 * is recoverable and a lost one is not. Thinking summaries follow the same rule block by block: the
 * summaries an unfinished stream showed are skipped by prefix, and every other one is shown, so a
 * progress update that never streamed is never dropped. Tool calls are not matched this way: the retry's tool_use ids
 * differ from the stream's, and a second `onTool` only re-flushes an empty prose segment.
 *
 * COMPACTION: Claude Code compacts a long conversation itself (automatically past its threshold, or on
 * `/compact`) and says so with a `system` message, subtype `compact_boundary`, carrying the trigger and the
 * tokens before (and, in newer versions, after). It is forwarded as `onCompaction` in the runtime's shared
 * shape (adapter.ts `CompactionEvent`), at its place in the stream, so a transcript can mark the point after
 * which the model sees only a summary.
 *
 * Not every unstreamed `assistant` message is a model response. Claude Code makes some up itself
 * ("API Error: Connection lost mid-response…"), marked `model: "<synthetic>"` and
 * `is_api_error_message`. Those are the CLI talking, not the agent, so they are never replayed as prose.
 */

import type { CompactionEvent } from "../adapter.ts";

export interface StreamCallbacks {
  onToken?: (text: string, thinking?: boolean) => void;
  onTool?: (name: string, input?: unknown) => void;
  onToolDraft?: (name: string) => void;
  onCompaction?: (event: CompactionEvent) => void;
}

/** The compaction a `system` / `compact_boundary` message reports, or null for any other message. */
export function compactionOf(message: { type: string; subtype?: unknown; compact_metadata?: unknown }): CompactionEvent | null {
  if (message.type !== "system" || message.subtype !== "compact_boundary") return null;
  // A beta surface: guard each field rather than trust the type.
  const m = (message.compact_metadata ?? {}) as { trigger?: unknown; pre_tokens?: unknown; post_tokens?: unknown };
  return {
    trigger: m.trigger === "manual" ? "manual" : "auto",
    tokensBefore: typeof m.pre_tokens === "number" ? m.pre_tokens : 0,
    ...(typeof m.post_tokens === "number" ? { tokensAfter: m.post_tokens } : {}),
  };
}

type ContentBlock = { type?: string; text?: string; thinking?: string; id?: string; name?: string; input?: unknown };

export class ClaudeStreamTranslator {
  /** Message ids that arrived as stream events: their `assistant` message is already accounted for. */
  private readonly streamed = new Set<string>();
  /** The tool_use block being streamed, by block index, accumulating its input JSON. */
  private readonly toolBlocks = new Map<number, { id: string; name: string; json: string }>();
  /** What the current, not-yet-stopped streamed message has emitted — for de-duplicating a fallback. */
  private partialText = "";
  private partialTools = new Set<string>();
  /** The fallback response being replayed, and what of it is still to skip as already streamed. */
  private fallbackId: string | null = null;
  private skipText = "";
  private skipTools = new Set<string>();
  /** Blocks of unstreamed responses already replayed, so a repeated message is not emitted twice. */
  private readonly replayed = new Set<string>();
  /**
   * The thinking summaries an unfinished stream already showed, one per block in order, so a fallback
   * replay skips exactly those (by prefix, as for text) and still shows every block that never streamed.
   */
  private partialThinking = new Map<number, string>();
  private skipThinking: string[] = [];
  /** The streamed message in progress, so a block can be told apart from the one before it. */
  private currentId = "";
  /**
   * Prose since the last tool call, and which block wrote it last. Two blocks of the same kind in one
   * stretch (two thinking summaries, or two text blocks) get a blank line between them, or the first's
   * last sentence and the second's first run together into one paragraph. A thinking block followed
   * by text needs none: the switch itself starts a new transcript row.
   */
  private segOpen = false;
  private lastBlock: string | null = null;
  private lastThinking = false;
  private tail = "";

  constructor(private readonly cb: StreamCallbacks) {}

  /** Feed one SDK message. `stream_event`, `assistant` and a compaction boundary matter; anything else is ignored. */
  handle(message: { type: string; event?: unknown; message?: unknown; is_api_error_message?: unknown; subtype?: unknown; compact_metadata?: unknown }): void {
    if (message.type === "stream_event") this.streamEvent(message.event);
    else if (message.type === "assistant" && message.is_api_error_message !== true) this.assistant(message.message);
    else if (message.type === "system") {
      const c = compactionOf(message);
      if (c) this.cb.onCompaction?.(c);
    }
  }

  private streamEvent(raw: unknown): void {
    // Deeply-nested beta stream event; guard each field rather than trust the type.
    const ev = (raw ?? {}) as {
      type?: string;
      index?: number;
      message?: { id?: string };
      delta?: { type?: string; text?: string; thinking?: string; partial_json?: string };
      content_block?: ContentBlock;
    };
    switch (ev.type) {
      case "message_start":
        if (typeof ev.message?.id === "string") this.streamed.add(ev.message.id);
        this.currentId = ev.message?.id ?? "";
        this.resetPartial();
        this.fallbackId = null;
        return;
      case "message_stop":
        this.resetPartial(); // a complete stream: nothing for a fallback to de-duplicate against
        return;
      case "content_block_start":
        if (ev.content_block?.type === "tool_use" && typeof ev.content_block.name === "string") {
          this.toolBlocks.set(ev.index ?? -1, { id: ev.content_block.id ?? "", name: ev.content_block.name, json: "" });
          this.cb.onToolDraft?.(ev.content_block.name);
        }
        return;
      case "content_block_delta":
        if (ev.delta?.type === "text_delta" && typeof ev.delta.text === "string") {
          this.partialText += ev.delta.text;
          this.prose(`${this.currentId}#${ev.index ?? -1}`, ev.delta.text, false);
        } else if (ev.delta?.type === "thinking_delta" && typeof ev.delta.thinking === "string" && ev.delta.thinking) {
          // A thinking SUMMARY (the session asks for `summarized` display). On Opus 5.5 this is where the
          // model's notes to the human between tool calls arrive, so it is shown as prose, tagged as thinking.
          const i = ev.index ?? -1;
          this.partialThinking.set(i, (this.partialThinking.get(i) ?? "") + ev.delta.thinking);
          this.prose(`${this.currentId}#${ev.index ?? -1}`, ev.delta.thinking, true);
        } else if (ev.delta?.type === "input_json_delta" && typeof ev.delta.partial_json === "string") {
          const b = this.toolBlocks.get(ev.index ?? -1);
          if (b) b.json += ev.delta.partial_json;
        }
        return;
      case "content_block_stop": {
        const b = this.toolBlocks.get(ev.index ?? -1);
        if (!b) return;
        this.toolBlocks.delete(ev.index ?? -1);
        let parsed: unknown;
        try {
          parsed = b.json ? JSON.parse(b.json) : undefined;
        } catch {
          parsed = undefined;
        }
        if (b.id) this.partialTools.add(b.id);
        this.tool(b.name, parsed);
        return;
      }
    }
  }

  private assistant(raw: unknown): void {
    const msg = (raw ?? {}) as { id?: string; model?: string; content?: unknown };
    if (msg.model === "<synthetic>") return;
    if (typeof msg.id === "string" && this.streamed.has(msg.id)) return;
    if (!Array.isArray(msg.content)) return;
    const id = typeof msg.id === "string" ? msg.id : "";
    // The first message of a response that never streamed (the non-streaming fallback): whatever an
    // unfinished stream before it emitted becomes what this replay skips.
    if (id !== this.fallbackId) {
      this.fallbackId = id;
      this.skipText = this.partialText;
      this.skipTools = this.partialTools;
      this.skipThinking = [...this.partialThinking.values()];
      this.resetPartial();
    }
    for (const block of msg.content as ContentBlock[]) {
      if (block.type === "thinking" && typeof block.thinking === "string" && block.thinking) {
        let key = `${id}|thinking|${block.thinking}`;
        if (!this.firstReplay(key)) continue;
        let text = block.thinking;
        const shown = this.skipThinking[0];
        if (shown !== undefined) {
          if (text.startsWith(shown)) {
            text = text.slice(shown.length);
            this.skipThinking.shift();
            key = this.lastBlock ?? key;
          } else if (shown.startsWith(text)) {
            this.skipThinking[0] = shown.slice(text.length);
            continue;
          } else {
            this.skipThinking.shift(); // summarised differently the second time: show it whole
          }
        }
        this.prose(key, text, true);
      } else if (block.type === "text" && typeof block.text === "string") {
        if (!this.firstReplay(`${id}|text|${block.text}`)) continue;
        let text = block.text;
        // The rest of a block the stream had started continues it, so it takes that block's key and
        // gets no separator; a block of its own gets its own key.
        let key = `${id}|text|${block.text}`;
        if (this.skipText) {
          if (text.startsWith(this.skipText)) {
            text = text.slice(this.skipText.length);
            this.skipText = "";
            key = this.lastBlock ?? key;
          } else if (this.skipText.startsWith(text)) {
            this.skipText = this.skipText.slice(text.length);
            continue;
          } else {
            this.skipText = "";
          }
        }
        this.prose(key, text, false);
      } else if (block.type === "tool_use" && typeof block.name === "string") {
        if (!this.firstReplay(`${id}|tool|${block.id ?? block.name}`)) continue;
        if (block.id && this.skipTools.has(block.id)) continue;
        this.tool(block.name, block.input);
      }
    }
  }

  private prose(block: string, text: string, thinking: boolean): void {
    if (!text) return;
    // A switch between thinking and the reply starts a new transcript row (the sink cuts there), so it
    // needs no blank line to keep the two paragraphs apart; one would only lead the new row.
    const switched = this.segOpen && thinking !== this.lastThinking;
    if (switched) this.tail = "";
    else if (this.segOpen && block !== this.lastBlock && !this.tail.endsWith("\n\n")) {
      const sep = this.tail.endsWith("\n") ? "\n" : "\n\n";
      this.cb.onToken?.(sep, thinking);
    }
    this.cb.onToken?.(text, thinking);
    this.segOpen = true;
    this.lastBlock = block;
    this.lastThinking = thinking;
    this.tail = (this.tail + text).slice(-2);
  }

  private tool(name: string, input: unknown): void {
    // The sink flushes its prose segment on a tool call, so the next prose starts a fresh one.
    this.segOpen = false;
    this.lastBlock = null;
    this.lastThinking = false;
    this.tail = "";
    this.cb.onTool?.(name, input);
  }

  private firstReplay(key: string): boolean {
    if (this.replayed.has(key)) return false;
    this.replayed.add(key);
    return true;
  }

  private resetPartial(): void {
    this.partialText = "";
    this.partialThinking = new Map();
    this.partialTools = new Set();
    this.toolBlocks.clear();
  }
}
