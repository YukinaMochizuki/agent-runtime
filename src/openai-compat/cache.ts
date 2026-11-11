import { SUMMARY_PREFIX } from "./compaction.ts";

/**
 * Prompt-cache breakpoints for the providers that need them marked.
 *
 * Some endpoints cache a request's prefix by themselves (OpenAI, DeepSeek, Gemini's implicit cache): for
 * those, a history that only grows at the end is all it takes, and that is what the loop keeps. Others
 * (Anthropic and Gemini models through OpenRouter, some self-hosted servers) cache only up to a
 * `cache_control` marker on a content part. For those, this marks three places on the wire body:
 *
 *   - the system prompt: identical on every request of a session;
 *   - the compaction summary, when there is one: identical until the next compaction;
 *   - the LAST message: the rolling breakpoint, so the next request (this one plus what follows) is read
 *     from cache up to here.
 *
 * Three, because Anthropic allows four and a marker costs a cache write. A marker goes on a content PART:
 * a string `content` becomes a one-part array, which every OpenAI-compatible endpoint accepts for these
 * roles; a message that already has parts gets it on its last part.
 */
export const CACHE_CONTROL = { type: "ephemeral" } as const;

type WireMessage = { role: string; content?: unknown; [k: string]: unknown };

function mark(m: WireMessage): WireMessage {
  if (typeof m.content === "string") return { ...m, content: [{ type: "text", text: m.content, cache_control: CACHE_CONTROL }] };
  if (Array.isArray(m.content) && m.content.length) {
    const parts = [...m.content] as Array<Record<string, unknown>>;
    parts[parts.length - 1] = { ...parts[parts.length - 1], cache_control: CACHE_CONTROL };
    return { ...m, content: parts };
  }
  return m; // an assistant message that is only tool calls: nothing to mark, and no need
}

/** The request body with the breakpoints above added (a `transformRequestBody` for the provider). */
export function withCacheBreakpoints(body: Record<string, unknown>): Record<string, unknown> {
  const messages = body.messages as WireMessage[] | undefined;
  if (!Array.isArray(messages) || !messages.length) return body;
  const at = new Set<number>();
  const system = messages.findIndex((m) => m.role === "system");
  if (system >= 0) at.add(system);
  const summary = messages.findIndex((m) => m.role === "user" && typeof m.content === "string" && m.content.startsWith(SUMMARY_PREFIX));
  if (summary >= 0) at.add(summary);
  at.add(messages.length - 1);
  return { ...body, messages: messages.map((m, i) => (at.has(i) ? mark(m) : m)) };
}
