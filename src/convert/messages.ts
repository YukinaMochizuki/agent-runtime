import type { AssistantModelMessage, ModelMessage, ToolModelMessage, ToolResultPart, UserModelMessage } from "ai";
import type { ClaudeBlock, ClaudeMessage } from "../claude/session.ts";
import { unansweredCalls, wireToolName } from "../openai-compat/backend.ts";
import { isToolImages, TOOL_IMAGES_PREFIX } from "../openai-compat/compaction.ts";

/**
 * One conversation, two drivers' forms: a Claude Code session's messages (the Anthropic Messages API's
 * shape, which is what its transcript entries carry) and the AI SDK `ModelMessage[]` the OpenAI-compatible
 * driver keeps through its `ConversationStore`. These functions turn one into the other, so a host can
 * carry a conversation from one driver to the other. Pure: no files, no SDK (the session files are
 * ../claude/session.ts).
 *
 * WHAT CARRIES: text; images (a prompt's and a tool's, inline or by URL); a text document as its text, and
 * toward Claude a PDF or text `file` as a `document`; every tool call with its result, paired by id. Nothing
 * is added to say a message was carried.
 * WHAT DOES NOT:
 *   - reasoning. A thinking block is signed by the model that wrote it; Claude Code drops thinking a
 *     different model recorded, and endpoints disagree on whether they take another's. So it is dropped
 *     both ways, and an assistant message that held only reasoning goes with it;
 *   - what one side cannot hold, or is not known to take: a PDF toward an OpenAI-compatible endpoint (many
 *     refuse a file part, and one refused fails every turn after it), a file by a provider's reference, a
 *     file type Claude has no block for. Dropped, not described;
 *   - text that is empty or only whitespace: the Messages API refuses such a block. Text otherwise carries
 *     as written, blank lines included.
 *
 * THE SAME TOOLS, NAMED BY EACH DRIVER: Claude publishes `mcp__<server>__<tool>`, the OpenAI-compatible
 * driver `<server>__<tool>` (`wireToolName`). Names are mapped both ways; a name of neither form passes
 * through unchanged.
 *
 * IMAGES A TOOL RETURNED live inside its `tool_result` on Claude, and on the OpenAI-compatible side in one
 * user message right after the tool results (`TOOL_IMAGES_PREFIX`), each result's text naming its images in
 * their place (`toToolOutput`). Each side's convention is rebuilt from the other's, numbering included.
 *
 * WHAT IS CLAUDE CODE'S OWN and is never carried: the `[Image: source: <path>]` line it writes next to an
 * image (a path on the host), a local command's echo (`<command-name>/compact</command-name>`,
 * `<local-command-stdout>…`) and its `[Request interrupted by user]` lines. A compaction's summary IS
 * carried: it is what the model works from.
 *
 * EVERY CALL ANSWERED: the Messages API refuses a `tool_use` with no `tool_result`, so a call nothing in the
 * history answers (a turn whose process died mid-tool) gets an error result, beside the results that did
 * arrive. A result whose call is not in what is converted is dropped (an endpoint refuses that too).
 *
 * Ids: Anthropic takes a tool-use id of `[A-Za-z0-9_-]` only, so another provider's ids (`functions.x:0`)
 * are mapped onto that alphabet, the call and its result alike.
 */

/** The text Claude Code writes beside an image it was given: a path on the host, never carried. */
const IMAGE_SOURCE_LINE = /^\[Image: source: [^\]]*\]$/;
/** Claude Code's own user lines: a local command's echo, a stop. Not the conversation's. */
const CLAUDE_CODE_LINE = /^\s*(<(command-name|command-message|command-args|local-command-stdout|local-command-stderr|local-command-caveat)>|\[Request interrupted by user( for tool use)?\]\s*$)/;
const ownLine = (text: string) => IMAGE_SOURCE_LINE.test(text.trim()) || CLAUDE_CODE_LINE.test(text);

export interface ConvertOptions {
  /** Claude's tool name for an OpenAI-compatible one. Default: `mcp__<name>` for a `<server>__<tool>` name. */
  claudeToolName?: (wireName: string) => string;
  /** The OpenAI-compatible tool name for a Claude one. Default: `wireToolName` of an `mcp__<server>__<tool>`. */
  wireToolName?: (claudeName: string) => string;
}

export function defaultWireToolName(claudeName: string): string {
  const m = /^mcp__(.+?)__(.+)$/.exec(claudeName);
  return m ? wireToolName(m[1]!, m[2]!) : claudeName;
}

export function defaultClaudeToolName(wireName: string): string {
  return wireName.includes("__") && !wireName.startsWith("mcp__") ? `mcp__${wireName}` : wireName;
}

/** A tool-use id Anthropic accepts: `[A-Za-z0-9_-]+`. The same input always maps to the same id. */
export function claudeToolUseId(id: string): string {
  const clean = id.replace(/[^A-Za-z0-9_-]/g, "_");
  return clean || "tool_use";
}

const imageLine = (n: number, mediaType: string) => `[image ${n} (${mediaType}): sent in the next message]`;
const IMAGE_LINE = /^\[image (\d+) \(([^)]+)\): sent in the next message\]$/;
const NO_RESULT = "No result: the turn ended before this call returned.";

// ---------------------------------------------------------------------------------------------------------
// What a part holds, in either form

/** Inline bytes as base64, a URL, inline text, or nothing either side can hold (a provider's reference). */
type Payload = { base64: string } | { url: string } | { text: string } | null;

function payloadOf(data: unknown): Payload {
  if (typeof data === "string") {
    // The AI SDK reads a URL-like string as a URL, a `data:` URL as its bytes, anything else as base64.
    const inline = /^data:[^;,]*;base64,(.*)$/s.exec(data);
    if (inline) return { base64: inline[1]! };
    if (/^https?:\/\//i.test(data)) return { url: data };
    return { base64: data };
  }
  if (data instanceof URL) return data.protocol === "data:" ? payloadOf(data.href) : { url: data.href };
  if (data instanceof Uint8Array) return { base64: Buffer.from(data).toString("base64") };
  if (data instanceof ArrayBuffer) return { base64: Buffer.from(new Uint8Array(data)).toString("base64") };
  if (data && typeof data === "object" && "type" in data) {
    const d = data as { type: string; data?: unknown; url?: unknown; text?: unknown };
    if (d.type === "data") return payloadOf(d.data);
    if (d.type === "url" && (typeof d.url === "string" || d.url instanceof URL)) return { url: String(d.url) };
    if (d.type === "text" && typeof d.text === "string") return { text: d.text };
  }
  return null;
}

/** A Claude block for a file or image part; null when Claude has no block for it. */
function claudeFileBlock(mediaType: string, data: unknown): ClaudeBlock | null {
  const p = payloadOf(data);
  if (!p) return null;
  if (mediaType.startsWith("image/") || mediaType === "image") {
    if ("base64" in p) return { type: "image", source: { type: "base64", media_type: mediaType === "image" ? "image/png" : mediaType, data: p.base64 } };
    if ("url" in p) return { type: "image", source: { type: "url", url: p.url } };
    return null;
  }
  if (mediaType === "application/pdf") {
    if ("base64" in p) return { type: "document", source: { type: "base64", media_type: "application/pdf", data: p.base64 } };
    if ("url" in p) return { type: "document", source: { type: "url", url: p.url } };
    return null;
  }
  if (mediaType.startsWith("text/")) {
    const text = "text" in p ? p.text : "base64" in p ? Buffer.from(p.base64, "base64").toString("utf8") : null;
    return text !== null ? { type: "document", source: { type: "text", media_type: "text/plain", data: text } } : null;
  }
  return null;
}

/**
 * What a Claude image or document block carries over as: an image as a `file` part, a text document as its
 * text. A PDF does not: an OpenAI-compatible endpoint is not known to take one (many refuse a file part,
 * and the provider cannot send a PDF by URL at all), and one refused fails every turn after it.
 */
function carriedOf(b: ClaudeBlock): { image: { type: "file"; data: string; mediaType: string } } | { text: string } | null {
  const s = (b as { source?: { type?: string; media_type?: string; data?: unknown; url?: unknown } }).source;
  if (!s) return null;
  if (b.type === "image") {
    if (s.type === "base64" && typeof s.data === "string") return { image: { type: "file", data: s.data, mediaType: s.media_type ?? "image/png" } };
    if (s.type === "url" && typeof s.url === "string") return { image: { type: "file", data: s.url, mediaType: s.media_type ?? "image" } };
    return null;
  }
  if (b.type === "document" && s.type === "text" && typeof s.data === "string") return { text: s.data };
  return null;
}

// ---------------------------------------------------------------------------------------------------------
// Claude → ModelMessage[]

/**
 * A Claude conversation as the OpenAI-compatible driver keeps one. `messages` is in order, user and
 * assistant (Claude Code writes one transcript entry per block: merge them by message first, as
 * ../claude/session.ts does).
 */
export function claudeToModelMessages(messages: ClaudeMessage[], opts: ConvertOptions = {}): ModelMessage[] {
  const toWire = opts.wireToolName ?? defaultWireToolName;
  const out: ModelMessage[] = [];
  const callNames = new Map<string, string>();
  for (const m of messages) {
    if (m.role === "assistant") {
      const blocks: ClaudeBlock[] = typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content;
      const parts: Exclude<AssistantModelMessage["content"], string> = [];
      for (const b of blocks) {
        if (b.type === "text" && typeof b.text === "string" && b.text.trim()) parts.push({ type: "text", text: b.text });
        else if (b.type === "tool_use") {
          const u = b as unknown as { id: string; name: string; input: unknown };
          const name = toWire(u.name);
          callNames.set(u.id, name);
          parts.push({ type: "tool-call", toolCallId: u.id, toolName: name, input: u.input });
        }
        // thinking / redacted_thinking: not carried (see the header)
      }
      if (parts.length) out.push({ role: "assistant", content: parts });
      continue;
    }
    if (typeof m.content === "string") {
      if (m.content.trim() && !ownLine(m.content)) out.push({ role: "user", content: m.content });
      continue;
    }
    // Only results whose call is in what is converted: an orphan result is refused by every endpoint.
    const results = m.content.filter((b): b is ClaudeBlock & { tool_use_id: string } => b.type === "tool_result" && callNames.has(String(b.tool_use_id)));
    if (results.length) {
      const images: Array<{ type: "file"; data: string; mediaType: string }> = [];
      const content: ToolResultPart[] = results.map((r) => {
        const raw = (r as { content?: string | ClaudeBlock[] }).content;
        const blocks: ClaudeBlock[] = typeof raw === "string" ? [{ type: "text", text: raw }] : (raw ?? []);
        let n = 0;
        const lines: string[] = [];
        for (const c of blocks) {
          if (c.type === "text" && typeof c.text === "string") {
            if (!ownLine(c.text)) lines.push(c.text);
          } else {
            const carried = carriedOf(c);
            if (carried && "image" in carried) {
              images.push(carried.image);
              lines.push(imageLine(++n, carried.image.mediaType));
            } else if (carried) lines.push(carried.text);
          }
        }
        const value = lines.join("\n");
        const isError = (r as { is_error?: boolean }).is_error === true;
        return { type: "tool-result", toolCallId: r.tool_use_id, toolName: callNames.get(r.tool_use_id)!, output: isError ? { type: "error-text", value } : { type: "text", value } };
      });
      out.push({ role: "tool", content } satisfies ToolModelMessage);
      if (images.length) out.push({ role: "user", content: [{ type: "text", text: `${TOOL_IMAGES_PREFIX} (${images.length}):` }, ...images] });
    }
    const rest = userParts(m.content.filter((b) => b.type !== "tool_result"));
    if (rest.length) out.push({ role: "user", content: rest });
  }
  return out;
}

function userParts(blocks: ClaudeBlock[]): Exclude<UserModelMessage["content"], string> {
  const parts: Exclude<UserModelMessage["content"], string> = [];
  for (const b of blocks) {
    if (b.type === "text" && typeof b.text === "string") {
      if (b.text.trim() && !ownLine(b.text)) parts.push({ type: "text", text: b.text });
    } else {
      const carried = carriedOf(b);
      if (carried && "image" in carried) parts.push(carried.image);
      else if (carried?.text.trim()) parts.push({ type: "text", text: carried.text });
    }
  }
  return parts;
}

// ---------------------------------------------------------------------------------------------------------
// ModelMessage[] → Claude

/**
 * An OpenAI-compatible conversation as Claude messages, ready to be written into a session. Consecutive
 * messages of one role are merged (the Messages API alternates, its `tool_result` blocks first), a tool's
 * images go back into its `tool_result`, and every call nothing answered gets an error result.
 */
export function modelMessagesToClaude(messages: ModelMessage[], opts: ConvertOptions = {}): ClaudeMessage[] {
  const toClaude = opts.claudeToolName ?? defaultClaudeToolName;
  const out: Array<{ role: "user" | "assistant"; content: ClaudeBlock[] }> = [];
  const push = (role: "user" | "assistant", blocks: ClaudeBlock[]) => {
    if (!blocks.length) return;
    const prev = out.at(-1);
    if (prev?.role === role) prev.content.push(...blocks);
    else out.push({ role, content: blocks });
  };
  const dangling = new Set(unansweredCalls(messages).map((d) => d.toolCallId));
  /** The unanswered calls of the last assistant message, still to be given a result. */
  let owed: string[] = [];
  const noResults = (ids: string[]): ClaudeBlock[] => ids.map((id) => ({ type: "tool_result", tool_use_id: claudeToolUseId(id), content: [{ type: "text", text: NO_RESULT }], is_error: true }));
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i]!;
    if (m.role === "system") continue;
    if (m.role === "assistant") {
      const parts = typeof m.content === "string" ? [{ type: "text" as const, text: m.content }] : m.content;
      const blocks: ClaudeBlock[] = [];
      for (const p of parts) {
        if (p.type === "text" && p.text.trim()) blocks.push({ type: "text", text: p.text });
        else if (p.type === "tool-call") blocks.push({ type: "tool_use", id: claudeToolUseId(p.toolCallId), name: toClaude(p.toolName), input: p.input ?? {} });
      }
      push("assistant", blocks);
      owed = parts.flatMap((p) => (p.type === "tool-call" && dangling.has(p.toolCallId) ? [p.toolCallId] : []));
      // No tool message comes to answer them: answer them here.
      if (owed.length && messages[i + 1]?.role !== "tool") {
        push("user", noResults(owed));
        owed = [];
      }
      continue;
    }
    if (m.role === "tool") {
      // The tool images message that follows these results, if any: its images go back into the results.
      const next = messages[i + 1];
      const pool = next && isToolImages(next) && typeof next.content !== "string" ? next.content.flatMap((p) => (p.type === "file" || p.type === "image" ? [p] : [])) : [];
      if (pool.length) i++;
      const blocks: ClaudeBlock[] = [];
      for (const p of m.content) {
        if (p.type !== "tool-result") continue;
        const { content, isError } = resultContent(p.output, pool);
        blocks.push({ type: "tool_result", tool_use_id: claudeToolUseId(p.toolCallId), ...(content.length ? { content } : {}), ...(isError ? { is_error: true } : {}) });
      }
      // Calls this message did not answer, beside the ones it did.
      blocks.push(...noResults(owed));
      owed = [];
      // Images the lines did not account for still reach the model, after the results.
      for (const p of pool) {
        const block = claudeFileBlock(p.mediaType ?? "image/png", p.type === "image" ? p.image : p.data);
        if (block) blocks.push(block);
      }
      push("user", blocks);
      continue;
    }
    // user
    if (typeof m.content === "string") {
      if (m.content.trim()) push("user", [{ type: "text", text: m.content }]);
      continue;
    }
    const blocks: ClaudeBlock[] = [];
    for (const p of m.content) {
      if (p.type === "text") {
        if (p.text.trim()) blocks.push({ type: "text", text: p.text });
      } else if (p.type === "image" || p.type === "file") {
        const block = claudeFileBlock(p.mediaType ?? "image/png", p.type === "image" ? p.image : p.data);
        if (block) blocks.push(block);
      }
    }
    push("user", blocks);
  }
  return out;
}

/**
 * A tool result's output as `tool_result` content: text and images, an image named in its line put in its
 * place. Text is kept as written, blank lines included; only a block with nothing but whitespace is left out.
 */
function resultContent(o: ToolResultPart["output"], pool: Array<{ type: string; mediaType?: string; data?: unknown; image?: unknown }>): { content: ClaudeBlock[]; isError: boolean } {
  const content: ClaudeBlock[] = [];
  const text = (t: string) => {
    if (t.trim()) content.push({ type: "text", text: t });
  };
  switch (o.type) {
    case "text":
    case "error-text": {
      // The text between image lines, each stretch as it was written.
      let stretch: string[] = [];
      for (const line of o.value.split("\n")) {
        const img = IMAGE_LINE.test(line) ? pool.shift() : undefined;
        const block = img ? claudeFileBlock(img.mediaType ?? "image/png", img.type === "image" ? img.image : img.data) : null;
        if (!block) {
          stretch.push(line);
          continue;
        }
        text(stretch.join("\n"));
        stretch = [];
        content.push(block);
      }
      text(stretch.join("\n"));
      return { content, isError: o.type === "error-text" };
    }
    case "json":
    case "error-json":
      text(JSON.stringify(o.value));
      return { content, isError: o.type === "error-json" };
    case "execution-denied":
      text(o.reason ? `Not run: ${o.reason}` : "Not run: the call was denied.");
      return { content, isError: true };
    case "content":
      for (const part of o.value) {
        if (part.type === "text") text(part.text);
        else if (part.type === "file-data" || part.type === "image-data") {
          const block = claudeFileBlock(part.mediaType, part.data);
          if (block) content.push(block);
        } else if (part.type === "file") {
          const block = claudeFileBlock(part.mediaType, part.data);
          if (block) content.push(block);
        } else if (part.type === "image-url" || part.type === "file-url") {
          const block = claudeFileBlock(part.type === "image-url" ? "image" : (part.mediaType ?? "application/pdf"), part.url);
          if (block) content.push(block);
        }
        // a provider's file id or reference: nothing Claude can hold
      }
      return { content, isError: false };
    default:
      return { content, isError: false };
  }
}
