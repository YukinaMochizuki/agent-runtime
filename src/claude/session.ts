import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

/**
 * A Claude Code session's FILE, read and extended from outside the CLI: what a host needs to carry a
 * conversation into or out of the Claude driver (../convert/). Claude Code keeps each session as JSONL at
 * `<config dir>/projects/<cwd, sanitized>/<session id>.jsonl`, one entry per line. The format is the CLI's
 * own (the SDK calls it internal); what this module relies on was measured against the real CLI
 * (scripts/verify-convert-fake.ts checks it end to end):
 *
 *   - The CONVERSATION is a chain: each entry names its `parentUuid`, and the CLI follows it back from the
 *     latest message. A missing `uuid`, or a broken chain, silently cuts the history short; a missing
 *     `timestamp` makes the CLI refuse the session ("No conversation found"). `cwd`, `version`, `userType`
 *     and `isSidechain` are written as the CLI writes them; it does not depend on their values.
 *   - Only `user` and `assistant` entries are conversation. The rest are the CLI's (attachments, prompt
 *     snapshots, cost state, ...), and so is a user entry marked `isMeta` (the `[Image: source: …]` line).
 *   - An assistant reply is written one entry per content block, all with its `message.id`: merged back here.
 *   - A COMPACTION writes a boundary whose `parentUuid` is null, then the summary as a user message; the
 *     chain from the latest message therefore ends at the summary, which is what the model works from.
 *
 * POSITIONS: a place in a session is the uuid of a conversation entry. `readClaudeSession(…, { after })`
 * returns what came after it; when it is no longer on the chain (a compaction cut it away), everything the
 * chain still holds is returned. A position between a tool call and its result reads from the call, so a
 * result is never returned without it. Either way the read holds some of what came before `after`, and
 * `afterFound` is false to say so.
 *
 * Writing: messages are APPENDED after the latest user or assistant entry, chained to it, so the CLI
 * resumes onto them; their timestamps come after every one already in the file. An appended assistant
 * message is labelled with a model id no Claude model has (`CONVERTED_MODEL`): Claude Code treats a
 * recorded model id as a claim about who wrote it (it drops another model's thinking). Nothing here may run
 * while a turn is using the session.
 */

/** A content block of a Claude message (the Messages API's), as far as reading and converting needs. */
export type ClaudeBlock = { type: string; [k: string]: unknown };

/** One message of a session's conversation: a user or assistant turn of the Messages API. */
export interface ClaudeMessage {
  role: "user" | "assistant";
  content: string | ClaudeBlock[];
}

/** The model id appended assistant messages are recorded under: not a Claude model's. */
export const CONVERTED_MODEL = "converted";

export interface ClaudeSessionPlace {
  /** The login's config dir (CLAUDE_CONFIG_DIR). Default: the environment's, else ~/.claude. */
  configDir?: string | null;
}

export interface ReadClaudeSessionOptions extends ClaudeSessionPlace {
  /** Only what comes after this entry (a position returned by a read or an append). */
  after?: string | null;
}

export interface ClaudeSessionRead {
  /** The conversation (or the part after `after`), user and assistant, in order. */
  messages: ClaudeMessage[];
  /** The position of its latest message; null for a session with none. */
  end: string | null;
  /**
   * False when the read could not start exactly after `after`: it is no longer on the chain (compacted away:
   * `messages` is all the chain holds), or it fell between a tool call and its result (the read starts at the
   * call). `messages` then holds some of what came before `after`.
   */
  afterFound: boolean;
}

export interface AppendClaudeSessionOptions extends ClaudeSessionPlace {
  /** The working directory the CLI runs the session in: where a NEW session's file goes. Default: process.cwd(). */
  cwd?: string;
}

export function claudeConfigDir(configDir?: string | null): string {
  return configDir ?? process.env.CLAUDE_CONFIG_DIR ?? join(homedir(), ".claude");
}

/** The directory the CLI keeps a working directory's sessions in. */
export function claudeProjectDir(cwd: string, configDir?: string | null): string {
  const name = cwd.replace(/[^A-Za-z0-9]/g, "-");
  // Longer names are shortened with a hash by the CLI; that rule is not reproduced here.
  if (name.length > 200) throw new Error(`the working directory ${cwd} is too long to place a Claude session for`);
  return join(claudeConfigDir(configDir), "projects", name);
}

/** The file of session `id`, in whichever project directory holds it; null when there is none. */
export function findClaudeSession(id: string, configDir?: string | null): string | null {
  const projects = join(claudeConfigDir(configDir), "projects");
  if (!existsSync(projects)) return null;
  for (const dir of readdirSync(projects)) {
    const file = join(projects, dir, `${id}.jsonl`);
    if (existsSync(file)) return file;
  }
  return null;
}

interface Entry {
  type?: string;
  uuid?: string;
  parentUuid?: string | null;
  isSidechain?: boolean;
  isMeta?: boolean;
  timestamp?: string;
  message?: { id?: string; role?: string; content?: string | ClaudeBlock[] };
}

const isConversation = (e: Entry): boolean => (e.type === "user" || e.type === "assistant") && !!e.uuid && e.isSidechain !== true;

function entries(file: string): Entry[] {
  return readFileSync(file, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .flatMap((l) => {
      try {
        return [JSON.parse(l) as Entry];
      } catch {
        return []; // a line cut short by a crash: the CLI skips it too
      }
    });
}

/** The chain from the root (or the last compaction) to the latest conversation entry, in order. */
function chain(all: Entry[]): Entry[] {
  const byUuid = new Map<string, Entry>();
  for (const e of all) if (e.uuid) byUuid.set(e.uuid, e);
  const leaf = all.findLast(isConversation);
  const out: Entry[] = [];
  const seen = new Set<string>();
  for (let e: Entry | undefined = leaf; e?.uuid && !seen.has(e.uuid); e = e.parentUuid ? byUuid.get(e.parentUuid) : undefined) {
    seen.add(e.uuid);
    out.push(e);
  }
  return out.reverse();
}

/** Read session `id`'s conversation as Claude messages (assistant blocks merged back into one message). */
export function readClaudeSession(id: string, opts: ReadClaudeSessionOptions = {}): ClaudeSessionRead {
  const file = findClaudeSession(id, opts.configDir);
  if (!file) throw new Error(`no Claude session ${id} in ${claudeConfigDir(opts.configDir)}`);
  const conv = chain(entries(file)).filter((e) => isConversation(e) && !e.isMeta && e.message);
  let from = 0;
  let afterFound = true;
  if (opts.after) {
    const at = conv.findIndex((e) => e.uuid === opts.after);
    if (at >= 0) from = at + 1;
    else afterFound = false;
  }
  // Never start on a tool's result: back up to the assistant message (all its entries) that made the call.
  const answersCall = (e: Entry | undefined) => e?.type === "user" && Array.isArray(e.message?.content) && e.message.content.some((b) => b.type === "tool_result");
  while (from > 0 && from < conv.length && answersCall(conv[from])) {
    let start = from - 1;
    while (start > 0 && conv[start]!.type === "assistant" && conv[start - 1]!.type === "assistant" && conv[start - 1]!.message?.id === conv[start]!.message?.id) start--;
    if (conv[start]!.type !== "assistant") break;
    from = start;
    afterFound = false;
  }
  const messages: ClaudeMessage[] = [];
  let lastId: string | undefined;
  for (const e of conv.slice(from)) {
    const role = e.type as "user" | "assistant";
    const content = e.message!.content ?? [];
    const prev = messages.at(-1);
    if (role === "assistant" && prev?.role === "assistant" && e.message!.id && e.message!.id === lastId && Array.isArray(prev.content)) {
      prev.content.push(...(typeof content === "string" ? [{ type: "text", text: content }] : content));
    } else {
      messages.push({ role, content: typeof content === "string" ? content : [...content] });
    }
    lastId = role === "assistant" ? e.message!.id : undefined;
  }
  return { messages, end: conv.at(-1)?.uuid ?? null, afterFound };
}

/**
 * Append `messages` to session `id` (null: a new session), chained after its latest entry. Returns the
 * session id and the position of the last message written. Throws when `id` names no session.
 */
export function appendClaudeSession(id: string | null, messages: ClaudeMessage[], opts: AppendClaudeSessionOptions = {}): { sessionId: string; end: string | null } {
  const cwd = opts.cwd ?? process.cwd();
  let sessionId = id;
  let file: string;
  let parent: string | null = null;
  if (sessionId) {
    const found = findClaudeSession(sessionId, opts.configDir);
    if (!found) throw new Error(`no Claude session ${sessionId} in ${claudeConfigDir(opts.configDir)}`);
    file = found;
    parent = chain(entries(file)).at(-1)?.uuid ?? null;
  } else {
    sessionId = crypto.randomUUID();
    const dir = claudeProjectDir(cwd, opts.configDir);
    mkdirSync(dir, { recursive: true });
    file = join(dir, `${sessionId}.jsonl`);
  }
  if (!messages.length) return { sessionId, end: parent };
  const latest = existsSync(file) ? entries(file).reduce((max, e) => Math.max(max, e.timestamp ? Date.parse(e.timestamp) || 0 : 0), 0) : 0;
  const t0 = Math.max(Date.now(), latest + 1);
  const lines = messages.map((m, i) => {
    const uuid = crypto.randomUUID();
    const entry = {
      parentUuid: parent,
      isSidechain: false,
      userType: "external",
      cwd,
      sessionId,
      type: m.role,
      uuid,
      // Increasing, and after every entry already there.
      timestamp: new Date(t0 + i).toISOString(),
      message:
        m.role === "assistant"
          ? {
              id: `msg_${uuid.replace(/-/g, "")}`,
              type: "message",
              role: "assistant",
              model: CONVERTED_MODEL,
              content: typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content,
              stop_reason: Array.isArray(m.content) && m.content.some((b) => b.type === "tool_use") ? "tool_use" : "end_turn",
              stop_sequence: null,
              usage: { input_tokens: 0, output_tokens: 0 },
            }
          : { role: "user", content: m.content },
    };
    parent = uuid;
    return JSON.stringify(entry);
  });
  const existing = existsSync(file) ? readFileSync(file, "utf8") : "";
  appendFileSync(file, (existing && !existing.endsWith("\n") ? "\n" : "") + lines.join("\n") + "\n");
  return { sessionId, end: parent };
}

/**
 * Copy session `id` into a new session beside it, as it is: the same entries (uuids, signed thinking, the
 * CLI's own entries), each with `sessionId` rewritten to the new id; a line that does not parse (cut short)
 * is copied as it is. The CLI resumes the copy exactly as it would the original, and the two then go their
 * own ways (verified against the real CLI 2.1.322: the same request, byte for byte, from either). Returns
 * the new id and its position, which is the original's. Throws when `id` names no session. Nothing may run on
 * the session while it is copied.
 */
export function forkClaudeSession(id: string, opts: ClaudeSessionPlace = {}): { sessionId: string; end: string | null } {
  const file = findClaudeSession(id, opts.configDir);
  if (!file) throw new Error(`no Claude session ${id} in ${claudeConfigDir(opts.configDir)}`);
  const sessionId = crypto.randomUUID();
  const text = readFileSync(file, "utf8");
  const copied = text
    .split("\n")
    .map((line) => {
      if (!line.trim()) return line;
      try {
        const entry = JSON.parse(line) as Record<string, unknown>;
        if (entry.sessionId !== id) return line;
        entry.sessionId = sessionId;
        return JSON.stringify(entry);
      } catch {
        return line;
      }
    })
    .join("\n");
  writeFileSync(join(dirname(file), `${sessionId}.jsonl`), copied, { flag: "wx" });
  return { sessionId, end: chain(entries(file)).filter((e) => isConversation(e) && !e.isMeta && e.message).at(-1)?.uuid ?? null };
}
