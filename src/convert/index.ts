import type { ModelMessage } from "ai";
import { appendClaudeSession, readClaudeSession, type AppendClaudeSessionOptions, type ReadClaudeSessionOptions } from "../claude/session.ts";
import { claudeToModelMessages, modelMessagesToClaude, type ConvertOptions } from "./messages.ts";

/**
 * `agent-runtime/convert`: carry a conversation between the Claude driver (a Claude Code session) and the
 * OpenAI-compatible driver (`ModelMessage[]`, kept by the host's `ConversationStore`). Which conversation
 * is whose, and when to carry what, is the host's: this module only reads and writes the two forms.
 *
 *   claudeSessionMessages(id, { after })   a session (or its part after a position) as ModelMessage[]
 *   appendToClaudeSession(id, messages)    ModelMessage[] written onto a session (null: a new one)
 *   forkClaudeSession(id)                  a copy of a session, as it is, under a new id (one login)
 *
 * What carries and what does not: ./messages.ts. How a session file is read and extended: ../claude/session.ts.
 * A carried part is the messages themselves; nothing is added to say they were carried.
 */

export function claudeSessionMessages(id: string, opts: ReadClaudeSessionOptions & ConvertOptions = {}): { messages: ModelMessage[]; end: string | null; afterFound: boolean } {
  const read = readClaudeSession(id, opts);
  return { messages: claudeToModelMessages(read.messages, opts), end: read.end, afterFound: read.afterFound };
}

export function appendToClaudeSession(id: string | null, messages: ModelMessage[], opts: AppendClaudeSessionOptions & ConvertOptions = {}): { sessionId: string; end: string | null } {
  return appendClaudeSession(id, modelMessagesToClaude(messages, opts), opts);
}

export {
  appendClaudeSession,
  claudeConfigDir,
  claudeProjectDir,
  CONVERTED_MODEL,
  findClaudeSession,
  forkClaudeSession,
  readClaudeSession,
  type AppendClaudeSessionOptions,
  type ClaudeBlock,
  type ClaudeMessage,
  type ClaudeSessionPlace,
  type ClaudeSessionRead,
  type ReadClaudeSessionOptions,
} from "../claude/session.ts";
export {
  claudeToModelMessages,
  claudeToolUseId,
  defaultClaudeToolName,
  defaultWireToolName,
  modelMessagesToClaude,
  type ConvertOptions,
} from "./messages.ts";
