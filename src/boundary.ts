import type { ToolContent, ToolResult } from "./tool.ts";

/**
 * The TOOL BOUNDARY: the one seam every driver passes each tool call through, and where a host runs what
 * must happen around a call without the tool knowing.
 *
 * It is where a host:
 *   - REFUSES a call before it runs (a wind-down asking the agent to stop, a per-run tool-call cap): the
 *     refusal is the call's result, so nothing is killed halfway and the agent reads why;
 *   - RECORDS the call: a running row as it starts (a call that hangs is otherwise invisible for exactly as
 *     long as it hangs), completed in place by `callId` when it returns, so parallel calls cannot
 *     mis-attribute; a thrown handler is recorded too, or a failed call vanishes from history;
 *   - runs the handler inside its own context (an async-local carrying the call's abort signal and id);
 *   - APPENDS to the result what arrived while the agent was working (messages to it, pictures in them),
 *     which is how a running turn hears about them at the next tool call instead of after the turn ends.
 *
 * The ORDER is fixed here and is the point of the module: refuse → record start → run → record end →
 * append. Every hook is optional, and every recording hook is best-effort: recording must never break,
 * or delay the failure of, the call itself.
 */

export interface ToolCall {
  /** The tool's bare name, as its def declares it. */
  name: string;
  /** The name a host records the call under (by default, the bare name). */
  recordedName: string;
  args: unknown;
  /** A fresh id tying the call's start to its completion. Absent on a refused call (it never started). */
  callId?: string;
  /** Aborts when the caller gives up on the call (an interrupt, a client timeout). */
  signal?: AbortSignal;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface ToolBoundary<R extends ToolResult = any> {
  /** A refusal text to return INSTEAD of running the call, or null to let it run. */
  refuse?(call: ToolCall): string | null | Promise<string | null>;
  /** Record a refused call (it has no callId: it never started). */
  recordRefusal?(call: ToolCall, text: string): Promise<void> | void;
  /** Record the call's start. */
  recordStart?(call: ToolCall): Promise<void> | void;
  /** Record the call's completion (with the handler's result, before anything is appended to it). */
  recordEnd?(call: ToolCall, result: R): Promise<void> | void;
  /** Record a handler that THREW. The error is re-thrown to the driver afterwards. */
  recordThrow?(call: ToolCall, error: unknown): Promise<void> | void;
  /** Run the handler inside the host's per-call context. Default: just run it. */
  around?<T>(call: ToolCall, fn: () => Promise<T>): Promise<T>;
  /**
   * What to append to the result, after the tool's own content (e.g. messages that arrived), or null: a
   * text (one block), or blocks of text and images in order. Every driver delivers them after the tool's
   * own result, as it delivers an image a tool returns.
   */
  append?(call: ToolCall, result: R): Promise<Appended> | Appended;
}

/** What `append` may return. */
export type Appended = string | ToolContent[] | null;

/** A fresh id tying a call's start row to its completion. */
export const newCallId = (): string => crypto.randomUUID();

async function quietly(fn: (() => unknown) | undefined): Promise<void> {
  if (!fn) return;
  try {
    await fn();
  } catch {
    // best-effort: recording never breaks the call
  }
}

/**
 * Pass one call through the boundary. `invoke` runs the handler; the driver supplies it because only the
 * driver knows how its SDK hands the handler its arguments.
 */
export async function runAtBoundary<R extends ToolResult>(
  boundary: ToolBoundary<R> | undefined,
  call: Omit<ToolCall, "callId">,
  invoke: () => Promise<R>,
): Promise<R> {
  const b = boundary ?? {};
  const refusal = b.refuse ? await b.refuse(call) : null;
  if (refusal) {
    const res = { content: [{ type: "text" as const, text: refusal }], isError: true } as R;
    await quietly(() => b.recordRefusal?.(call, refusal));
    return res;
  }
  const started: ToolCall = { ...call, callId: newCallId() };
  await quietly(() => b.recordStart?.(started));
  let res: R;
  try {
    res = b.around ? await b.around(started, invoke) : await invoke();
  } catch (e) {
    await quietly(() => b.recordThrow?.(started, e));
    throw e;
  }
  await quietly(() => b.recordEnd?.(started, res));
  // After the call's own record, so a transcript reads call → message.
  let extra: Appended = null;
  try {
    extra = b.append ? await b.append(started, res) : null;
  } catch {
    return res; // a hiccup in the host's inbox never breaks the call; it can deliver later
  }
  const blocks: ToolContent[] = typeof extra === "string" ? (extra ? [{ type: "text", text: extra }] : []) : (extra ?? []);
  if (!blocks.length) return res;
  return { ...res, content: [...(res.content ?? []), ...blocks] };
}
