import type { ZodRawShape } from "zod";

/**
 * The backend-neutral tool layer: what a tool IS (name, description, args schema, the handler that does
 * the work), without binding to any agent SDK. Every driver in this package consumes the same defs.
 * Handlers must not import an agent-SDK module.
 *
 * A host extends `ToolDef` with what only it knows (a permission level, the data an approval card shows,
 * ...) by interface extension, and runs that policy at the tool boundary (boundary.ts). The runtime
 * never reads those fields, which is what keeps a host's policy out of the runtime.
 */

/**
 * What a tool may hand back, as a NARROW union rather than MCP's full `CallToolResult["content"]`.
 *
 * The drivers would accept the wider MCP union, and it would even work. It is narrowed on purpose: the
 * wider union lets any tool put audio or an embedded resource into a result, down code paths nothing
 * tests and nothing renders. A type naming exactly what is supported is the difference between a tool
 * that fails to compile and one that quietly does something nobody designed.
 *
 * `image.data` is base64, `mimeType` its media type. The field names are MCP's, so this stays
 * structurally assignable to what the Claude SDK expects.
 */
export type ToolContent = { type: "text"; text: string } | { type: "image"; data: string; mimeType: string };

/** A tool's result. A host may add fields of its own (never sent to the model) by extending it. */
export interface ToolResult {
  content: ToolContent[];
  isError?: boolean;
}

/**
 * The TEXT of a result. Non-text blocks are dropped: an image goes to the model for the turn it was
 * asked for, and a transcript keeps the text. Persisting base64 in a transcript is worse than useless.
 */
export function textOf(res: { content?: Array<{ type?: string; text?: string }> }): string {
  return (res.content ?? [])
    .filter((c) => c.type === "text" && typeof c.text === "string")
    .map((c) => c.text)
    .join("\n");
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface ToolDef<TArgs = any, R extends ToolResult = ToolResult> {
  name: string;
  description: string;
  /** Zod shape (a record of zod schemas), mirroring the Claude SDK's tool() input. */
  inputSchema: ZodRawShape;
  handler: (args: TArgs) => Promise<R>;
}

/**
 * A group of tools published under one server name. On Claude the name is the prefix of the
 * `mcp__<server>__<tool>` allowed-tool identifier; other drivers use it the same way.
 */
export interface ToolBundle<T extends ToolDef = ToolDef> {
  serverName: string;
  tools: T[];
}

/** Standard helpers so handlers don't repeat the boilerplate. */
export const ok = (text: string): ToolResult => ({ content: [{ type: "text", text }] });
export const err = (text: string): ToolResult => ({ content: [{ type: "text", text }], isError: true });
