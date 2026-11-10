import { createSdkMcpServer, tool, type SdkMcpToolDefinition } from "@anthropic-ai/claude-agent-sdk";
import { runAtBoundary, type ToolBoundary } from "../boundary.ts";
import type { ToolBundle, ToolDef, ToolResult } from "../tool.ts";

/**
 * Tool bundles as the Claude Agent SDK takes them: each bundle an in-process SDK MCP server, each call
 * passed through the host's tool boundary (boundary.ts). The allowed-tool identifier is
 * `mcp__<serverName>__<toolName>`.
 */
export interface ClaudeTools {
  mcpServers: Record<string, ReturnType<typeof createSdkMcpServer>>;
  allowedTools: string[];
}

export interface AdaptToolsOptions {
  /** What every call passes through. Absent: the handler runs bare. */
  boundary?: ToolBoundary;
  /** The name a call is recorded under. Default: the bare tool name. */
  recordAs?: (serverName: string, toolName: string) => string;
}

/** The SDK types `extra` as unknown; at runtime it is the MCP RequestHandlerExtra, whose signal aborts on an interrupt. */
function signalOf(extra: unknown): AbortSignal | undefined {
  const signal = (extra as { signal?: unknown } | undefined)?.signal;
  return signal instanceof AbortSignal ? signal : undefined;
}

type SdkResult = { content: ToolResult["content"]; isError?: boolean; [k: string]: unknown };

/**
 * Wrap already-built SDK tool definitions so each call passes through a boundary. For a host that builds
 * some SDK tools itself (rather than from ToolDefs).
 */
export function wrapSdkTools(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tools: SdkMcpToolDefinition<any>[],
  boundary: ToolBoundary | undefined,
  recordAs: (name: string) => string = (name) => name,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): SdkMcpToolDefinition<any>[] {
  return tools.map((t) =>
    tool(t.name, t.description, t.inputSchema, async (args, extra) =>
      (await runAtBoundary(boundary, { name: t.name, recordedName: recordAs(t.name), args, signal: signalOf(extra) }, async () => (await t.handler(args, extra)) as unknown as ToolResult)) as unknown as SdkResult,
    ),
  );
}

function toSdkTool(def: ToolDef): SdkMcpToolDefinition<typeof def.inputSchema> {
  return tool(def.name, def.description, def.inputSchema, async (args) => {
    // ToolResult is the SDK's expected `{ content: [...], isError? }`; the SDK's declared return type also
    // carries an index signature not modelled here.
    return (await def.handler(args)) as unknown as SdkResult;
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function adaptTools(bundles: ToolBundle<any>[], opts: AdaptToolsOptions = {}): ClaudeTools {
  const mcpServers: ClaudeTools["mcpServers"] = {};
  const allowedTools: string[] = [];
  for (const bundle of bundles) {
    const sdkTools = (bundle.tools as ToolDef[]).map(toSdkTool);
    const recordAs = opts.recordAs ? (name: string) => opts.recordAs!(bundle.serverName, name) : undefined;
    mcpServers[bundle.serverName] = createSdkMcpServer({ name: bundle.serverName, tools: wrapSdkTools(sdkTools, opts.boundary, recordAs) });
    for (const t of bundle.tools as ToolDef[]) allowedTools.push(`mcp__${bundle.serverName}__${t.name}`);
  }
  return { mcpServers, allowedTools };
}
