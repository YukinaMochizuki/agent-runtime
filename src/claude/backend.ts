import type { AdapterContinueInput, AdapterRunInput, AdapterRunOptions, AdapterRunResult, BackendAdapter } from "../adapter.ts";
import type { ToolBoundary } from "../boundary.ts";
import { adaptTools } from "./mcp.ts";
import { runClaudeAgent, type RateLimitReading } from "./sdk.ts";

export interface ClaudeBackendOptions {
  /** The login turns run on (absent: the process's own environment). */
  configDir?: string | null;
  /** The model when a run names none (absent: the SDK's default). */
  defaultModel?: string;
  /** What every tool call passes through. */
  boundary?: ToolBoundary;
  /** The name a call is recorded under. Default: the bare tool name. */
  recordAs?: (serverName: string, toolName: string) => string;
  /** Claude Code built-in tools to expose (default none; see `RunClaudeAgentInput.builtinTools`). */
  builtinTools?: string[];
  /** Each `rate_limit_event` of a turn. */
  onRateLimit?: (reading: RateLimitReading) => void;
}

/**
 * The Claude Agent SDK as a BackendAdapter. Tools are in-process SDK MCP servers; there is no subprocess
 * of the host's to start. The conversation id IS the SDK's session_id.
 *
 * Built-in tools are hidden (`tools: []`) unless the host names some, so by default the only callable
 * surface is the host's own tools: that is the safety boundary.
 */
export class ClaudeBackend implements BackendAdapter {
  readonly id: string = "claude-agent-sdk";

  constructor(private readonly opts: ClaudeBackendOptions = {}) {}

  run(input: AdapterRunInput, opts: AdapterRunOptions): Promise<AdapterRunResult> {
    return this.turn(input, opts, undefined);
  }

  continueRun(input: AdapterContinueInput, opts: AdapterRunOptions): Promise<AdapterRunResult> {
    return this.turn(input, opts, input.backendSessionId); // continue in place — no fork
  }

  private async turn(input: AdapterRunInput, opts: AdapterRunOptions, resumeSessionId: string | undefined): Promise<AdapterRunResult> {
    const tools = adaptTools(input.tools, { boundary: this.opts.boundary, recordAs: this.opts.recordAs });
    const builtin = this.opts.builtinTools ?? [];
    const r = await runClaudeAgent({
      configDir: this.opts.configDir ?? null,
      prompt: input.prompt,
      // The system prompt is the conversation's: a resume keeps the one it started with.
      ...(input.systemPrompt && !resumeSessionId ? { systemPrompt: input.systemPrompt } : {}),
      // The model is forwarded on a resume too: dropping it there moved a resumed conversation onto the
      // default model from turn two.
      model: input.model ?? this.opts.defaultModel,
      effort: input.effort,
      ...(resumeSessionId ? { resumeSessionId } : {}),
      mcpServers: tools.mcpServers,
      allowedTools: [...tools.allowedTools, ...builtin],
      ...(builtin.length ? { builtinTools: builtin } : {}),
      onToken: opts.onToken,
      onTool: opts.onTool,
      onToolDraft: opts.onToolDraft,
      onSessionId: opts.onSessionId,
      onRateLimit: this.opts.onRateLimit,
      signal: opts.signal,
    });
    return { text: r.text, backendSessionId: r.sessionId, usage: r.usage };
  }
}
