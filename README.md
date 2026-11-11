# agent-runtime

The agent runtime shared by [workstream](https://github.com/YukinaMochizuki/workstream) and other hosts:
the drivers that run an agent's turn, the tool layer they all consume, and the tool boundary every tool
call passes through. A host brings its own tools, its own store and its own policy; this package owns the
parts that are the same for every host and that took real use to get right.

No store, no UI, no product concepts live here. Where a driver needs one (a conversation's history, a
transcript row, a message to deliver mid-turn), it asks the host through a small interface.

## What is in it

```
agent-runtime            the contract and the tool layer
  adapter.ts             BackendAdapter: one TURN per call (run / continueRun), streamed through callbacks
  tool.ts                ToolDef / ToolBundle / ToolResult: what a tool IS, independent of any SDK
  boundary.ts            the tool boundary: what runs around every call, in a fixed order
  effort.ts              the neutral reasoning-effort vocabulary
agent-runtime/claude     the Claude Agent SDK driver
agent-runtime/openai-compat
                         any OpenAI-compatible endpoint, on an agent loop written here
agent-runtime/tools/web  web_search / web_scrape / web_fetch, configured by the host
```

### Shared tools (`agent-runtime/tools/*`)

Most tools belong to a host, because they act on its own concepts (workstream's boxes, a chat's
messages). A tool lives here instead when it needs no host store or state and more than one host wants
it: today that is outbound web access. Such a module reads no environment; the host passes in its
configuration (API keys, endpoints) and decides which runs get the tools.

### The tool boundary

Every driver passes every tool call through one `ToolBoundary`, so a host's policy behaves the same
whichever model runs. The order is fixed:

```
refuse? ──yes──▶ the refusal IS the result (recorded; the handler never runs)
   │no
record start ─▶ run the handler (inside the host's per-call context) ─▶ record end ─▶ append
                                       │throws
                                       └──▶ record the throw, re-throw
```

`append` is how a running turn hears about something that arrived while it worked: the host returns
text (e.g. its inbox) or content blocks (text and images, in order), and they reach the model after the
tool's own result, at the next tool call rather than after the turn ends. Recording is best-effort and
never breaks the call.

### Images in a turn

A turn's `prompt` is a string or a list of parts, `{ type: "text", text }` and
`{ type: "image", data, mimeType }` (base64): the same shapes a tool returns (`ToolContent`). A string
works exactly as before. Where an image can reach the model:

| From | Claude | OpenAI-compatible (`vision: true`) |
|---|---|---|
| the prompt | content blocks of the opening user message, on `run` and on a resume | content parts of the opening user message |
| a tool's result, or `append` | image blocks inside the `tool_result`, after the tool's text | named in the tool message's text, sent in a user message right after the tool results |

On Claude a resume keeps earlier images in the session's history (Claude Code stores them). On an
OpenAI-compatible model without `vision`, each image is replaced by a short note saying it was not sent,
and compaction turns older images into notes like any other image.

### The Claude driver (`agent-runtime/claude`)

The Claude Agent SDK, with the lessons that cost something to learn baked into its options:
built-in tools hidden by default (`tools: []`; a host can name some), no settings files read, the
claude.ai account's connectors switched off, thinking summaries requested so newer models' progress
updates between tool calls reach the host, the session id surfaced from the first message (so an
interrupted turn is resumable), context usage taken from the last request rather than the turn's sum,
and Claude Code's non-streaming fallback replayed without duplicating what already streamed. Which login
a turn runs on is a `configDir` (`CLAUDE_CONFIG_DIR`); choosing among several is the host's policy.
Claude Code records a conversation's system prompt and replays it on every resume, ignoring a new one; a
host whose prompt changes mid-conversation sets `liveSystemPrompt`, and each turn's prompt is then sent
fresh (an edit costs the prompt cache once).

### The OpenAI-compatible driver (`agent-runtime/openai-compat`)

The AI SDK (`ai`, `@ai-sdk/openai-compatible`) makes ONE model call per step (`streamText`, tools without
`execute`); the loop around it is this package's: each tool call goes through the same boundary as on
Claude, a stop is honoured between steps, and the conversation is the host's (`ConversationStore`),
appended as it is produced so a crash loses at most the step in flight. The system prompt is sent every
turn, never stored. Reasoning effort is sent as `reasoning_effort` only at the levels an endpoint has a
name for (low / medium / high).

**Context and the prompt cache.** A provider caches a request's prefix, so the loop only ever appends to
history: the messages of every request start with those of the previous one, unchanged, across turns and
reloads too. Compaction (`compaction` option, with a host store that implements `compact`) waits until the
context crosses a threshold (measured by the endpoint where it reports usage, carried across turns through
`contextTokens`), then compacts once: older tool outputs and images become short notes; if that is not
enough, the older part becomes one summary written by the model (the request being worked on is kept word
for word); if even that is not enough, the tool output of every step but the latest is shortened. It cuts
only before a step, so a tool call is never separated from its result, and a summary that fails falls back
to pruning instead of failing the turn. For providers that cache only up to a marker (Anthropic and Gemini
models through OpenRouter, ...), `cacheBreakpoints` marks `cache_control` on the system prompt, the summary
and the latest message (a rolling marker: it moves to the newest message on every request, as Claude Code's
does, so each request reads the one before it from cache).

**What a model can do** (`probe.ts`). `listOpenAICompatModels` returns, per model, what `GET /models`
reported (context window, input modalities, tools, reasoning), leaving whatever an endpoint does not say
undefined rather than false. `probeModel` sends one small streamed request offering a single `ping` tool
and reports whether a structured tool call came back (`ok`), the model answered in text instead
(`text-instead`: a served model whose tool-call parser is off), or the request failed (an error kind a
human acts on: auth, not-found, tools-unsupported, timeout, unreachable, ...), with its latency. Optionally
it also sends a small picture and reports only whether the endpoint accepted it. It never retries and
waits minutes by default, for endpoints that scale to zero.

## Using it

It is consumed as TypeScript source (both hosts run Bun); there is no build step. Pin a commit:

```jsonc
// package.json
"dependencies": { "agent-runtime": "github:YukinaMochizuki/agent-runtime#<sha>" }
```

`zod` and `@anthropic-ai/claude-agent-sdk` are peer dependencies: the host's copies are used, so a tool's
schema and the SDK's are the same instances.

## Developing it

```bash
bun install
bun run typecheck
bun test
bun scripts/verify-claude-fake.ts   # the Claude driver through the real CLI, against a fake API
```

The unit tests drive the OpenAI-compatible driver against both a mock model and a fake HTTP endpoint;
the script is the Claude driver's equivalent, a real Claude Code subprocess with no login or quota (it
also checks that images reach the API from a prompt, an append, and a resume).

A change here is a change to every host. Before pinning a new commit in a host, run that host's own
typecheck and tests against it (in workstream: point `agent-runtime` at a local checkout with
`file:<path>`, `bun install`, then `bun run typecheck && bun test`).
