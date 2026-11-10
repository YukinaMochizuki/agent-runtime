import { describe, expect, test } from "bun:test";
import { runAtBoundary, type ToolBoundary, type ToolCall } from "../src/boundary.ts";
import { ok, type ToolResult } from "../src/tool.ts";

const call = { name: "bash", recordedName: "bash", args: { cmd: "ls" } };

/** A boundary that logs every hook, in the order it ran. */
function logging(over: Partial<ToolBoundary> = {}): { log: string[]; b: ToolBoundary } {
  const log: string[] = [];
  const b: ToolBoundary = {
    refuse: () => null,
    recordRefusal: (_c, t) => void log.push(`refused:${t}`),
    recordStart: (c) => void log.push(`start:${c.callId ? "id" : "no-id"}`),
    recordEnd: (c, r: ToolResult) => void log.push(`end:${c.callId ? "id" : "no-id"}:${r.content.length}`),
    recordThrow: () => void log.push("throw"),
    around: async (_c, fn) => {
      log.push("around");
      return fn();
    },
    append: () => {
      log.push("append");
      return null;
    },
    ...over,
  };
  return { log, b };
}

describe("runAtBoundary", () => {
  test("runs the hooks in order: start → around → end → append, with one callId", async () => {
    const ids = new Set<string>();
    const { log, b } = logging({
      recordStart: (c) => void (log.push("start"), ids.add(c.callId!)),
      recordEnd: (c) => void (log.push("end"), ids.add(c.callId!)),
    });
    const res = await runAtBoundary(b, call, async () => {
      log.push("handler");
      return ok("done");
    });
    expect(log).toEqual(["start", "around", "handler", "end", "append"]);
    expect(ids.size).toBe(1);
    expect(res).toEqual(ok("done"));
  });

  test("a refusal is the result, is recorded, and the handler never runs", async () => {
    const { log, b } = logging({ refuse: () => "winding down" });
    let ran = false;
    const res = await runAtBoundary(b, call, async () => {
      ran = true;
      return ok("x");
    });
    expect(ran).toBe(false);
    expect(res).toEqual({ content: [{ type: "text", text: "winding down" }], isError: true });
    expect(log).toEqual(["refused:winding down"]);
  });

  test("an appended block follows the result as its own text block; the record sees the result without it", async () => {
    let recorded: ToolResult | null = null;
    const { b } = logging({ append: () => "[inbox] a message", recordEnd: (_c, r: ToolResult) => void (recorded = r) });
    const res = await runAtBoundary(b, call, async () => ok("out"));
    expect(res.content).toEqual([
      { type: "text", text: "out" },
      { type: "text", text: "[inbox] a message" },
    ]);
    expect(recorded!.content).toEqual([{ type: "text", text: "out" }]);
  });

  test("a throwing handler is recorded and re-thrown; nothing is appended", async () => {
    const { log, b } = logging();
    await expect(
      runAtBoundary(b, call, async () => {
        throw new Error("boom");
      }),
    ).rejects.toThrow("boom");
    expect(log).toEqual(["start:id", "around", "throw"]);
  });

  test("a failing record or append never breaks the call", async () => {
    const boom = () => {
      throw new Error("store down");
    };
    const res = await runAtBoundary({ recordStart: boom, recordEnd: boom, append: boom }, call, async () => ok("fine"));
    expect(res).toEqual(ok("fine"));
  });

  test("passes the call's signal and recorded name through", async () => {
    const ac = new AbortController();
    let seen: ToolCall | null = null;
    await runAtBoundary({ recordStart: (c) => void (seen = c) }, { ...call, recordedName: "local__bash", signal: ac.signal }, async () => ok(""));
    expect(seen!.signal).toBe(ac.signal);
    expect(seen!.recordedName).toBe("local__bash");
  });

  test("no boundary: the handler runs bare", async () => {
    expect(await runAtBoundary(undefined, call, async () => ok("bare"))).toEqual(ok("bare"));
  });
});
