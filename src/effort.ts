/**
 * How much reasoning to spend on a turn, in the Claude SDK's five names (it takes them directly on
 * `Options.effort`). Other drivers translate into their own terms, and a driver that cannot honour a level
 * says so by ignoring it, never by silently running at a different one. Ordered low → high, which is what
 * a picker renders and what a range label reads off.
 */
export const EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"] as const;
export type EffortLevel = (typeof EFFORT_LEVELS)[number];
