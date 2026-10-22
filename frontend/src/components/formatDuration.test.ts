import { describe, expect, it } from "vitest";
import { formatDuration } from "./formatDuration";

describe("formatDuration", () => {
  it("shows either signed zero as ordinary zero seconds", () => {
    expect(formatDuration(0)).toBe("0 s");
    expect(formatDuration(-0)).toBe("0 s");
  });
});
