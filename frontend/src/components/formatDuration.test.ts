import { describe, expect, it } from "vitest";
import { formatCountdown, formatDuration, formatElapsedDuration } from "./formatDuration";

describe("formatDuration", () => {
  it("shows either signed zero as ordinary zero seconds", () => {
    expect(formatDuration(0)).toBe("0 s");
    expect(formatDuration(-0)).toBe("0 s");
  });
});

describe("recording and playback clocks", () => {
  it("shows elapsed time as whole seconds without changing macro duration precision", () => {
    expect(formatElapsedDuration(10.1)).toBe("10 s");
    expect(formatElapsedDuration(61.9)).toBe("1 min 01 s");
    expect(formatDuration(10.1)).toBe("10,1 s");
  });

  it("rounds countdown values up to the next whole second", () => {
    expect(formatCountdown(3)).toBe(3);
    expect(formatCountdown(2.0149864654)).toBe(3);
    expect(formatCountdown(0.95954646)).toBe(1);
    expect(formatCountdown(0)).toBe(0);
  });
});
