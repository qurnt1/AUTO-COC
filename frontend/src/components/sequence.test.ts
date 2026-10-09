import { describe, expect, it } from "vitest";
import {
  getSequencePageBounds,
  getTimelineMarkers,
  MAX_TIMELINE_MARKERS,
  nextSequencePageStart,
  previousSequencePageStart,
} from "./sequence";

describe("sequence pagination", () => {
  it("handles empty, first-page, and next-page boundaries", () => {
    expect(getSequencePageBounds(0, 0)).toEqual({ start: 0, end: 0 });
    expect(getSequencePageBounds(1, 0)).toEqual({ start: 0, end: 1 });
    expect(getSequencePageBounds(8, 0)).toEqual({ start: 0, end: 8 });
    expect(nextSequencePageStart(8, 0)).toBe(0);

    expect(getSequencePageBounds(9, 0)).toEqual({ start: 0, end: 8 });
    const nineNext = nextSequencePageStart(9, 0);
    expect(getSequencePageBounds(9, nineNext)).toEqual({ start: 8, end: 9 });
    expect(previousSequencePageStart(nineNext)).toBe(0);
    expect(nextSequencePageStart(9, nineNext)).toBe(nineNext);

    expect(getSequencePageBounds(100, 0)).toEqual({ start: 0, end: 8 });
    const hundredNext = nextSequencePageStart(100, 0);
    expect(getSequencePageBounds(100, hundredNext)).toEqual({ start: 8, end: 100 });
    expect(previousSequencePageStart(hundredNext)).toBe(0);
    expect(nextSequencePageStart(100, hundredNext)).toBe(hundredNext);
  });

  it("keeps pages bounded to the first 8 events and then 100", () => {
    const total = 250_000;
    const first = getSequencePageBounds(total, 0);
    expect(first).toEqual({ start: 0, end: 8 });

    const secondStart = nextSequencePageStart(total, first.start);
    expect(getSequencePageBounds(total, secondStart)).toEqual({ start: 8, end: 108 });
    expect(previousSequencePageStart(secondStart)).toBe(0);

    const lastStart = 249_908;
    expect(getSequencePageBounds(total, lastStart)).toEqual({ start: lastStart, end: total });
    expect(previousSequencePageStart(lastStart)).toBe(249_808);
    expect(nextSequencePageStart(total, lastStart)).toBe(lastStart);
  });
});

describe("timeline markers", () => {
  it("caps markers at 160 for the maximum supported macro and retains endpoints", () => {
    const step = { t: 0.01, type: "mouse_move", data: null };
    const steps = Array.from({ length: 250_000 }, () => step);

    const markers = getTimelineMarkers(steps);

    expect(markers).toHaveLength(MAX_TIMELINE_MARKERS);
    expect(markers[0].index).toBe(0);
    expect(markers.at(-1)?.index).toBe(steps.length - 1);
    expect(markers.every((marker, index) => index === 0 || Number.parseFloat(marker.left) >= Number.parseFloat(markers[index - 1].left))).toBe(true);
  });

  it("samples by elapsed time instead of spacing events uniformly", () => {
    const steps = [
      { t: 90, type: "nop", data: null },
      ...Array.from({ length: 200 }, () => ({ t: 0.05, type: "mouse_move", data: null })),
    ];

    const markers = getTimelineMarkers(steps, 4);

    expect(markers.length).toBeLessThanOrEqual(4);
    expect(markers[1].index).toBe(1);
    expect(Number.parseFloat(markers[1].left)).toBeGreaterThan(80);
    expect(markers.at(-1)?.index).toBe(steps.length - 1);
  });
});
