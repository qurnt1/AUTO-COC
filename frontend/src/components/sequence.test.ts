import { describe, expect, it } from "vitest";
import { getPlaybackPosition, getTimelineMarkers, getTraceNodeClass, MAX_TIMELINE_MARKERS } from "./sequence";

describe("timeline markers", () => {
  it("represents each contiguous movement run by its final step and keeps every other action", () => {
    const steps = [
      { t: 0.1, type: "mouse_move", data: { x: 1 } },
      { t: 0.1, type: "mouse_move", data: { x: 2 } },
      { t: 0.2, type: "mouse_click", data: null },
      { t: 0.1, type: "mouse_move", data: { x: 3 } },
      { t: 0.1, type: "mouse_move", data: { x: 4 } },
      { t: 0.2, type: "key_down", data: null },
      { t: 0.1, type: "key_up", data: null },
      { t: 0.1, type: "mouse_move", data: { x: 5 } },
    ];

    const markers = getTimelineMarkers(steps);

    expect(markers.map(({ index }) => index)).toEqual([1, 2, 4, 5, 6, 7]);
    expect(markers.map(({ left }) => left)).toEqual(["20%", "40%", "60%", "80%", "90%", "99%"]);
    expect(markers[0].step.data).toEqual({ x: 2 });
    expect(markers.map(({ step }) => step.type)).toEqual([
      "mouse_move",
      "mouse_click",
      "mouse_move",
      "key_down",
      "key_up",
      "mouse_move",
    ]);
  });

  it("caps long traces while retaining their first and last actions", () => {
    const steps = Array.from({ length: 250_000 }, (_, index) => ({
      t: 0.01,
      type: index % 2 === 0 ? "mouse_click" : "key_down",
      data: null,
    }));
    const markers = getTimelineMarkers(steps);

    expect(markers).toHaveLength(MAX_TIMELINE_MARKERS);
    expect(markers[0].index).toBe(0);
    expect(markers.at(-1)?.index).toBe(steps.length - 1);
    expect(markers.some(({ step }) => step.type === "mouse_click")).toBe(true);
    expect(markers.some(({ step }) => step.type === "key_down")).toBe(true);
    expect(markers.every((marker, index) => index === 0 || Number.parseFloat(marker.left) >= Number.parseFloat(markers[index - 1].left))).toBe(true);
  });

  it("places markers at the time of each event, after its recorded delay", () => {
    const markers = getTimelineMarkers([
      { t: 0.1, type: "mouse_move", data: null },
      { t: 0.2, type: "mouse_move", data: null },
      { t: 0.7, type: "key_down", data: null },
    ]);

    expect(markers.map(({ index }) => index)).toEqual([1, 2]);
    expect(markers.map(({ left }) => left)).toEqual(["30%", "99%"]);
  });

  it("samples long traces by elapsed time", () => {
    const steps = [
      { t: 0.1, type: "nop", data: null },
      { t: 0.4, type: "key_down", data: null },
      { t: 90, type: "nop", data: null },
      ...Array.from({ length: 200 }, () => ({ t: 0.05, type: "mouse_move", data: null })),
    ];

    const markers = getTimelineMarkers(steps, 4);

    expect(markers.length).toBeLessThanOrEqual(4);
    expect(markers.map(({ index }) => index)).toEqual([0, 2, steps.length - 1]);
    expect(Number.parseFloat(markers[1].left)).toBeGreaterThan(80);
    expect(markers.at(-1)?.left).toBe("99%");
  });

  it("assigns distinct marker classes to mouse clicks and keyboard actions", () => {
    expect(getTraceNodeClass("mouse_click")).toBe("is-click");
    expect(getTraceNodeClass("key_down")).toBe("is-key");
    expect(getTraceNodeClass("key_up")).toBe("is-key");
    expect(getTraceNodeClass("mouse_move")).toBe("is-move");
    expect(getTraceNodeClass("scroll")).toBe("is-other");
  });

  it("returns no markers for an empty sequence", () => {
    expect(getTimelineMarkers([])).toEqual([]);
  });
});

describe("playback position", () => {
  it("maps elapsed time to a bounded position in a single playback", () => {
    expect(getPlaybackPosition(4.25, 10, false)).toEqual({ elapsedSeconds: 4.25, percent: 42.5, cycle: 0 });
    expect(getPlaybackPosition(12, 10, false)).toEqual({ elapsedSeconds: 10, percent: 100, cycle: 0 });
  });

  it("wraps progress and identifies the current loop cycle", () => {
    expect(getPlaybackPosition(12.5, 10, true)).toEqual({ elapsedSeconds: 2.5, percent: 25, cycle: 1 });
  });

  it("returns a safe zero position for invalid or empty durations", () => {
    expect(getPlaybackPosition(4, 0, true)).toEqual({ elapsedSeconds: 0, percent: 0, cycle: 0 });
    expect(getPlaybackPosition(Number.NaN, 10, false)).toEqual({ elapsedSeconds: 0, percent: 0, cycle: 0 });
  });
});
