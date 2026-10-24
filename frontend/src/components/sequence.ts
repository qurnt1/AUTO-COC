import type { MacroStep } from "../api";

export const MAX_TIMELINE_MARKERS = 32;

export type TimelineMarker = { step: MacroStep; index: number; left: string };
export type PlaybackPosition = { elapsedSeconds: number; percent: number; cycle: number };

export function getPlaybackPosition(
  elapsedSeconds: number,
  durationSeconds: number,
  looped: boolean,
): PlaybackPosition {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return { elapsedSeconds: 0, percent: 0, cycle: 0 };
  }

  const elapsed = Number.isFinite(elapsedSeconds) ? Math.max(0, elapsedSeconds) : 0;
  const cycle = looped ? Math.floor(elapsed / durationSeconds) : 0;
  const position = looped ? elapsed % durationSeconds : Math.min(elapsed, durationSeconds);
  return {
    elapsedSeconds: position,
    percent: Number(((position / durationSeconds) * 100).toFixed(2)),
    cycle,
  };
}

export function getTimelineMarkers(
  steps: readonly MacroStep[],
  requestedLimit = MAX_TIMELINE_MARKERS,
): TimelineMarker[] {
  if (steps.length === 0) return [];

  const limit = Math.max(1, Math.trunc(requestedLimit));
  let totalTime = 0;
  for (const step of steps) totalTime += Math.max(0, step.t);
  const duration = Math.max(totalTime, 0.001);
  const timePerMarker = duration / Math.max(limit - 1, 1);
  const markers: TimelineMarker[] = [];
  let elapsed = 0;
  let nextTarget = 0;
  let finalMarker: TimelineMarker | undefined;

  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    elapsed += Math.max(0, step.t);
    const isLastMoveInRun = step.type === "mouse_move" && steps[index + 1]?.type !== "mouse_move";
    if (step.type !== "mouse_move" || isLastMoveInRun) {
      const percent = Math.min(99, Math.max(1, (elapsed / duration) * 100));
      const marker = { step, index, left: `${Number(percent.toFixed(2))}%` };
      finalMarker = marker;

      if (markers.length < limit - 1 && elapsed >= nextTarget) {
        markers.push(marker);
        nextTarget = (Math.floor(elapsed / timePerMarker) + 1) * timePerMarker;
      }
    }
  }

  if (finalMarker && markers.at(-1)?.index !== finalMarker.index) markers.push(finalMarker);
  return markers;
}

export function getTraceNodeClass(type: string): string {
  if (type === "mouse_move") return "is-move";
  if (type === "mouse_click") return "is-click";
  if (type.startsWith("key")) return "is-key";
  return "is-other";
}
