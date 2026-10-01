import type { MacroStep } from "../api";

export const INITIAL_SEQUENCE_PAGE_SIZE = 8;
export const SEQUENCE_PAGE_SIZE = 100;
export const MAX_TIMELINE_MARKERS = 160;

export type SequencePage = { start: number; end: number };
export type TimelineMarker = { step: MacroStep; index: number; left: string };

export function getSequencePageBounds(total: number, requestedStart: number): SequencePage {
  const count = Math.max(0, Math.trunc(total));
  const start = count === 0 ? 0 : Math.min(count - 1, Math.max(0, Math.trunc(requestedStart)));
  const pageSize = start === 0 ? INITIAL_SEQUENCE_PAGE_SIZE : SEQUENCE_PAGE_SIZE;
  return { start, end: Math.min(count, start + pageSize) };
}

export function nextSequencePageStart(total: number, start: number): number {
  const { end } = getSequencePageBounds(total, start);
  return end < total ? end : start;
}

export function previousSequencePageStart(start: number): number {
  const index = Math.max(0, Math.trunc(start));
  return index <= INITIAL_SEQUENCE_PAGE_SIZE
    ? 0
    : Math.max(INITIAL_SEQUENCE_PAGE_SIZE, index - SEQUENCE_PAGE_SIZE);
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
  const indexes: number[] = [];

  if (steps.length <= limit) {
    for (let index = 0; index < steps.length; index += 1) indexes.push(index);
  } else {
    const timePerMarker = duration / Math.max(limit - 1, 1);
    let elapsed = 0;
    let nextTarget = 0;
    for (let index = 0; index < steps.length && indexes.length < limit - 1; index += 1) {
      if (elapsed >= nextTarget) {
        indexes.push(index);
        nextTarget = (Math.floor(elapsed / timePerMarker) + 1) * timePerMarker;
      }
      elapsed += Math.max(0, steps[index].t);
    }

    const lastIndex = steps.length - 1;
    if (indexes.at(-1) !== lastIndex) indexes.push(lastIndex);
  }

  const selected = new Set(indexes);
  const markers: TimelineMarker[] = [];
  let elapsed = 0;
  for (let index = 0; index < steps.length; index += 1) {
    if (selected.has(index)) {
      const percent = Math.min(99, Math.max(1, (elapsed / duration) * 100));
      markers.push({ step: steps[index], index, left: `${percent}%` });
    }
    elapsed += Math.max(0, steps[index].t);
  }
  return markers;
}
