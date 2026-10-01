import { describe, expect, it } from "vitest";
import { normalizeShortcut } from "./normalizeShortcut";

function keyboardEvent(code: string, key: string): KeyboardEvent {
  return { code, key, ctrlKey: true, altKey: false, shiftKey: true, metaKey: false } as KeyboardEvent;
}

describe("normalizeShortcut", () => {
  it.each([
    { code: "Digit1", key: "!", expected: "Ctrl+Shift+1" },
    { code: "Numpad1", key: "1", expected: "Ctrl+Shift+Numpad1" },
  ])("uses the canonical key for $code", ({ code, key, expected }) => {
    expect(normalizeShortcut(keyboardEvent(code, key))).toBe(expected);
  });
});
