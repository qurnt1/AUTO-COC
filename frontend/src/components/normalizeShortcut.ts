export function normalizeShortcut(event: KeyboardEvent): string | null {
  if (event.metaKey) return null;
  const modifiers = [event.ctrlKey ? "Ctrl" : "", event.altKey ? "Alt" : "", event.shiftKey ? "Shift" : ""].filter(Boolean);
  const physicalDigit = /^(Digit|Numpad)([0-9])$/.exec(event.code);
  let key = physicalDigit ? (physicalDigit[1] === "Numpad" ? `Numpad${physicalDigit[2]}` : physicalDigit[2]) : event.key;
  if (key === " ") key = "Space";
  else if (key.length === 1) key = key.toUpperCase();
  else key = key.replace(/^Arrow/, "");
  if (key === "Control" || key === "Alt" || key === "Shift") return null;
  if (!modifiers.length && !/^F([1-9]|1[0-2])$/.test(key)) return null;
  return [...modifiers, key].join("+");
}
