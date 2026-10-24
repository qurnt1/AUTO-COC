import type { MacroSummary } from "../api";

export function filterMacros(macros: readonly MacroSummary[], query: string): MacroSummary[] {
  const normalizedQuery = query.trim().toLocaleLowerCase("fr");
  return normalizedQuery
    ? macros.filter((macro) => macro.name.toLocaleLowerCase("fr").includes(normalizedQuery))
    : [...macros];
}

export function handleMacroNavigationKey(
  event: { key: string; preventDefault: () => void },
  index: number,
  itemCount: number,
): number | null {
  let destination: number;
  switch (event.key) {
    case "ArrowDown": destination = index + 1; break;
    case "ArrowUp": destination = index - 1; break;
    case "Home": destination = 0; break;
    case "End": destination = itemCount - 1; break;
    default: return null;
  }

  event.preventDefault();
  return Math.max(0, Math.min(destination, itemCount - 1));
}
