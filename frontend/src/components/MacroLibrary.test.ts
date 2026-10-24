import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { MacroSummary } from "../api";
import { MacroLibrary } from "./MacroLibrary";
import { filterMacros, handleMacroNavigationKey } from "./macroLibraryUtils";

const macros: MacroSummary[] = Array.from({ length: 7 }, (_, index) => ({
  name: index === 0 ? "Récolte" : `Macro ${index + 1}`,
  eventCount: index + 1,
  durationSeconds: index + 1,
  readable: true,
  issue: null,
}));

function renderLibrary(overrides: Partial<Parameters<typeof MacroLibrary>[0]> = {}) {
  return renderToStaticMarkup(createElement(MacroLibrary, {
    macros,
    selectedMacro: "Macro 3",
    query: "",
    loading: false,
    disabled: false,
    onQueryChange: () => {},
    onCreate: () => {},
    onSelect: () => {},
    ...overrides,
  }));
}

describe("macro library", () => {
  it("shows every macro and marks the selected option", () => {
    const html = renderLibrary();

    expect(html.match(/role="option"/g)).toHaveLength(7);
    for (const macro of macros) expect(html).toContain(macro.name);
    expect(html).toContain('aria-selected="true"');
    expect(html).toContain('aria-label="Créer une macro"');
  });

  it("filters names case-insensitively and trims the query", () => {
    expect(filterMacros(macros, "  RÉCOLTE ").map(({ name }) => name)).toEqual(["Récolte"]);
    expect(filterMacros(macros, "macro 5").map(({ name }) => name)).toEqual(["Macro 5"]);
  });

  it("shows an empty-search result and keeps the search accessible", () => {
    const html = renderLibrary({ query: "introuvable" });

    expect(html).toContain('aria-label="Rechercher une macro"');
    expect(html).toContain("Aucune macro trouvée.");
    expect(html).toContain("0 / 7");
    expect(html).not.toContain('role="listbox"');
  });

  it("announces loading while the first snapshot is pending", () => {
    const html = renderLibrary({ macros: [], selectedMacro: null, loading: true });

    expect(html).toContain('role="status"');
    expect(html).toContain("Chargement des macros…");
  });
});

describe("macro list keyboard navigation", () => {
  it("moves through options and prevents page scrolling at boundaries", () => {
    for (const [key, index, expected] of [
      ["ArrowDown", 0, 1],
      ["ArrowUp", 0, 0],
      ["Home", 1, 0],
      ["End", 0, 1],
    ] as const) {
      let prevented = false;
      const destination = handleMacroNavigationKey({
        key,
        preventDefault: () => { prevented = true; },
      }, index, 2);

      expect(prevented).toBe(true);
      expect(destination).toBe(expected);
    }
  });

  it("leaves Enter to the native button behavior", () => {
    let prevented = false;
    const destination = handleMacroNavigationKey({
      key: "Enter",
      preventDefault: () => { prevented = true; },
    }, 0, 2);

    expect(prevented).toBe(false);
    expect(destination).toBeNull();
  });
});
