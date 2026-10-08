import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ActionRunner, BackendClient, Snapshot } from "../api";
import { Settings } from "./Settings";
import { normalizeShortcut } from "./normalizeShortcut";

const snapshot: Snapshot = {
  revision: 1,
  status: { kind: "idle" },
  selectedMacro: null,
  macros: [],
  settings: {
    loop: false,
    cocPath: "",
    requireCocForeground: true,
    shortcuts: { toggle: "F1", play: "Ctrl+Shift+1", stop: "Ctrl+Shift+0" },
    telegram: {
      tokenConfigured: false,
      paired: false,
      status: "not_configured",
      pairingCode: null,
      pairingExpiresAt: null,
    },
  },
  session: { elapsedSeconds: 0, cycles: 0 },
  migration: { sourceSelected: false, available: false, alreadyImported: false },
  onboardingComplete: true,
  lastError: null,
};

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

describe("Settings", () => {
  function renderSettings(requireCocForeground: boolean, initialTab: "general" | "shortcuts" = "general") {
    return renderToStaticMarkup(createElement(Settings, {
      api: {} as BackendClient,
      snapshot: { ...snapshot, settings: { ...snapshot.settings, requireCocForeground } },
      busy: null,
      run: (async (_label, action) => action()) as ActionRunner,
      online: true,
      initialTab,
    }));
  }

  it("gives each shortcut action a distinct accessible name", () => {
    const html = renderSettings(true, "shortcuts");
    const names = [...html.matchAll(/aria-label="(Modifier le raccourci [^"]+)"/g)].map((match) => match[1]);

    expect(names).toEqual([
      "Modifier le raccourci Basculer lecture / arrêt",
      "Modifier le raccourci Lire la macro",
      "Modifier le raccourci Arrêter",
    ]);
  });

  it("explains the safety pause and its Android crash limit", () => {
    const html = renderSettings(true);

    expect(html).toContain("Pause de sécurité si Clash of Clans perd le premier plan");
    expect(html).toContain("Au démarrage depuis l’atelier, l’attente de détection de Clash of Clans au premier plan dure au maximum 30 secondes.");
    expect(html).toContain("Cliquez sur « Reprendre » dans l’atelier, puis revenez dans Clash of Clans");
    expect(html).toContain("« Annuler la reprise » laisse la macro en pause.");
    expect(html).toContain("Un crash Android peut rester invisible si la fenêtre hôte crosvm reste ouverte avec le même titre.");
    expect(html).toContain('type="checkbox" checked=""');
  });

  it("renders the foreground requirement unchecked when it is disabled", () => {
    const html = renderSettings(false);
    const control = html.match(/<label class="switch-control"><span class="sr-only">Activer la pause de sécurité Clash of Clans<\/span>(.*?)<\/label>/)?.[1];

    expect(control).toContain('type="checkbox"');
    expect(control).not.toContain("checked");
  });
});
