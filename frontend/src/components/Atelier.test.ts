import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ActionRunner, BackendClient, Snapshot } from "../api";
import { Atelier } from "./Atelier";

const baseSnapshot: Snapshot = {
  revision: 2,
  status: { kind: "idle" },
  selectedMacro: "Récolte",
  macros: [{ name: "Récolte", eventCount: 1, durationSeconds: 0.5, protected: false, readable: true, issue: null }],
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

function renderWaiting(action: "recording" | "playing") {
  const snapshot: Snapshot = {
    ...baseSnapshot,
    status: { kind: "waiting_for_foreground", action, macroName: "Récolte" },
  };
  return renderToStaticMarkup(createElement(Atelier, {
    api: {} as BackendClient,
    snapshot,
    busy: null,
    run: (async (_label, callback) => callback()) as ActionRunner,
    online: true,
    showOnboarding: false,
    onDismissOnboarding: () => {},
    onCompleteOnboarding: () => {},
  }));
}

describe("foreground wait in the workshop", () => {
  it("shows recording as pending, with a bounded wait and cancellation", () => {
    const html = renderWaiting("recording");

    expect(html).toContain("En attente de Clash of Clans");
    expect(html).toContain("La préparation de 3 s commencera");
    expect(html).toContain("Cette attente expire automatiquement au bout de 30 secondes.");
    expect(html).toContain("Annuler l’attente");
    expect(html).toMatch(/class="run-copy" role="status" aria-atomic="true"/);
    expect(html).not.toContain("Capture des actions en cours");
  });

  it("shows playback as pending instead of running", () => {
    const html = renderWaiting("playing");

    expect(html).toContain("Lecture en attente pour « Récolte »");
    expect(html).toContain("Aucune action ne démarre avant la détection de Clash of Clans.");
    expect(html).toContain("Annuler l’attente");
    expect(html).not.toContain("Lecture en cours</strong>");
  });
});
