import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ActionRunner, BackendClient, Snapshot } from "../api";
import { Atelier } from "./Atelier";

const baseSnapshot: Snapshot = {
  revision: 2,
  status: { kind: "idle" },
  selectedMacro: "Récolte",
  macros: [{ name: "Récolte", eventCount: 1, durationSeconds: 0.5, readable: true, issue: null }],
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
  onboardingComplete: true,
  lastError: null,
};

function renderWaiting(action: "recording" | "playing" | "resuming") {
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

function renderPaused() {
  const snapshot: Snapshot = {
    ...baseSnapshot,
    status: { kind: "paused", macroName: "Récolte", elapsedSeconds: 12.5 },
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

  it("distinguishes a pending resume and offers its dedicated cancellation", () => {
    const html = renderWaiting("resuming");

    expect(html).toContain("Reprise en attente pour « Récolte »");
    expect(html).toContain("La lecture reprendra dès que le jeu sera de nouveau au premier plan.");
    expect(html).toContain(">Annuler la reprise</button>");
    expect(html).not.toContain(">Annuler l’attente</button>");
  });

  it("shows the safety pause with resume and stop controls", () => {
    const html = renderPaused();

    expect(html).toContain("En pause, CoC n&#x27;est plus la fenêtre active");
    expect(html).toContain("Cliquez sur « Reprendre », puis revenez dans Clash of Clans.");
    expect(html).toContain("La lecture reprendra dès que le jeu sera de nouveau au premier plan.");
    expect(html).toMatch(/class="run-copy" role="status" aria-atomic="true"/);
    expect(html).toContain(">Reprendre</button>");
    expect(html).toContain(">Arrêter la lecture</button>");
    expect(html).not.toContain(">Lire la macro</button>");
  });
});

describe("first-use guide and macro selection", () => {
  it("explains the highlighted selection without an ambiguous marker", () => {
    const html = renderToStaticMarkup(createElement(Atelier, {
      api: {} as BackendClient,
      snapshot: baseSnapshot,
      busy: null,
      run: (async (_label, callback) => callback()) as ActionRunner,
      online: true,
      showOnboarding: false,
      onDismissOnboarding: () => {},
      onCompleteOnboarding: () => {},
    }));

    expect(html).toContain("La ligne en surbrillance est celle qui est sélectionnée.");
    expect(html).toContain('aria-selected="true"');
    expect(html).not.toContain("macro-item-marker");
    expect(html).not.toContain("selected-chevron");
    expect(html).not.toContain("Nouvelle macro");
  });

  it("hides the selection hint when the library has no visible macros", () => {
    const html = renderToStaticMarkup(createElement(Atelier, {
      api: {} as BackendClient,
      snapshot: { ...baseSnapshot, selectedMacro: null, macros: [] },
      busy: null,
      run: (async (_label, callback) => callback()) as ActionRunner,
      online: true,
      showOnboarding: false,
      onDismissOnboarding: () => {},
      onCompleteOnboarding: () => {},
    }));

    expect(html).not.toContain("library-selection-hint");
  });

  it("offers the next useful action based on the selected macro", () => {
    const html = renderToStaticMarkup(createElement(Atelier, {
      api: {} as BackendClient,
      snapshot: baseSnapshot,
      busy: null,
      run: (async (_label, callback) => callback()) as ActionRunner,
      online: true,
      showOnboarding: true,
      onDismissOnboarding: () => {},
      onCompleteOnboarding: () => {},
    }));

    expect(html).toContain("Première utilisation · étape 3 sur 3");
    expect(html).toContain("La séquence est prête. Clash of Clans doit être la fenêtre active pour la lire.");
    expect(html).toContain(">Aller au bouton Lire</button>");
    expect(html).toContain('id="play-macro-button"');
    expect(html).toContain(">Lire la macro</button>");
    expect(html.match(/>Lire la macro<\/button>/g)).toHaveLength(1);
    expect(html).toContain(">Terminer le guide</button>");
    expect(html).toContain("Pour lire, Clash of Clans doit être la fenêtre active.");
  });

  it("labels the select-stage action as browsing and leaves selection to the user", () => {
    const html = renderToStaticMarkup(createElement(Atelier, {
      api: {} as BackendClient,
      snapshot: { ...baseSnapshot, selectedMacro: null },
      busy: null,
      run: (async (_label, callback) => callback()) as ActionRunner,
      online: true,
      showOnboarding: true,
      onDismissOnboarding: () => {},
      onCompleteOnboarding: () => {},
    }));

    expect(html).toContain(">Parcourir la liste des macros</button>");
    expect(html).toContain('id="macro-R%C3%A9colte"');
    expect(html).toContain('role="option" aria-selected="false"');
  });
});
