import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ActionRunner, BackendClient, Snapshot } from "../api";
import { Atelier, handleMacroNavigationKey, Sequence } from "./Atelier";

const baseSnapshot: Snapshot = {
  revision: 2,
  status: { kind: "idle" },
  selectedMacro: "Récolte",
  macros: [{ name: "Récolte", eventCount: 1, durationSeconds: 0.5, readable: true, issue: null }],
  settings: {
    loop: false,
    cocPath: "",
    allowCocBackgroundPlayback: true,
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

describe("macro trace presentation", () => {
  it("renders movement endpoints and distinguishes mouse and keyboard markers with an accessible legend", () => {
    const summary = baseSnapshot.macros[0];
    const detail = {
      name: "Récolte",
      steps: [
        { t: 0.1, type: "mouse_move", data: { x: 1 } },
        { t: 0.1, type: "mouse_move", data: { x: 2 } },
        { t: 0.2, type: "mouse_click", data: null },
        { t: 0.1, type: "key_down", data: null },
      ],
    };

    const html = renderToStaticMarkup(createElement(Sequence, { detail, summary }));

    expect(html.match(/class="trace-node/g)).toHaveLength(3);
    expect(html).toContain('class="trace-node is-click"');
    expect(html).toContain('class="trace-node is-key"');
    expect(html).toContain('aria-label="Légende de la trace"');
    expect(html).toContain("Fin d’un déplacement");
    expect(html).toContain("Clic souris");
    expect(html).toContain("Touche clavier");
  });

  it("shows raw event count and recorded duration above execution, without a scrollable list or pagination", () => {
    const snapshot: Snapshot = {
      ...baseSnapshot,
      macros: [{ ...baseSnapshot.macros[0], eventCount: 1_234, durationSeconds: 90 }],
    };
    const html = renderToStaticMarkup(createElement(Atelier, {
      api: {} as BackendClient,
      snapshot,
      busy: null,
      run: (async (_label, callback) => callback()) as ActionRunner,
      online: true,
      showOnboarding: false,
      onDismissOnboarding: () => {},
      onCompleteOnboarding: () => {},
    }));

    const metricsPosition = html.indexOf("sequence-summary");
    const executionPosition = html.indexOf("run-copy");
    expect(html).toContain("1 234");
    expect(html).toContain("1 min 30 s");
    expect(metricsPosition).toBeGreaterThanOrEqual(0);
    expect(metricsPosition).toBeLessThan(executionPosition);
    expect(html).not.toContain("sequence-list");
    expect(html).not.toContain("sequence-more");
    expect(html).not.toContain("sequence-scroll-hint");
    expect(html).not.toContain("Faites défiler la liste");
    expect(html).not.toContain("Séquence prête à être examinée ou rejouée.");
  });

  it("keeps the read error visible for a macro that cannot be read", () => {
    const snapshot: Snapshot = {
      ...baseSnapshot,
      macros: [{ ...baseSnapshot.macros[0], readable: false, issue: "Fichier de macro invalide." }],
    };
    const html = renderToStaticMarkup(createElement(Atelier, {
      api: {} as BackendClient,
      snapshot,
      busy: null,
      run: (async (_label, callback) => callback()) as ActionRunner,
      online: true,
      showOnboarding: false,
      onDismissOnboarding: () => {},
      onCompleteOnboarding: () => {},
    }));

    expect(html).toContain("Fichier de macro invalide.");
    expect(html).not.toContain("Séquence prête à être examinée ou rejouée.");
  });
});

describe("first-use guide and macro selection", () => {
  it("prevents page scrolling at the listbox boundaries without moving selection", () => {
    const boundaryKeys = [
      ["ArrowUp", 0],
      ["Home", 0],
      ["ArrowDown", 1],
      ["End", 1],
    ] as const;

    for (const [key, index] of boundaryKeys) {
      let prevented = false;
      const destination = handleMacroNavigationKey({
        key,
        preventDefault: () => { prevented = true; },
      }, index, 2);

      expect(prevented).toBe(true);
      expect(destination).toBe(index);
    }
  });

  it("leaves unrelated keys to the native button behavior", () => {
    let prevented = false;
    const destination = handleMacroNavigationKey({
      key: "Enter",
      preventDefault: () => { prevented = true; },
    }, 0, 2);

    expect(prevented).toBe(false);
    expect(destination).toBeNull();
  });

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

    expect(html).toContain("La coche marque la macro sélectionnée, affichée à droite.");
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
    expect(html).toContain("La séquence est prête. Ouvrez Clash of Clans; le replay arrière-plan s’active dans Réglages &gt; Général.");
    expect(html).toContain(">Aller au bouton Lire</button>");
    expect(html).toContain('id="play-macro-button"');
    expect(html).toContain(">Lire la macro</button>");
    expect(html.match(/>Lire la macro<\/button>/g)).toHaveLength(1);
    expect(html).toContain(">Terminer le guide</button>");
    expect(html).toContain("Clash of Clans doit être ouvert; le replay reste ciblé sur sa fenêtre.");
  });

  it("keeps both creation actions without repeating instructions during onboarding", () => {
    const html = renderToStaticMarkup(createElement(Atelier, {
      api: {} as BackendClient,
      snapshot: { ...baseSnapshot, selectedMacro: null, macros: [], onboardingComplete: false },
      busy: null,
      run: (async (_label, callback) => callback()) as ActionRunner,
      online: true,
      showOnboarding: true,
      onDismissOnboarding: () => {},
      onCompleteOnboarding: () => {},
    }));

    expect(html).toContain(">Créer une macro</button>");
    expect(html).toContain('aria-label="Créer une macro"');
    expect(html).toContain("Votre première macro apparaîtra ici.");
    expect(html).not.toContain("Utilisez le bouton + dans la bibliothèque pour commencer.");
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
