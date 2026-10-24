import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ActionRunner, BackendClient, Snapshot } from "../api";
import { Atelier, Sequence } from "./Atelier";

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

  it("shows the live playback cursor and freezes it when playback is paused", () => {
    const summary = { ...baseSnapshot.macros[0], durationSeconds: 20 };
    const detail = { name: "Récolte", steps: [{ t: 20, type: "mouse_click", data: null }] };
    const playing = renderToStaticMarkup(createElement(Sequence, {
      detail,
      summary,
      playback: { elapsedSeconds: 5, isRunning: true, looped: false },
    }));
    const paused = renderToStaticMarkup(createElement(Sequence, {
      detail,
      summary,
      playback: { elapsedSeconds: 5, isRunning: false, looped: false },
    }));

    expect(playing).toContain('class="trace-playhead is-running"');
    expect(playing).toContain('style="left:25%"');
    expect(playing).toContain("5 s / 20 s");
    expect(paused).toContain('class="trace-playhead"');
    expect(paused).toContain('style="left:25%"');
  });

  it("shows session and active time as whole seconds", () => {
    const snapshot: Snapshot = {
      ...baseSnapshot,
      status: { kind: "playing", macroName: "Récolte", elapsedSeconds: 10.1 },
      session: { elapsedSeconds: 10.1, cycles: 1 },
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

    expect(html).toContain("Temps écoulé : 10 s");
    expect(html).toContain("<strong>10 s</strong><small>temps de session</small>");
    expect(html).not.toContain("10,1 s");
  });

  it("shows the recording countdown as a whole 3, 2, 1 value", () => {
    const snapshot: Snapshot = {
      ...baseSnapshot,
      status: { kind: "recording", macroName: "Récolte", phase: "preparing", countdownSeconds: 2.0149864654, elapsedSeconds: 0 },
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

    expect(html).toContain("Préparation · 3 s");
    expect(html).toContain("La capture commence dans 3 s.");
    expect(html).not.toContain("2.0149864654");
  });

  it("places pause-on-focus-loss beside loop and reflects the shared setting", () => {
    const snapshot: Snapshot = {
      ...baseSnapshot,
      settings: { ...baseSnapshot.settings, allowCocBackgroundPlayback: false },
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

    expect(html.match(/class="macro-option(?: |")/g)).toHaveLength(2);
    expect(html).toMatch(/Pause hors CoC<\/span><small>Quand le jeu perd le focus<\/small><\/span><input type="checkbox" checked=""/);
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
  it("keeps macro browsing out of the workshop content", () => {
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

    expect(html).not.toContain("Bibliothèque des macros");
    expect(html).not.toContain("La coche marque la macro sélectionnée");
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

  it("keeps the onboarding creation action without duplicating library instructions", () => {
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
  });
});
