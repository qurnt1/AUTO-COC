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
  function renderSettings(allowCocBackgroundPlayback: boolean, initialTab: "general" | "shortcuts" | "telegram" | "system" = "general", telegram = snapshot.settings.telegram) {
    return renderToStaticMarkup(createElement(Settings, {
      api: {} as BackendClient,
      snapshot: { ...snapshot, settings: { ...snapshot.settings, allowCocBackgroundPlayback, telegram } },
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

  it("shows the Google Play Games URI as selectable copy text", () => {
    const html = renderSettings(true);
    const uriField = html.match(/<textarea[^>]*class="path-uri-copy"[^>]*>(.*?)<\/textarea>/)?.[1];

    expect(html).toContain("copiez l’URI Google Play Games affichée sous le champ");
    expect(html).toContain('placeholder="Ex. C:\\…\\Clash of Clans.lnk"');
    expect(html).toContain("URI Google Play Games. Faites Tab puis Ctrl+C, ou sélectionnez le texte à la souris.");
    expect(html).toContain('aria-label="URI Google Play Games à copier"');
    expect(html).toContain('readOnly=""');
    expect(uriField).toBe("googleplaygames://launch/?id=com.supercell.clashofclans&amp;lid=1&amp;pid=1");
  });

  it("explains that a screenshot activates and captures Clash of Clans", () => {
    const html = renderSettings(true, "system");

    expect(html).toContain("Capture de Clash of Clans");
    expect(html).toContain("CoC passe au premier plan.");
    expect(html).toContain(">Capturer CoC</button>");
  });

  it("explains the focus-loss pause and targeted background behavior", () => {
    const html = renderSettings(true);

    expect(html).toContain("Mettre en pause si CoC perd le focus");
    expect(html).toContain("Activée, la lecture se met en pause dès que vous quittez le jeu.");
    expect(html).toContain("Si sa fenêtre ou son processus disparaît, elle s’arrête toujours.");
    expect(html).toContain("Désactivée, le replay peut continuer en arrière-plan, avec les actions ciblées sur CoC.");
    expect(html).not.toContain('type="checkbox" checked=""');
  });

  it("checks the focus-loss pause when background replay is disabled", () => {
    const html = renderSettings(false);
    const control = html.match(/<label class="switch-control"><span class="sr-only">Mettre en pause si CoC perd le focus<\/span>(.*?)<\/label>/)?.[1];

    expect(control).toContain('type="checkbox"');
    expect(control).toContain('checked=""');
    expect(html).toContain("Activée, la lecture se met en pause dès que vous quittez le jeu.");
    expect(html).toContain("L’enregistrement reste limité au premier plan.");
  });

  it("guides bot creation first and unlocks token verification only after saving", () => {
    const html = renderSettings(true, "telegram");

    expect(html).toContain('aria-label="Étapes pour connecter Telegram"');
    expect(html).toContain("Créer votre bot");
    expect(html).toContain("@BotFather");
    expect(html).toContain('id="telegram-bot-username"');
    expect(html).toContain("Nom d’utilisateur du bot (facultatif)");
    expect(html).toContain("envoyez /newbot. Choisissez un nom visible");
    expect(html).toContain("Vous n’avez pas besoin de le saisir ici");
    expect(html).toContain('placeholder="Ex. @mon_bot"');
    expect(html).toContain("Copier /newbot");
    expect(html).toContain('id="telegram-token" type="password"');
    expect(html).toContain('class="telegram-feedback" role="status"></p>');
    expect(html).not.toMatch(/<button[^>]*>Vérifier le token<\/button>/);
    expect(html).not.toContain("Générer une commande d’appairage");
    expect(html).not.toContain("/start");

    const savedToken = {
      ...snapshot.settings.telegram,
      tokenConfigured: true,
      status: "disconnected" as const,
    };
    const nextStep = renderSettings(true, "telegram", savedToken);

    expect(nextStep).toContain("Vérifier le token");
    expect(nextStep).toContain("Remplacer le token");
    expect(nextStep).not.toContain("Générer une commande d’appairage");
    expect(nextStep).not.toContain("/start");
    expect(nextStep).not.toContain('id="telegram-token"');
  });

  it("reveals the pairing command only after generation and tells the user to return", () => {
    const verifiedTelegram = {
      ...snapshot.settings.telegram,
      tokenConfigured: true,
      status: "connected" as const,
    };
    const verified = renderSettings(true, "telegram", verifiedTelegram);

    expect(verified).toContain("Générer une commande d’appairage");
    expect(verified).toContain("Revérifier le token");
    expect(verified).not.toContain("/start");

    const waitingTelegram = {
      ...verifiedTelegram,
      status: "waiting_pairing" as const,
      pairingCode: "CODE-SANS-SECRET",
      pairingExpiresAt: new Date(Date.now() + 60_000).toISOString(),
    };
    const waiting = renderSettings(true, "telegram", waitingTelegram);

    expect(waiting).toContain("Dans Telegram, ouvrez une conversation privée avec le nom du bot donné par BotFather");
    expect(waiting).toContain("/start CODE-SANS-SECRET");
    expect(waiting).toContain("Revenez dans AUTO-COC : la confirmation apparaîtra ici automatiquement.");
    expect(waiting).toContain('class="telegram-feedback copy-feedback" role="status"></p>');
    expect(waiting).toContain('class="telegram-feedback verification-feedback" role="status"></p>');
    expect(waiting).toContain("Token enregistré et protégé. Il reste masqué.");
    expect(waiting).not.toContain('id="telegram-token"');
  });

  it("keeps the optional Telegram purpose and distinguishes its current statuses", () => {
    const html = renderSettings(true, "telegram");

    expect(html).toContain("L’atelier sur ce PC fonctionne sans Telegram.");
    expect(html).toContain("Les macros sur ce PC fonctionnent sans Telegram.");
    expect(html).toContain("Vérifier le token");
    expect(html).toContain("Associer votre conversation");
    expect(html).not.toContain("Tester la connexion");
    expect(html).not.toContain("migration");
  });

  it("keeps Telegram token, BotFather copy, verification, and pairing feedback separate", () => {
    const firstStep = renderSettings(true, "telegram");
    expect(firstStep.match(/class="telegram-feedback" role="status"/g)).toHaveLength(2);

    const waitingTelegram = {
      ...snapshot.settings.telegram,
      tokenConfigured: true,
      status: "waiting_pairing" as const,
      pairingCode: "CODE-SANS-SECRET",
      pairingExpiresAt: new Date(Date.now() + 60_000).toISOString(),
    };
    const html = renderSettings(true, "telegram", waitingTelegram);

    expect(html.match(/role="status"/g)).toHaveLength(3);
    expect(html).toContain("Copier la commande");
    expect(html).toContain('class="telegram-feedback copy-feedback" role="status"></p>');
    expect(html).toContain('class="telegram-feedback verification-feedback" role="status"></p>');
    expect(html).not.toContain("Copier la commande d’appairage");
  });

  it("distinguishes a saved token from a verified bot and a pending pairing code", () => {
    const savedTelegram = {
      ...snapshot.settings.telegram,
      tokenConfigured: true,
      status: "disconnected" as const,
    };
    const saved = renderSettings(true, "telegram", savedTelegram);
    expect(saved).toContain("Token enregistré");
    expect(saved).toContain("À vérifier");
    expect(saved).toContain("Vérifier le token");
    expect(saved).toContain("Remplacer le token");

    const waitingTelegram = {
      ...snapshot.settings.telegram,
      tokenConfigured: true,
      status: "waiting_pairing" as const,
      pairingCode: "CODE-SANS-SECRET",
      pairingExpiresAt: new Date(Date.now() + 60_000).toISOString(),
    };
    const waiting = renderSettings(true, "telegram", waitingTelegram);
    expect(waiting).toContain("Code en attente");
    expect(waiting).toContain("/start CODE-SANS-SECRET");
    expect(waiting).toContain("Le code expire le");
  });

  it("shows a completed Telegram conversation association explicitly", () => {
    const pairedTelegram = {
      ...snapshot.settings.telegram,
      tokenConfigured: true,
      paired: true,
      status: "connected" as const,
    };

    expect(renderSettings(true, "telegram", pairedTelegram)).toContain("Conversation associée");
  });

  it("does not report a paired bot as verified after Telegram rejects its token", () => {
    const rejectedTelegram = {
      ...snapshot.settings.telegram,
      tokenConfigured: true,
      paired: true,
      status: "token_rejected" as const,
    };
    const html = renderSettings(true, "telegram", rejectedTelegram);

    expect(html).toContain("Token refusé");
    expect(html).toContain("Vérification requise");
    const pairingStepStateClass = html.match(/<span class="([^"]*telegram-step-state[^"]*)">Vérification requise<\/span>/)?.[1];
    expect(pairingStepStateClass).not.toContain("done");
    expect(html).not.toContain("Bot vérifié");
    expect(html).toContain("Vérifier le token");
  });

  it.each(["error", "token_rejected", "network_error"] as const)("marks Telegram as needing attention for %s", (status) => {
    const telegram = {
      ...snapshot.settings.telegram,
      tokenConfigured: true,
      status,
    };
    const html = renderSettings(true, "general", telegram);

    expect(html).toContain('Telegram<i aria-hidden="true"></i>');
  });
});
