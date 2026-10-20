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
  function renderSettings(requireCocForeground: boolean, initialTab: "general" | "shortcuts" | "telegram" = "general", telegram = snapshot.settings.telegram) {
    return renderToStaticMarkup(createElement(Settings, {
      api: {} as BackendClient,
      snapshot: { ...snapshot, settings: { ...snapshot.settings, requireCocForeground, telegram } },
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

  it("explains the CoC-only replay and the automatic pause option", () => {
    const html = renderSettings(true);

    expect(html).toContain("Pause automatique si Clash of Clans perd le premier plan");
    expect(html).toContain("Le replay démarre seulement si CoC est au premier plan.");
    expect(html).toContain("Activée, l’option attend jusqu’à 30 s que le jeu soit actif");
    expect(html).toContain("Désactivée, CoC doit déjà être actif et une perte de focus arrête le replay.");
    expect(html).toContain("Les entrées restent ciblées sur la fenêtre du jeu.");
    expect(html).toContain('type="checkbox" checked=""');
  });

  it("renders the foreground requirement unchecked when it is disabled", () => {
    const html = renderSettings(false);
    const control = html.match(/<label class="switch-control"><span class="sr-only">Activer la pause automatique Clash of Clans<\/span>(.*?)<\/label>/)?.[1];

    expect(control).toContain('type="checkbox"');
    expect(control).not.toContain("checked");
    expect(html).toContain("Le replay démarre seulement si CoC est au premier plan.");
    expect(html).toContain("Les entrées restent ciblées sur la fenêtre du jeu.");
  });

  it("shows Telegram setup as three concrete steps and names the token check accurately", () => {
    const html = renderSettings(true, "telegram");

    expect(html).toContain('aria-label="Étapes pour connecter Telegram"');
    expect(html).toContain("Créer votre bot");
    expect(html).toContain("@BotFather");
    expect(html).toContain("/newbot");
    expect(html).toContain("Vérifier le token");
    expect(html).toContain("Associer votre conversation");
    expect(html).toContain("Reprenez le nom <code>@username</code> donné par BotFather, recherchez-le dans Telegram et ouvrez une conversation privée avec ce bot.");
    expect(html).toContain("Envoyez ensuite <code>/start CODE</code> ci-dessous.");
    expect(html).toContain("Enregistrez le token pour pouvoir le vérifier.");
    expect(html).not.toContain("Tester la connexion");
    expect(html).not.toContain("migration");
  });

  it("keeps Telegram save, verification, and copy feedback separate and labels the copy action", () => {
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
    expect(waiting).toContain("Ce code expire le");
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
});
