import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ActionRunner, BackendClient, HelpDocument, Snapshot } from "../api";
import { Help, visibleHelpSections } from "./Help";

function renderHelp(snapshot: Snapshot | null = null) {
  return renderToStaticMarkup(createElement(Help, {
    api: {} as BackendClient,
    snapshot,
    busy: null,
    run: (async (_label, action) => action()) as ActionRunner,
    online: true,
    navigate: () => {},
    onResumeOnboarding: () => {},
    onCompleteOnboarding: () => {},
  }));
}

describe("help page", () => {
  it("offers task-oriented starting points and keeps safety details behind a disclosure", () => {
    const html = renderHelp();

    expect(html).toContain("Que voulez-vous faire ?");
    expect(html).toContain("Créer une macro");
    expect(html).toContain("Lire une macro");
    expect(html).toContain("Piloter avec Telegram");
    expect(html).toContain("Créez le bot avec @BotFather (/newbot)");
    expect(html).toContain("l’identifiant du bot est facultatif");
    expect(html).toContain("générez la commande d’appairage et envoyez-la au bot");
    expect(html.match(/Ouvrir l’atelier<svg/g)).toHaveLength(2);
    expect(html).toContain("Configurer Telegram<svg");
    expect(html).toContain('aria-label="Ouvrir l’atelier : Créer une macro"');
    expect(html).toContain('aria-label="Ouvrir l’atelier : Lire une macro"');
    expect(html).toContain('aria-label="Configurer Telegram : Piloter avec Telegram"');
    expect(html).toContain("enregistrez plus de 3 s");
    expect(html).toContain("Les 3 dernières secondes sont retirées.");
    expect(html).toContain("Pour autoriser le replay en arrière-plan, désactivez « Mettre en pause si CoC perd le focus » dans Réglages &gt; Général.");
    expect(html).toContain("leurs relâchements restent enregistrés pour éviter une touche ou un bouton bloqué");
    expect(html).toContain("Voir les détails de sûreté");
    expect(html).toContain("Les nouvelles macros suivent les coordonnées de CoC si sa fenêtre bouge");
    expect(html).toContain("Les déplacements souris hors zone sont ignorés, mais un clic ou un défilement hors zone arrête le replay");
    expect(html).not.toMatch(/<details[^>]*open/);
  });

  it("does not render legacy migration or import language", () => {
    const html = renderHelp();

    expect(html).not.toMatch(/migration|importer vos données|ancienne installation/i);
  });

  it("keeps live shortcut guidance and removes stale import and migration help", () => {
    const sections: HelpDocument["sections"] = [
      { title: "Créer une macro", body: "Enregistrez une séquence dans l’atelier." },
      { title: "Raccourcis par défaut", body: "F1 démarre la lecture." },
      { title: "Importer vos données", body: "Récupérez une ancienne installation." },
      { title: "Migration", body: "Transférez vos macros." },
      { title: "Importer", body: "Importation depuis une ancienne version." },
      { title: "Imports", body: "Cette action reprend votre bibliothèque." },
      { title: "Migrations", body: "Les macros seront transférées." },
      { title: "Données", body: "Les macros sont conservées sur ce PC." },
    ];

    expect(visibleHelpSections(sections)).toEqual([sections[0], sections[7]]);
  });

  it("shows the saved shortcuts rather than hard-coded defaults", () => {
    const html = renderHelp({
      onboardingComplete: true,
      selectedMacro: null,
      macros: [],
      settings: { shortcuts: { toggle: "F8", play: "Ctrl+Alt+P", stop: "F10" } },
    } as unknown as Snapshot);

    expect(html).toContain("F8");
    expect(html).toContain("Ctrl+Alt+P");
    expect(html).toContain("F10");
    expect(html).not.toContain("Ctrl+Shift+1");
  });

  it("shows one active onboarding instruction with compact progress and working actions", () => {
    const html = renderHelp({ onboardingComplete: false, selectedMacro: null, macros: [], settings: { shortcuts: null } } as unknown as Snapshot);

    expect(html).toContain("Étape active · 1 sur 3");
    expect(html).toContain('<h2 id="onboarding-title">Créer une macro</h2>');
    expect(html).toContain("Donnez un nom à la macro que vous allez enregistrer.");
    expect(html).toContain('aria-label="Progression du guide, étape 1 sur 3"');
    expect(html).toContain('aria-label="Enregistrer une séquence, à venir"');
    expect(html.match(/class="onboarding-checklist[\s\S]*?<\/ol>/)?.[0].match(/<li /g)).toHaveLength(3);
    expect(html).toContain("class=\"onboarding-checklist\"");
    expect(html).not.toContain("id=\"first-steps-title\"");
    expect(html).not.toContain("Enregistrer une séquence</strong>");
    expect(html).toContain("Reprendre dans l’atelier");
    expect(html).toContain("Terminer le guide");
  });

  it("keeps the three-second rule in the active recording instruction", () => {
    const html = renderHelp({
      onboardingComplete: false,
      selectedMacro: "Récolte",
      macros: [{ name: "Récolte", eventCount: 0, durationSeconds: 0, readable: true, issue: null }],
      settings: { shortcuts: null },
    } as unknown as Snapshot);

    expect(html).toContain("Étape active · 2 sur 3");
    expect(html).toContain('<h2 id="onboarding-title">Enregistrer une séquence</h2>');
    expect(html).toContain("enregistrez plus de 3 s");
    expect(html).toContain("Les 3 dernières secondes sont retirées.");
    expect(html).toContain('aria-label="Progression du guide, étape 2 sur 3"');
    expect(html).not.toContain("Lire dans Clash of Clans</strong>");
  });

  it("describes macro selection as the active first step for screen readers", () => {
    const html = renderHelp({
      onboardingComplete: false,
      selectedMacro: null,
      macros: [{ name: "Récolte", eventCount: 12, durationSeconds: 8, readable: true, issue: null }],
      settings: { shortcuts: null },
    } as unknown as Snapshot);

    expect(html).toContain("Étape active · 1 sur 3");
    expect(html).toContain('<h2 id="onboarding-title">Choisir une macro</h2>');
    expect(html).toContain('aria-label="Choisir une macro, étape actuelle"');
    expect(html).not.toContain('aria-label="Créer une macro, étape actuelle"');
  });

  it("advances the active instruction when a readable sequence is ready", () => {
    const html = renderHelp({
      onboardingComplete: false,
      selectedMacro: "Récolte",
      macros: [{ name: "Récolte", eventCount: 12, durationSeconds: 8, readable: true, issue: null }],
      settings: { shortcuts: null },
    } as unknown as Snapshot);

    expect(html).toContain("Étape active · 3 sur 3");
    expect(html).toContain('<h2 id="onboarding-title">Lire dans Clash of Clans</h2>');
    expect(html).toContain("La séquence est prête. Ouvrez Clash of Clans; le replay arrière-plan s’active dans Réglages &gt; Général.");
    expect(html).toContain('aria-label="Progression du guide, étape 3 sur 3"');
  });
});
