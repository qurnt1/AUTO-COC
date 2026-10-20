import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ActionRunner, BackendClient, Snapshot } from "../api";
import { Help } from "./Help";

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
    expect(html.match(/Ouvrir l’atelier<svg/g)).toHaveLength(2);
    expect(html).toContain("Ouvrir Telegram<svg");
    expect(html).toContain('aria-label="Ouvrir l’atelier : Créer une macro"');
    expect(html).toContain('aria-label="Ouvrir l’atelier : Lire une macro"');
    expect(html).toContain("enregistrez plus de 3 s");
    expect(html).toContain("Les 3 dernières secondes sont retirées.");
    expect(html).toContain("Clash of Clans doit être au premier plan.");
    expect(html).toContain("Voir les détails de sûreté");
    expect(html).not.toMatch(/<details[^>]*open/);
  });

  it("does not render legacy migration or import language", () => {
    const html = renderHelp();

    expect(html).not.toMatch(/migration|importer vos données|ancienne installation/i);
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
    expect(html).toContain("La séquence est prête. Clash of Clans doit être la fenêtre active pour la lire.");
    expect(html).toContain('aria-label="Progression du guide, étape 3 sur 3"');
  });
});
