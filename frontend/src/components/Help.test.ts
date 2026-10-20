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
  it("keeps frequent actions concise and puts safety details behind a disclosure", () => {
    const html = renderHelp();

    expect(html).toContain("Les gestes essentiels");
    expect(html).toContain("Une fois le compte à rebours terminé, laissez tourner l’enregistrement plus de 3 s.");
    expect(html).toContain("Clash of Clans doit être la fenêtre active");
    expect(html).toContain("Voir les détails de sûreté");
    expect(html).not.toMatch(/<details[^>]*open/);
  });

  it("does not render legacy migration or import language", () => {
    const html = renderHelp();

    expect(html).not.toMatch(/migration|importer vos données|ancienne installation/i);
  });

  it("shows either the unfinished onboarding checklist or the repeated quick guides", () => {
    const html = renderHelp({ onboardingComplete: false, settings: { shortcuts: null } } as unknown as Snapshot);

    expect(html).toContain("Les trois gestes pour démarrer");
    expect(html).toContain("class=\"onboarding-checklist\"");
    expect(html).not.toContain("id=\"first-steps-title\"");
    expect(html).not.toContain("Créer et enregistrer");
  });
});
