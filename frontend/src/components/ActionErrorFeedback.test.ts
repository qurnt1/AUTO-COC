import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { ActionErrorFeedback } from "./ActionErrorFeedback";

describe("ActionErrorFeedback", () => {
  it("exposes playback failures as visible alert text with a dismiss control", () => {
    const markup = renderToStaticMarkup(
      createElement(ActionErrorFeedback, {
        message: "La lecture a été interrompue par une erreur de saisie Windows.",
        onDismiss: vi.fn(),
      }),
    );

    expect(markup).toContain('role="alert"');
    expect(markup).toContain("La lecture a été interrompue par une erreur de saisie Windows.");
    expect(markup).toContain('aria-label="Fermer le message"');
  });
});
