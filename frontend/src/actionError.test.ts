import { describe, expect, it } from "vitest";
import { reconcileBackendActionError } from "./actionError";

describe("reconcileBackendActionError", () => {
  it("clears a backend error when the backend clears it", () => {
    const error = "La lecture a échoué. Étape 3 : test.";

    expect(reconcileBackendActionError(error, error, null)).toBe("");
  });

  it("keeps another action error when an old backend error is cleared", () => {
    expect(reconcileBackendActionError("Échec de fermeture", "Ancienne erreur de lecture", null))
      .toBe("Échec de fermeture");
  });

  it("shows the newest backend error", () => {
    expect(reconcileBackendActionError("Ancienne erreur", "Ancienne erreur", "Nouvelle erreur"))
      .toBe("Nouvelle erreur");
  });
});
