import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Dialog } from "./Dialog";

describe("Dialog description semantics", () => {
  it("associates a visible description with the dialog", () => {
    const html = renderToStaticMarkup(createElement(Dialog, {
      title: "Supprimer la macro ?",
      description: "Cette action est définitive.",
      onClose: () => {},
      children: createElement("button", { type: "button" }, "Annuler"),
    }));

    expect(html).toContain('aria-describedby="dialog-description"');
    expect(html).toContain('<p class="dialog-description" id="dialog-description">Cette action est définitive.</p>');
  });

  it("omits the description reference when there is no description", () => {
    const html = renderToStaticMarkup(createElement(Dialog, {
      title: "Action",
      onClose: () => {},
      children: createElement("button", { type: "button" }, "Fermer"),
    }));

    expect(html).not.toContain("aria-describedby");
    expect(html).not.toContain('class="dialog-description"');
  });
});
