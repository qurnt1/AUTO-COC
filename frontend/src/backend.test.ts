import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import App from "./App";
import { BackendClientError } from "./api";
import { createBackendClient } from "./backend";
import { TauriBackendClient } from "./tauriBackend";

const runtime = vi.hoisted(() => ({ isTauri: vi.fn() }));

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: runtime.isTauri,
  invoke: vi.fn(),
}));

describe("createBackendClient", () => {
  beforeEach(() => runtime.isTauri.mockReset());

  it("creates the Tauri client in the desktop WebView", () => {
    runtime.isTauri.mockReturnValue(true);

    expect(createBackendClient()).toBeInstanceOf(TauriBackendClient);
  });

  it("fails explicitly outside the desktop WebView", () => {
    runtime.isTauri.mockReturnValue(false);

    expect(createBackendClient).toThrow(expect.objectContaining({
      name: "BackendClientError",
      code: "unsupported_runtime",
    } satisfies Partial<BackendClientError>));
  });

  it("shows the desktop launch instruction outside the desktop WebView", () => {
    runtime.isTauri.mockReturnValue(false);
    const markup = renderToStaticMarkup(createElement(App));

    expect(markup).toContain("AUTO-COC est une application de bureau");
    expect(markup).toContain("cargo tauri dev");
  });
});
