import { describe, expect, it, vi } from "vitest";
import { ApiClient, type MigrationImportResult, type PairingPreparation, type Snapshot } from "./api";

function snapshotResponse(): Snapshot {
  return {
    revision: 1,
    status: { kind: "idle" },
    selectedMacro: null,
    macros: [],
    settings: {
      loop: false,
      cocPath: "",
      shortcuts: { toggle: "F1", play: "Ctrl+Shift+1", stop: "Ctrl+Shift+0" },
      telegram: { tokenConfigured: false, paired: false, status: "not_configured", pairingCode: null, pairingExpiresAt: null },
    },
    session: { elapsedSeconds: 0, cycles: 0 },
    migration: { sourceSelected: false, available: false, alreadyImported: false },
    onboardingComplete: false,
  };
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } });
}

describe("ApiClient", () => {
  it("calls the default fetch transport with the global receiver", async () => {
    const fetcher = vi.fn(function (this: typeof globalThis) {
      expect(this).toBe(globalThis);
      return Promise.resolve(jsonResponse({ token: "session-only" }));
    });
    vi.stubGlobal("fetch", fetcher);

    try {
      const api = new ApiClient();
      await expect(api.openSession()).resolves.toBeUndefined();
      expect(fetcher).toHaveBeenCalledOnce();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("keeps the session token in memory and encodes macro names on requests", async () => {
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse({ name: "Récolte / nuit", steps: [] }));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await api.getMacro("Récolte / nuit");

    expect(fetcher.mock.calls[1][0]).toBe("/api/macros/R%C3%A9colte%20%2F%20nuit");
    expect(new Headers(fetcher.mock.calls[1][1]?.headers).get("X-Auto-Coc-Session")).toBe("session-only");
    expect(fetcher.mock.calls[1][1]?.credentials).toBe("same-origin");
  });

  it("sends JSON for local actions and returns the authoritative snapshot", async () => {
    const updated = snapshotResponse();
    updated.revision = 2;
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse({ snapshot: updated }));
    const api = new ApiClient(fetcher);
    await api.openSession();

    const result = await api.startRecording();

    expect(result.revision).toBe(2);
    expect(fetcher.mock.calls[1][1]?.method).toBe("POST");
    expect(new Headers(fetcher.mock.calls[1][1]?.headers).get("Content-Type")).toBe("application/json");
    expect(fetcher.mock.calls[1][1]?.body).toBe("{}");
  });

  it("does not issue protected requests before session bootstrap", async () => {
    const fetcher = vi.fn<typeof fetch>();
    const api = new ApiClient(fetcher);

    await expect(api.getSnapshot()).rejects.toMatchObject({ code: "session_missing" });
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("reports non-JSON server responses as a local service error", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response("<html>not the API</html>", { status: 404 }));
    const api = new ApiClient(fetcher);

    await expect(api.openSession()).rejects.toMatchObject({ code: "invalid_response", status: 404 });
  });

  it("preserves structured backend errors", async () => {
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse({ error: { code: "macro_not_found", message: "Macro introuvable." } }, 404));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await expect(api.selectMacro("absente")).rejects.toEqual(expect.objectContaining({
      message: "Macro introuvable.",
      status: 404,
      code: "macro_not_found",
    }));
  });

  it("requests screenshots as local PNG blobs without sending them elsewhere", async () => {
    const png = new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" });
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(new Response(png, { status: 200, headers: { "Content-Type": "image/png" } }));
    const api = new ApiClient(fetcher);
    await api.openSession();

    const result = await api.getScreenshot();

    expect(result.type).toBe("image/png");
    expect(fetcher.mock.calls[1][0]).toBe("/api/screenshot");
    expect(fetcher.mock.calls[1][1]?.method).toBe("POST");
    expect(fetcher.mock.calls[1][1]?.body).toBe("{}");
    expect(new Headers(fetcher.mock.calls[1][1]?.headers).get("Content-Type")).toBe("application/json");
  });

  it("uses the persisted loop setting through an empty playback command", async () => {
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse({ snapshot: snapshotResponse() }));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await api.startPlayback();

    expect(fetcher.mock.calls[1][0]).toBe("/api/playback/start");
    expect(fetcher.mock.calls[1][1]?.body).toBe("{}");
  });

  it("treats a cancelled native migration picker as a normal cancellation", async () => {
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse({ error: { code: "migration_cancelled", message: "Sélection annulée." } }, 409));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await expect(api.selectMigrationSource()).resolves.toBeNull();
    expect(fetcher.mock.calls[1][0]).toBe("/api/migration/select-source");
    expect(fetcher.mock.calls[1][1]?.body).toBe("{}");
  });

  it("reads migration availability from the preview DTO", async () => {
    const migration = {
      status: { sourceSelected: true, available: true, alreadyImported: false },
      preview: { available: true, settingsFound: false, macros: [{ name: "Récolte", eventCount: 4, readable: true, issue: null }] },
    };
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse(migration));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await expect(api.getMigration()).resolves.toEqual(migration);
    expect(fetcher.mock.calls[1][0]).toBe("/api/migration");
  });

  it("reads the selected migration preview including availability", async () => {
    const selection = { snapshot: snapshotResponse(), preview: { available: true, settingsFound: true, macros: [] } };
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse(selection));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await expect(api.selectMigrationSource()).resolves.toEqual(selection);
    expect(fetcher.mock.calls[1][0]).toBe("/api/migration/select-source");
  });

  it("imports only through the dedicated route and returns the backend report", async () => {
    const report: MigrationImportResult = { snapshot: snapshotResponse(), imported: { settings: 1, macros: 2 }, collisions: [], preserved: [], errors: [] };
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse(report));
    const api = new ApiClient(fetcher);
    await api.openSession();

    const result = await api.importMigration();
    expect(result).toEqual(report);
    expect(result).not.toHaveProperty("fingerprint");
    expect(fetcher.mock.calls[1][0]).toBe("/api/migration/import");
    expect(fetcher.mock.calls[1][1]?.body).toBe("{}");
  });

  it("starts Telegram pairing with an empty JSON request and reads the expiring code", async () => {
    const pairing: PairingPreparation = { code: "TEST-PAIR", expiresAt: "2026-09-30T12:00:00Z", snapshot: snapshotResponse() };
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse(pairing));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await expect(api.startTelegramPairing()).resolves.toEqual(pairing);
    expect(fetcher.mock.calls[1][0]).toBe("/api/telegram/pairing/start");
    expect(fetcher.mock.calls[1][1]?.body).toBe("{}");
  });

  it("uses PUT to save a Telegram token and DELETE with an empty body to clear it", async () => {
    const updated = snapshotResponse();
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse({ snapshot: updated }))
      .mockResolvedValueOnce(jsonResponse({ snapshot: updated }));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await api.saveTelegramToken("test-token-only");
    await api.clearTelegramToken();

    expect(fetcher.mock.calls[1][1]?.method).toBe("PUT");
    expect(fetcher.mock.calls[1][1]?.body).toBe(JSON.stringify({ token: "test-token-only" }));
    expect(fetcher.mock.calls[2][0]).toBe("/api/telegram/config");
    expect(fetcher.mock.calls[2][1]?.method).toBe("DELETE");
    expect(fetcher.mock.calls[2][1]?.body).toBe("{}");
  });

  it("prepares shutdown and quits only through their dedicated local routes", async () => {
    const preparation = { confirmationId: "test-confirmation", expiresAt: "2026-09-30T12:00:00Z" };
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse(preparation))
      .mockResolvedValueOnce(jsonResponse({ snapshot: snapshotResponse() }));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await expect(api.prepareShutdown()).resolves.toEqual(preparation);
    await api.quitApplication();

    expect(fetcher.mock.calls[1][0]).toBe("/api/system/shutdown/prepare");
    expect(fetcher.mock.calls[1][1]?.body).toBe("{}");
    expect(fetcher.mock.calls[2][0]).toBe("/api/application/quit");
    expect(fetcher.mock.calls[2][1]?.body).toBe("{}");
  });

  it("marks onboarding complete from the backend snapshot", async () => {
    const updated = snapshotResponse();
    updated.onboardingComplete = true;
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ token: "session-only" }))
      .mockResolvedValueOnce(jsonResponse({ snapshot: updated }));
    const api = new ApiClient(fetcher);
    await api.openSession();

    await expect(api.completeOnboarding()).resolves.toMatchObject({ onboardingComplete: true });
    expect(fetcher.mock.calls[1][0]).toBe("/api/onboarding/complete");
    expect(fetcher.mock.calls[1][1]?.method).toBe("POST");
    expect(fetcher.mock.calls[1][1]?.body).toBe("{}");
  });
});
