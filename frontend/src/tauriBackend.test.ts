import type { Event } from "@tauri-apps/api/event";
import { describe, expect, it, vi } from "vitest";
import type { Snapshot } from "./api";
import { TauriBackendClient, type TauriBridge } from "./tauriBackend";

function snapshot(revision: number, elapsedSeconds = 0): Snapshot {
  return {
    revision,
    status: { kind: "idle" },
    selectedMacro: null,
    macros: [],
    settings: {
      loop: false,
      cocPath: "",
      shortcuts: { toggle: "F1", play: "Ctrl+Shift+1", stop: "Ctrl+Shift+0" },
      telegram: { tokenConfigured: false, paired: false, status: "not_configured", pairingCode: null, pairingExpiresAt: null },
    },
    session: { elapsedSeconds, cycles: 0 },
    migration: { sourceSelected: false, available: false, alreadyImported: false },
    onboardingComplete: false,
  };
}

function makeBridge(respond: (command: string, args?: Record<string, unknown>) => unknown = () => snapshot(1)) {
  const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
  const handlers = new Map<string, (event: Event<unknown>) => void>();
  const unlisten = vi.fn();
  const bridge: TauriBridge = {
    invoke: async <T>(command: string, args?: Record<string, unknown>) => {
      calls.push({ command, args });
      return respond(command, args) as T;
    },
    listen: async <T>(event: string, next: (event: Event<T>) => void) => {
      calls.push({ command: `listen:${event}` });
      handlers.set(event, next as (event: Event<unknown>) => void);
      return () => { handlers.delete(event); unlisten(); };
    },
  };
  return {
    bridge,
    calls,
    unlisten,
    emit(value: unknown, event = "snapshot-updated") { handlers.get(event)?.({ payload: value } as Event<unknown>); },
  };
}

describe("TauriBackendClient", () => {
  it("listens before the initial snapshot and keeps the newest event revision", async () => {
    const fake = makeBridge((command) => {
      if (command === "get_snapshot") {
        fake.emit(snapshot(2));
        return snapshot(1);
      }
      return snapshot(1);
    });
    const client = new TauriBackendClient(fake.bridge);

    await expect(client.connect()).resolves.toMatchObject({ revision: 2 });
    expect(fake.calls.slice(0, 2).map((call) => call.command)).toEqual(["listen:snapshot-updated", "get_snapshot"]);

    const sameRevision = client.waitForSnapshot(2);
    fake.emit(snapshot(2));
    await expect(sameRevision).resolves.toMatchObject({ revision: 2 });

    const updated = client.waitForSnapshot(2);
    fake.emit(snapshot(3));
    await expect(updated).resolves.toMatchObject({ revision: 3 });

    client.dispose();
    expect(fake.unlisten).toHaveBeenCalledOnce();
  });

  it("delivers newer activity snapshots that share a revision", async () => {
    const fake = makeBridge((command) => command === "get_snapshot" ? snapshot(5) : snapshot(5));
    const client = new TauriBackendClient(fake.bridge);
    await client.connect();

    const firstTick = client.waitForSnapshot(5);
    fake.emit(snapshot(5, 1));
    await expect(firstTick).resolves.toMatchObject({ revision: 5, session: { elapsedSeconds: 1 } });

    let secondTickResolved = false;
    const secondTick = client.waitForSnapshot(5).then((value) => {
      secondTickResolved = true;
      return value;
    });
    await Promise.resolve();
    expect(secondTickResolved).toBe(false);

    fake.emit(snapshot(5, 2));
    await expect(secondTick).resolves.toMatchObject({ revision: 5, session: { elapsedSeconds: 2 } });
    client.dispose();
  });

  it("ignores stale snapshot events while allowing the current revision to advance", async () => {
    const fake = makeBridge((command) => command === "get_snapshot" ? snapshot(5, 2) : snapshot(5));
    const client = new TauriBackendClient(fake.bridge);
    await client.connect();

    let updateResolved = false;
    const update = client.waitForSnapshot(5).then((value) => {
      updateResolved = true;
      return value;
    });
    fake.emit(snapshot(4, 99));
    await Promise.resolve();
    expect(updateResolved).toBe(false);

    fake.emit(snapshot(5, 3));
    await expect(update).resolves.toMatchObject({ revision: 5, session: { elapsedSeconds: 3 } });
    await expect(client.getSnapshot()).resolves.toMatchObject({ revision: 5, session: { elapsedSeconds: 3 } });
    client.dispose();
  });

  it("maps frontend operations to the agreed command names and DTO arguments", async () => {
    const fake = makeBridge((command) => {
      if (command === "screenshot") return Uint8Array.from([137, 80, 78, 71]).buffer;
      if (command === "select_migration_source") return null;
      if (command === "quit_application") return undefined;
      return snapshot(1);
    });
    const client = new TauriBackendClient(fake.bridge);

    await client.getMacro("Récolte / nuit");
    await client.createMacro("Récolte");
    await client.renameMacro("Ancien", "Nouveau");
    await client.deleteMacro("Nouveau");
    await client.selectMacro("Nouveau");
    await client.startRecording();
    await client.stopRecording();
    await client.startPlayback();
    await client.stopPlayback();
    await client.updateSettings({ loop: true, cocPath: "C:/CoC.exe" });
    await client.updateShortcuts({ toggle: "F2", play: "Ctrl+1", stop: "Ctrl+2" });
    await client.launchCoc();
    await client.saveTelegramToken("secret-test-token");
    await client.clearTelegramToken();
    await client.startTelegramPairing();
    await client.cancelTelegramPairing();
    await client.testTelegram();
    await client.openMacrosFolder();
    await client.quitApplication();
    await client.getMigration();
    await expect(client.selectMigrationSource()).resolves.toBeNull();
    await client.importMigration();
    await client.prepareShutdown();
    await client.confirmShutdown("confirm-1");
    await client.getHelp();
    await client.getDiagnostics();
    await client.completeOnboarding();
    const screenshot = await client.getScreenshot();

    expect(fake.calls.map((call) => call.command)).toEqual([
      "get_macro", "create_macro", "rename_macro", "delete_macro", "select_macro",
      "start_recording", "stop_recording", "start_playback", "stop_playback",
      "patch_settings", "put_shortcuts", "launch_coc", "telegram_config",
      "telegram_config_clear", "pairing_start", "pairing_cancel", "telegram_test",
      "open_macros_folder", "quit_application", "get_migration", "select_migration_source",
      "import_migration", "shutdown_prepare", "shutdown_confirm", "get_help",
      "get_diagnostics", "complete_onboarding", "screenshot",
    ]);
    expect(fake.calls[9].args).toEqual({ settings: { loop: true, cocPath: "C:/CoC.exe" } });
    expect(fake.calls[10].args).toEqual({ shortcuts: { toggle: "F2", play: "Ctrl+1", stop: "Ctrl+2" } });
    expect(fake.calls[23].args).toEqual({ confirmationId: "confirm-1" });
    expect(screenshot.type).toBe("image/png");
    expect([...new Uint8Array(await screenshot.arrayBuffer())]).toEqual([137, 80, 78, 71]);
  });

  it("accepts an IPC Uint8Array screenshot response", async () => {
    const fake = makeBridge((command) => command === "screenshot" ? new Uint8Array([1, 2, 3]) : snapshot(1));
    const client = new TauriBackendClient(fake.bridge);

    const screenshot = await client.getScreenshot();

    expect([...new Uint8Array(await screenshot.arrayBuffer())]).toEqual([1, 2, 3]);
  });

  it("forwards shutdown failures as retryable runtime errors and cleans up", async () => {
    const fake = makeBridge();
    const client = new TauriBackendClient(fake.bridge);
    const onError = vi.fn();
    const stop = client.listenShutdownErrors(onError);
    await Promise.resolve();

    fake.emit({ code: "shutdown_failed", message: "La sauvegarde n’a pas abouti." }, "shutdown-error");

    expect(onError).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({
      code: "shutdown_failed",
      message: "La sauvegarde n’a pas abouti.",
    }));
    stop();
    expect(fake.unlisten).toHaveBeenCalledOnce();
  });

  it("uses mutation snapshots to reconcile state if an event arrives late", async () => {
    const fake = makeBridge((command) => command === "start_recording" ? snapshot(2) : snapshot(1));
    const client = new TauriBackendClient(fake.bridge);
    await client.connect();

    await client.startRecording();

    await expect(client.waitForSnapshot(1)).resolves.toMatchObject({ revision: 2 });
    client.dispose();
  });

  it("preserves backend error codes and releases an aborted snapshot wait", async () => {
    const fake = makeBridge((command) => {
      if (command === "create_macro") throw { code: "conflict", message: "Une macro de ce nom existe déjà." };
      return snapshot(1);
    });
    const client = new TauriBackendClient(fake.bridge);
    await client.connect();

    await expect(client.createMacro("Déjà là")).rejects.toMatchObject({
      code: "conflict",
      message: "Une macro de ce nom existe déjà.",
    });

    const controller = new AbortController();
    const waiting = client.waitForSnapshot(1, controller.signal);
    controller.abort();
    await expect(waiting).rejects.toMatchObject({ name: "AbortError" });
    client.dispose();
  });
});
