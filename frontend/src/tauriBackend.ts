import { invoke } from "@tauri-apps/api/core";
import { listen, type Event, type UnlistenFn } from "@tauri-apps/api/event";
import {
  BackendClientError,
  type BackendClient,
  type Diagnostics,
  type HelpDocument,
  type MacroDetail,
  type MigrationImportResult,
  type MigrationSelectionResult,
  type MigrationState,
  type PairingPreparation,
  type ShutdownPreparation,
  type Snapshot,
} from "./api";

type SnapshotWaiter = {
  after: number;
  afterEventSequence: number;
  resolve: (snapshot: Snapshot) => void;
  reject: (error: Error) => void;
  signal?: AbortSignal;
  onAbort?: () => void;
};

export type TauriBridge = {
  invoke: <T>(command: string, args?: Record<string, unknown>) => Promise<T>;
  listen: <T>(event: string, handler: (event: Event<T>) => void) => Promise<UnlistenFn>;
};

const defaultBridge: TauriBridge = {
  invoke: <T>(command: string, args?: Record<string, unknown>) => invoke<T>(command, args),
  listen: <T>(event: string, handler: (event: Event<T>) => void) => listen<T>(event, handler),
};

function abortError(): DOMException {
  return new DOMException("The operation was aborted.", "AbortError");
}

function withSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortError());

  return new Promise((resolve, reject) => {
    const onAbort = () => reject(abortError());
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => { signal.removeEventListener("abort", onAbort); resolve(value); },
      (error: unknown) => { signal.removeEventListener("abort", onAbort); reject(error); },
    );
  });
}

function isSnapshot(value: unknown): value is Snapshot {
  return typeof value === "object"
    && value !== null
    && "revision" in value
    && typeof value.revision === "number"
    && "macros" in value
    && Array.isArray(value.macros);
}

function errorFields(value: unknown): { message?: string; code?: string; status?: number } {
  if (typeof value === "string") {
    try {
      return errorFields(JSON.parse(value) as unknown);
    } catch {
      return { message: value };
    }
  }
  if (typeof value !== "object" || value === null) return {};

  const record = value as Record<string, unknown>;
  const nested = typeof record.error === "object" && record.error !== null
    ? record.error as Record<string, unknown>
    : record;
  return {
    ...(typeof nested.message === "string" ? { message: nested.message } : {}),
    ...(typeof nested.code === "string" ? { code: nested.code } : {}),
    ...(typeof nested.status === "number" ? { status: nested.status } : {}),
  };
}

function backendError(error: unknown): BackendClientError {
  if (error instanceof BackendClientError) return error;
  const fields = errorFields(error);
  return new BackendClientError(fields.message ?? "L’opération locale n’a pas abouti.", fields.status, fields.code);
}

export class TauriBackendClient implements BackendClient {
  private latest: Snapshot | undefined;
  private unlisten: UnlistenFn | undefined;
  private listenerPromise: Promise<void> | undefined;
  private readonly waiters = new Set<SnapshotWaiter>();
  private readonly shutdownErrorListeners = new Set<() => void>();
  private lifecycle = 0;
  private eventSequence = 0;

  constructor(private readonly bridge: TauriBridge = defaultBridge) {}

  async connect(signal?: AbortSignal): Promise<Snapshot> {
    if (signal?.aborted) throw abortError();
    await withSignal(this.ensureListener(), signal);
    const initial = await this.invoke<Snapshot>("get_snapshot", undefined, signal);
    if (!isSnapshot(initial)) throw new BackendClientError("Le moteur Rust intégré a renvoyé un état illisible.", undefined, "invalid_response");
    this.publish(initial, false);
    return this.latest ?? initial;
  }

  async getSnapshot(signal?: AbortSignal): Promise<Snapshot> {
    const snapshot = await this.invoke<Snapshot>("get_snapshot", undefined, signal);
    if (!isSnapshot(snapshot)) throw new BackendClientError("Le moteur Rust intégré a renvoyé un état illisible.", undefined, "invalid_response");
    this.publish(snapshot, false);
    return this.latest ?? snapshot;
  }

  async waitForSnapshot(after: number, signal?: AbortSignal): Promise<Snapshot> {
    if (signal?.aborted) throw abortError();
    const afterEventSequence = this.eventSequence;
    if (this.latest && this.latest.revision > after) return this.latest;
    await withSignal(this.ensureListener(), signal);
    if (signal?.aborted) throw abortError();
    if (this.latest && this.latest.revision > after) return this.latest;
    if (this.eventSequence > afterEventSequence && this.latest && this.latest.revision >= after) return this.latest;

    return new Promise<Snapshot>((resolve, reject) => {
      const waiter: SnapshotWaiter = { after, afterEventSequence, resolve, reject, signal };
      if (signal) {
        waiter.onAbort = () => {
          this.removeWaiter(waiter);
          reject(abortError());
        };
        signal.addEventListener("abort", waiter.onAbort, { once: true });
      }
      this.waiters.add(waiter);
    });
  }

  getMacro(name: string, signal?: AbortSignal): Promise<MacroDetail> {
    return this.invoke("get_macro", { name }, signal);
  }

  createMacro(name: string): Promise<Snapshot> {
    return this.snapshotCommand("create_macro", { name });
  }

  renameMacro(name: string, newName: string): Promise<Snapshot> {
    return this.snapshotCommand("rename_macro", { name, newName });
  }

  deleteMacro(name: string): Promise<Snapshot> {
    return this.snapshotCommand("delete_macro", { name });
  }

  selectMacro(name: string): Promise<Snapshot> {
    return this.snapshotCommand("select_macro", { name });
  }

  startRecording(): Promise<Snapshot> {
    return this.snapshotCommand("start_recording");
  }

  stopRecording(): Promise<Snapshot> {
    return this.snapshotCommand("stop_recording");
  }

  startPlayback(): Promise<Snapshot> {
    return this.snapshotCommand("start_playback");
  }

  cancelResumeWait(): Promise<Snapshot> {
    return this.snapshotCommand("cancel_resume_wait");
  }

  stopPlayback(): Promise<Snapshot> {
    return this.snapshotCommand("stop_playback");
  }

  updateSettings(settings: { loop?: boolean; cocPath?: string; requireCocForeground?: boolean }): Promise<Snapshot> {
    return this.snapshotCommand("patch_settings", { settings });
  }

  updateShortcuts(shortcuts: Snapshot["settings"]["shortcuts"]): Promise<Snapshot> {
    return this.snapshotCommand("put_shortcuts", { shortcuts });
  }

  launchCoc(): Promise<Snapshot> {
    return this.snapshotCommand("launch_coc");
  }

  saveTelegramToken(token: string): Promise<Snapshot> {
    return this.snapshotCommand("telegram_config", { token });
  }

  clearTelegramToken(): Promise<Snapshot> {
    return this.snapshotCommand("telegram_config_clear");
  }

  async startTelegramPairing(): Promise<PairingPreparation> {
    const result = await this.invoke<PairingPreparation>("pairing_start");
    this.publish(result.snapshot, false);
    return result;
  }

  cancelTelegramPairing(): Promise<Snapshot> {
    return this.snapshotCommand("pairing_cancel");
  }

  testTelegram(): Promise<Snapshot> {
    return this.snapshotCommand("telegram_test");
  }

  openMacrosFolder(): Promise<Snapshot> {
    return this.snapshotCommand("open_macros_folder");
  }

  quitApplication(): Promise<void> {
    return this.invoke("quit_application");
  }

  getMigration(signal?: AbortSignal): Promise<MigrationState> {
    return this.invoke("get_migration", undefined, signal);
  }

  async selectMigrationSource(): Promise<MigrationSelectionResult | null> {
    try {
      const selection = await this.invoke<MigrationSelectionResult | null>("select_migration_source");
      if (selection) this.publish(selection.snapshot, false);
      return selection;
    } catch (error) {
      const failure = backendError(error);
      if (failure.code === "migration_cancelled") return null;
      throw failure;
    }
  }

  async importMigration(): Promise<MigrationImportResult> {
    const result = await this.invoke<MigrationImportResult>("import_migration");
    this.publish(result.snapshot, false);
    return result;
  }

  prepareShutdown(): Promise<ShutdownPreparation> {
    return this.invoke("shutdown_prepare");
  }

  confirmShutdown(confirmationId: string): Promise<Snapshot> {
    return this.snapshotCommand("shutdown_confirm", { confirmationId });
  }

  async getScreenshot(signal?: AbortSignal): Promise<Blob> {
    const payload = await this.invoke<number[] | Uint8Array | ArrayBuffer>("screenshot", undefined, signal);
    if (!Array.isArray(payload) && !(payload instanceof Uint8Array) && !(payload instanceof ArrayBuffer)) {
      throw new BackendClientError("Le moteur Rust intégré n’a pas renvoyé une image PNG.", undefined, "invalid_response");
    }
    const bytes = payload instanceof ArrayBuffer ? new Uint8Array(payload) : Uint8Array.from(payload);
    return new Blob([bytes], { type: "image/png" });
  }

  getHelp(signal?: AbortSignal): Promise<HelpDocument> {
    return this.invoke("get_help", undefined, signal);
  }

  getDiagnostics(signal?: AbortSignal): Promise<Diagnostics> {
    return this.invoke("get_diagnostics", undefined, signal);
  }

  completeOnboarding(): Promise<Snapshot> {
    return this.snapshotCommand("complete_onboarding");
  }

  listenShutdownErrors(handler: (error: BackendClientError) => void): () => void {
    let active = true;
    let unlisten: UnlistenFn | undefined;
    const stop = () => {
      active = false;
      unlisten?.();
      unlisten = undefined;
      this.shutdownErrorListeners.delete(stop);
    };
    this.shutdownErrorListeners.add(stop);
    void this.bridge.listen<unknown>("shutdown-error", (event) => {
      if (active) handler(backendError(event.payload));
    }).then((registered) => {
      if (active) unlisten = registered;
      else registered();
    }).catch((error: unknown) => {
      if (active) handler(backendError(error));
    });
    return stop;
  }

  dispose(): void {
    for (const stop of this.shutdownErrorListeners) stop();
    this.stopListening();
    for (const waiter of this.waiters) {
      this.removeWaiter(waiter);
      waiter.reject(abortError());
    }
    this.latest = undefined;
    this.eventSequence = 0;
  }

  private async ensureListener(): Promise<void> {
    const lifecycle = this.lifecycle;
    if (!this.unlisten && !this.listenerPromise) {
      const registration = this.bridge.listen<Snapshot>("snapshot-updated", (event) => {
        if (this.lifecycle === lifecycle) this.publish(event.payload, true);
      }).then((unlisten) => {
        if (this.lifecycle !== lifecycle) {
          unlisten();
          return;
        }
        this.unlisten = unlisten;
      }).catch((error: unknown) => {
        throw backendError(error);
      });
      this.listenerPromise = registration;
      void registration.then(
        () => { if (this.listenerPromise === registration) this.listenerPromise = undefined; },
        () => { if (this.listenerPromise === registration) this.listenerPromise = undefined; },
      );
    }
    const registration = this.listenerPromise;
    if (registration) await registration;
    if (this.lifecycle !== lifecycle || !this.unlisten) throw abortError();
  }

  private publish(snapshot: unknown, fromEvent: boolean): void {
    if (!isSnapshot(snapshot) || (this.latest && snapshot.revision < this.latest.revision)) return;
    if (!fromEvent && this.latest && snapshot.revision === this.latest.revision) return;
    if (fromEvent) this.eventSequence += 1;
    this.latest = snapshot;
    for (const waiter of this.waiters) {
      if (snapshot.revision > waiter.after
        || (fromEvent && this.eventSequence > waiter.afterEventSequence && snapshot.revision >= waiter.after)) {
        this.removeWaiter(waiter);
        waiter.resolve(snapshot);
      }
    }
  }

  private removeWaiter(waiter: SnapshotWaiter): void {
    this.waiters.delete(waiter);
    if (waiter.signal && waiter.onAbort) waiter.signal.removeEventListener("abort", waiter.onAbort);
  }

  private stopListening(): void {
    this.lifecycle += 1;
    this.listenerPromise = undefined;
    this.unlisten?.();
    this.unlisten = undefined;
  }

  private invoke<T>(command: string, args?: Record<string, unknown>, signal?: AbortSignal): Promise<T> {
    return withSignal(this.bridge.invoke<T>(command, args).catch((error: unknown) => { throw backendError(error); }), signal);
  }

  private async snapshotCommand(command: string, args?: Record<string, unknown>): Promise<Snapshot> {
    const snapshot = await this.invoke<Snapshot>(command, args);
    if (!isSnapshot(snapshot)) throw new BackendClientError("Le moteur Rust intégré a renvoyé un état illisible.", undefined, "invalid_response");
    this.publish(snapshot, false);
    return this.latest ?? snapshot;
  }
}
