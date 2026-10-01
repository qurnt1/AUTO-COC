export type MacroSummary = {
  name: string;
  eventCount: number;
  durationSeconds: number;
  protected: boolean;
  readable: boolean;
  issue: string | null;
};

export type MacroStep = {
  t: number;
  type: string;
  data: unknown;
};

export type MacroDetail = {
  name: string;
  steps: MacroStep[];
};

export type ActionRunner = <T>(label: string, action: () => Promise<T>, message?: string) => Promise<T | undefined>;

export type AppStatus =
  | { kind: "idle" }
  | { kind: "recording"; macroName: string; elapsedSeconds: number; phase: "preparing" | "capturing"; countdownSeconds: number }
  | { kind: "playing"; macroName: string; elapsedSeconds: number };

export type TelegramStatus = "not_configured" | "disconnected" | "waiting_pairing" | "connected" | "error";

export type MigrationMacro = { name: string; eventCount: number; readable: boolean; issue: string | null };
export type MigrationPreview = { available: boolean; settingsFound: boolean; macros: MigrationMacro[] };
export type MigrationState = {
  status: { sourceSelected: boolean; available: boolean; alreadyImported: boolean };
  preview: MigrationPreview | null;
};
export type MigrationSelectionResult = { snapshot: Snapshot; preview: MigrationPreview };
export type MigrationImportResult = {
  snapshot: Snapshot;
  imported: { settings: number; macros: number };
  collisions: string[];
  preserved: string[];
  errors: string[];
};

export type Snapshot = {
  revision: number;
  status: AppStatus;
  selectedMacro: string | null;
  macros: MacroSummary[];
  settings: {
    loop: boolean;
    cocPath: string;
    shortcuts: { toggle: string; play: string; stop: string };
    telegram: {
      tokenConfigured: boolean;
      paired: boolean;
      status: TelegramStatus;
      pairingCode: string | null;
      pairingExpiresAt: string | null;
    };
  };
  session: { elapsedSeconds: number; cycles: number };
  migration: MigrationState["status"];
  onboardingComplete: boolean;
};

export type Diagnostics = {
  appVersion: string;
  rustVersion: string;
  status: string;
  errors: string[];
};

export type HelpSection = { title: string; body: string };
export type HelpDocument = { sections: HelpSection[] };

type ApiError = { error?: { code?: string; message?: string } };
type SnapshotResponse = { snapshot: Snapshot };
export type ShutdownPreparation = { confirmationId: string; expiresAt: string };
export type PairingPreparation = { code: string; expiresAt: string; snapshot: Snapshot };

async function readJson<T>(response: Response): Promise<T> {
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiRequestError("Le service local a renvoyé une réponse illisible.", response.status, "invalid_response");
  }
}

export class ApiRequestError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
  }
}

export class ApiClient {
  private token: string | undefined;

  constructor(private readonly fetcher: typeof fetch = globalThis.fetch.bind(globalThis)) {}

  async openSession(signal?: AbortSignal): Promise<void> {
    const response = await this.fetcher("/api/session", {
      method: "GET",
      credentials: "same-origin",
      cache: "no-store",
      signal,
    });
    const payload = await readJson<{ token?: unknown }>(response);
    if (!response.ok || typeof payload.token !== "string" || payload.token.length === 0) {
      throw new ApiRequestError("Impossible d’ouvrir la session locale.", response.status);
    }
    this.token = payload.token;
  }

  getSnapshot(signal?: AbortSignal): Promise<Snapshot> {
    return this.request<Snapshot>("/api/snapshot", { signal });
  }

  getEvents(after: number, signal?: AbortSignal): Promise<Snapshot> {
    return this.request<Snapshot>(`/api/events?after=${encodeURIComponent(after)}`, { signal });
  }

  getMacro(name: string, signal?: AbortSignal): Promise<MacroDetail> {
    return this.request<MacroDetail>(`/api/macros/${encodeURIComponent(name)}`, { signal });
  }

  async createMacro(name: string): Promise<Snapshot> {
    return this.mutate("POST", "/api/macros", { name });
  }

  async renameMacro(name: string, newName: string): Promise<Snapshot> {
    return this.mutate("PATCH", `/api/macros/${encodeURIComponent(name)}`, { newName });
  }

  async deleteMacro(name: string): Promise<Snapshot> {
    return this.mutate("DELETE", `/api/macros/${encodeURIComponent(name)}`, {});
  }

  async selectMacro(name: string): Promise<Snapshot> {
    return this.mutate("POST", "/api/selection", { name });
  }

  async startRecording(): Promise<Snapshot> {
    return this.mutate("POST", "/api/recording/start", {});
  }

  async stopRecording(): Promise<Snapshot> {
    return this.mutate("POST", "/api/recording/stop", {});
  }

  async startPlayback(): Promise<Snapshot> {
    return this.mutate("POST", "/api/playback/start", {});
  }

  async stopPlayback(): Promise<Snapshot> {
    return this.mutate("POST", "/api/playback/stop", {});
  }

  async updateSettings(settings: { loop?: boolean; cocPath?: string }): Promise<Snapshot> {
    return this.mutate("PATCH", "/api/settings", settings);
  }

  async updateShortcuts(shortcuts: Snapshot["settings"]["shortcuts"]): Promise<Snapshot> {
    return this.mutate("PUT", "/api/shortcuts", shortcuts);
  }

  async launchCoc(): Promise<Snapshot> {
    return this.mutate("POST", "/api/launch-coc", {});
  }

  async saveTelegramToken(token: string): Promise<Snapshot> {
    return this.mutate("PUT", "/api/telegram/config", { token });
  }

  async clearTelegramToken(): Promise<Snapshot> {
    return this.mutate("DELETE", "/api/telegram/config", {});
  }

  startTelegramPairing(): Promise<PairingPreparation> {
    return this.request<PairingPreparation>("/api/telegram/pairing/start", { method: "POST", body: "{}", headers: { "Content-Type": "application/json" } });
  }

  async cancelTelegramPairing(): Promise<Snapshot> {
    return this.mutate("POST", "/api/telegram/pairing/cancel", {});
  }

  testTelegram(): Promise<Snapshot> {
    return this.mutate("POST", "/api/telegram/test", {});
  }

  async openMacrosFolder(): Promise<Snapshot> {
    return this.mutate("POST", "/api/macros-folder/open", {});
  }

  async quitApplication(): Promise<Snapshot> {
    return this.mutate("POST", "/api/application/quit", {});
  }

  getMigration(signal?: AbortSignal): Promise<MigrationState> {
    return this.request<MigrationState>("/api/migration", { signal });
  }

  async selectMigrationSource(): Promise<MigrationSelectionResult | null> {
    try {
      return await this.mutateResponse<MigrationSelectionResult>("POST", "/api/migration/select-source", {});
    } catch (error) {
      if (error instanceof ApiRequestError && error.code === "migration_cancelled") return null;
      throw error;
    }
  }

  importMigration(): Promise<MigrationImportResult> {
    return this.mutateResponse<MigrationImportResult>("POST", "/api/migration/import", {});
  }

  async prepareShutdown(): Promise<ShutdownPreparation> {
    return this.request<ShutdownPreparation>("/api/system/shutdown/prepare", { method: "POST", body: "{}", headers: { "Content-Type": "application/json" } });
  }

  async confirmShutdown(confirmationId: string): Promise<Snapshot> {
    return this.mutate("POST", "/api/system/shutdown/confirm", { confirmationId });
  }

  async getScreenshot(signal?: AbortSignal): Promise<Blob> {
    if (!this.token) throw new ApiRequestError("La session locale n’est pas prête.", 401, "session_missing");
    const response = await this.fetcher("/api/screenshot", {
      method: "POST",
      body: "{}",
      headers: { "Content-Type": "application/json", "X-Auto-Coc-Session": this.token },
      credentials: "same-origin",
      cache: "no-store",
      signal,
    });
    if (!response.ok) {
      const payload = await readJson<ApiError>(response);
      throw new ApiRequestError(payload.error?.message ?? "La capture n’a pas abouti.", response.status, payload.error?.code);
    }
    if (!response.headers.get("Content-Type")?.toLowerCase().startsWith("image/png")) {
      throw new ApiRequestError("Le service n’a pas renvoyé une image PNG.", response.status);
    }
    return response.blob();
  }

  getHelp(signal?: AbortSignal): Promise<HelpDocument> {
    return this.request<HelpDocument>("/api/help", { signal });
  }

  getDiagnostics(signal?: AbortSignal): Promise<Diagnostics> {
    return this.request<Diagnostics>("/api/diagnostics", { signal });
  }

  async completeOnboarding(): Promise<Snapshot> {
    return this.mutate("POST", "/api/onboarding/complete", {});
  }

  private async mutate(method: string, path: string, body?: unknown): Promise<Snapshot> {
    const response = await this.mutateResponse<SnapshotResponse>(method, path, body);
    return response.snapshot;
  }

  private mutateResponse<T>(method: string, path: string, body?: unknown): Promise<T> {
    return this.request<T>(path, {
      method,
      body: body === undefined ? undefined : JSON.stringify(body),
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    });
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    if (!this.token && path !== "/api/session") {
      throw new ApiRequestError("La session locale n’est pas prête.", 401, "session_missing");
    }
    const headers = new Headers(init.headers);
    if (this.token) headers.set("X-Auto-Coc-Session", this.token);
    const response = await this.fetcher(path, {
      ...init,
      headers,
      credentials: "same-origin",
      cache: "no-store",
    });
    const payload = await readJson<T & ApiError>(response);
    if (!response.ok) {
      throw new ApiRequestError(
        payload.error?.message ?? "La requête n’a pas abouti.",
        response.status,
        payload.error?.code,
      );
    }
    return payload;
  }
}
