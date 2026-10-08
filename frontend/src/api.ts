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
  | { kind: "waiting_for_foreground"; action: "recording" | "playing"; macroName: string }
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
    requireCocForeground: boolean;
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
  lastError: string | null;
};

export type Diagnostics = {
  appVersion: string;
  rustVersion: string;
  status: string;
  errors: string[];
};

export type HelpSection = { title: string; body: string };
export type HelpDocument = { sections: HelpSection[] };

export type ShutdownPreparation = { confirmationId: string; expiresAt: string };
export type PairingPreparation = { code: string; expiresAt: string; snapshot: Snapshot };

export interface BackendClient {
  connect(signal?: AbortSignal): Promise<Snapshot>;
  waitForSnapshot(after: number, signal?: AbortSignal): Promise<Snapshot>;
  getSnapshot(signal?: AbortSignal): Promise<Snapshot>;
  getMacro(name: string, signal?: AbortSignal): Promise<MacroDetail>;
  createMacro(name: string): Promise<Snapshot>;
  renameMacro(name: string, newName: string): Promise<Snapshot>;
  deleteMacro(name: string): Promise<Snapshot>;
  selectMacro(name: string): Promise<Snapshot>;
  startRecording(): Promise<Snapshot>;
  stopRecording(): Promise<Snapshot>;
  startPlayback(): Promise<Snapshot>;
  stopPlayback(): Promise<Snapshot>;
  updateSettings(settings: { loop?: boolean; cocPath?: string; requireCocForeground?: boolean }): Promise<Snapshot>;
  updateShortcuts(shortcuts: Snapshot["settings"]["shortcuts"]): Promise<Snapshot>;
  launchCoc(): Promise<Snapshot>;
  saveTelegramToken(token: string): Promise<Snapshot>;
  clearTelegramToken(): Promise<Snapshot>;
  startTelegramPairing(): Promise<PairingPreparation>;
  cancelTelegramPairing(): Promise<Snapshot>;
  testTelegram(): Promise<Snapshot>;
  openMacrosFolder(): Promise<Snapshot>;
  quitApplication(): Promise<void>;
  getMigration(signal?: AbortSignal): Promise<MigrationState>;
  selectMigrationSource(): Promise<MigrationSelectionResult | null>;
  importMigration(): Promise<MigrationImportResult>;
  prepareShutdown(): Promise<ShutdownPreparation>;
  confirmShutdown(confirmationId: string): Promise<Snapshot>;
  getScreenshot(signal?: AbortSignal): Promise<Blob>;
  getHelp(signal?: AbortSignal): Promise<HelpDocument>;
  getDiagnostics(signal?: AbortSignal): Promise<Diagnostics>;
  completeOnboarding(): Promise<Snapshot>;
  listenShutdownErrors(handler: (error: BackendClientError) => void): () => void;
  dispose(): void;
}

export class BackendClientError extends Error {
  readonly status?: number;
  readonly code?: string;

  constructor(message: string, status?: number, code?: string) {
    super(message);
    this.name = "BackendClientError";
    this.status = status;
    this.code = code;
  }
}
