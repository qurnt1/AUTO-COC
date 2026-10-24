import { isTauri } from "@tauri-apps/api/core";
import { BackendClientError, type BackendClient } from "./api";
import { TauriBackendClient } from "./tauriBackend";

export function createBackendClient(): BackendClient {
  if (!isTauri()) {
    throw new BackendClientError(
      "AUTO-COC fonctionne dans son application de bureau. En développement, lancez-la avec `cargo tauri dev`.",
      undefined,
      "unsupported_runtime",
    );
  }
  return new TauriBackendClient();
}
