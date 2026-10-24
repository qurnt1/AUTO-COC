export function reconcileBackendActionError(
  current: string,
  previousBackendError: string | null,
  nextBackendError: string | null,
): string {
  if (nextBackendError) return nextBackendError;
  return previousBackendError && current === previousBackendError ? "" : current;
}
