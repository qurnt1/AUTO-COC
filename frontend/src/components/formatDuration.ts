export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds === 0) return "0 s";
  if (seconds < 60) return `${seconds.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${minutes} min ${rest.toString().padStart(2, "0")} s`;
}
