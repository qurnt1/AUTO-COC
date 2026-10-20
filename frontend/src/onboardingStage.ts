import type { Snapshot } from "./api";

export type OnboardingStage = "create" | "select" | "record" | "play";

export function getOnboardingStage(snapshot: Pick<Snapshot, "macros" | "selectedMacro">): OnboardingStage {
  const selected = snapshot.macros.find((macro) => macro.name === snapshot.selectedMacro) ?? null;
  if (snapshot.macros.length === 0) return "create";
  const hasReadableMacro = snapshot.macros.some((macro) => macro.readable);
  if (!selected) return hasReadableMacro ? "select" : "create";
  if (!selected.readable && hasReadableMacro) return "select";
  if (!selected.readable) return "create";
  return selected.eventCount > 0 ? "play" : "record";
}
