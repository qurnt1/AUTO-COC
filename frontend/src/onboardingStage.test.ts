import { describe, expect, it } from "vitest";
import { getOnboardingStage } from "./onboardingStage";

const macro = (name: string, readable: boolean, eventCount = 0) => ({
  name,
  readable,
  eventCount,
  durationSeconds: 0,
  issue: null,
});

describe("onboarding stage", () => {
  it("guides an empty library to macro creation", () => {
    expect(getOnboardingStage({ macros: [], selectedMacro: null })).toBe("create");
  });

  it("guides a user to choose a readable macro when one exists", () => {
    expect(getOnboardingStage({
      macros: [macro("Empty", false), macro("Ready", true, 4)],
      selectedMacro: "Empty",
    })).toBe("select");
    expect(getOnboardingStage({
      macros: [macro("Ready", true, 4)],
      selectedMacro: null,
    })).toBe("select");
  });

  it("guides an unreadable selection to creation", () => {
    expect(getOnboardingStage({ macros: [macro("Broken", false)], selectedMacro: "Broken" })).toBe("create");
  });

  it("guides to creation when no readable macro can be selected", () => {
    expect(getOnboardingStage({ macros: [macro("Broken", false)], selectedMacro: null })).toBe("create");
  });

  it("guides a readable empty macro to recording and a populated one to playback", () => {
    expect(getOnboardingStage({ macros: [macro("Draft", true)], selectedMacro: "Draft" })).toBe("record");
    expect(getOnboardingStage({ macros: [macro("Ready", true, 4)], selectedMacro: "Ready" })).toBe("play");
  });
});
