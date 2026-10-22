import { Icon } from "./Icon";

type Stage = "create" | "select" | "record" | "play";

const steps = [
  { key: "create", title: "Créer une macro" },
  { key: "record", title: "Enregistrer une séquence" },
  { key: "play", title: "Lire dans Clash of Clans" },
] as const;

const actionLabels: Record<Stage, string> = {
  create: "Créer une macro",
  select: "Parcourir la liste des macros",
  record: "Enregistrer une séquence",
  play: "Aller au bouton Lire",
};

export function OnboardingGuide({ mode, stage = "create", disabled, busy, onPrimaryAction, onComplete, onDismiss, onResume }: {
  mode: "atelier" | "help";
  stage?: Stage;
  disabled: boolean;
  busy: boolean;
  onPrimaryAction?: () => void;
  onComplete: () => void;
  onDismiss?: () => void;
  onResume?: () => void;
}) {
  const helpMode = mode === "help";
  const activeStep = stage === "record" ? 1 : stage === "play" ? 2 : 0;
  const stageTitle: Record<Stage, string> = {
    create: steps[0].title,
    select: "Choisir une macro",
    record: steps[1].title,
    play: steps[2].title,
  };
  const stageCopy: Record<Stage, string> = {
    create: "Donnez un nom à la macro que vous allez enregistrer.",
    select: "Choisissez une macro lisible dans la bibliothèque.",
    record: "Après le compte à rebours, enregistrez plus de 3 s. Les 3 dernières secondes sont retirées.",
    play: "La séquence est prête. Clash of Clans doit être la fenêtre active pour la lire.",
  };

  return (
    <section className={`onboarding-guide ${helpMode ? "" : "onboarding-guide-atelier"}`} aria-labelledby="onboarding-title">
      <div className="onboarding-icon" aria-hidden="true"><Icon name="keyboard" size={18} /></div>
      <div className="onboarding-copy">
        <span className="section-kicker">{helpMode ? `Étape active · ${activeStep + 1} sur 3` : `Première utilisation · étape ${activeStep + 1} sur 3`}</span>
        <h2 id="onboarding-title">{stageTitle[stage]}</h2>
        <p>{stageCopy[stage]}</p>
        <ol className="onboarding-checklist" aria-label={`Progression du guide, étape ${activeStep + 1} sur 3`}>
          {steps.map((step, index) => {
            const title = index === activeStep ? stageTitle[stage] : step.title;
            return <li key={step.key} className={index < activeStep ? "complete" : index === activeStep ? "current" : ""} aria-label={`${title}, ${index < activeStep ? "terminée" : index === activeStep ? "étape actuelle" : "à venir"}`} aria-current={index === activeStep ? "step" : undefined}>
              <span>{index < activeStep ? <Icon name="check" size={12} /> : `0${index + 1}`}</span>
            </li>;
          })}
        </ol>
      </div>
      <div className="onboarding-actions">
        {helpMode
          ? <button className="button primary" type="button" onClick={onResume}>Reprendre dans l’atelier<Icon name="arrow" size={14} /></button>
          : <button className="button primary" type="button" disabled={disabled || busy || !onPrimaryAction} onClick={onPrimaryAction}>{busy ? "Enregistrement…" : actionLabels[stage]}</button>}
        <button className="text-button" type="button" disabled={disabled || busy} onClick={onComplete}>{busy ? "Enregistrement…" : "Terminer le guide"}</button>
      </div>
      {onDismiss && <button className="icon-button subtle onboarding-dismiss" type="button" aria-label="Masquer le guide de démarrage" title="Masquer le guide" onClick={onDismiss}><Icon name="close" size={16} /></button>}
    </section>
  );
}
