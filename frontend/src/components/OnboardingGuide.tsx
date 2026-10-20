import { Icon } from "./Icon";

type Stage = "create" | "select" | "record" | "play";

const steps = [
  { key: "create", title: "Créer ou choisir une macro", detail: "Donnez-lui un nom ou choisissez-la dans la bibliothèque." },
  { key: "record", title: "Capturer une séquence", detail: "Après le compte à rebours, enregistrez plus de 3 s. Les 3 dernières secondes seront retirées." },
  { key: "play", title: "Relire dans Clash of Clans", detail: "Le jeu doit être la fenêtre active." },
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
  const activeStep = stage === "select" ? 0 : steps.findIndex((step) => step.key === stage);
  const stageCopy: Record<Stage, string> = {
    create: "Commencez par nommer la macro que vous allez enregistrer.",
    select: "Parcourez la bibliothèque, puis sélectionnez une macro lisible pour continuer.",
    record: "Après le compte à rebours, laissez l’enregistrement tourner plus de 3 s avant de l’arrêter.",
    play: "La séquence est prête. Clash of Clans doit être la fenêtre active pour la lire.",
  };

  return (
    <section className={`onboarding-guide ${helpMode ? "" : "onboarding-guide-atelier"}`} aria-labelledby="onboarding-title">
      <div className="onboarding-icon" aria-hidden="true"><Icon name="keyboard" size={18} /></div>
      <div className="onboarding-copy">
        <span className="section-kicker">{helpMode ? "Guide de démarrage" : `Première utilisation · étape ${activeStep + 1} sur 3`}</span>
        <h2 id="onboarding-title">{helpMode ? "Les trois gestes pour démarrer" : stage === "select" ? "Choisir une macro" : steps[activeStep]?.title ?? steps[0].title}</h2>
        {!helpMode && <p>{stageCopy[stage]}</p>}
        <ol className="onboarding-checklist">
          {steps.map((step, index) => (
            <li key={step.key} className={!helpMode && index < activeStep ? "complete" : !helpMode && index === activeStep ? "current" : ""} aria-current={index === activeStep && !helpMode ? "step" : undefined}>
              <span>{!helpMode && index < activeStep ? <Icon name="check" size={12} /> : `0${index + 1}`}</span>
              <div><strong>{step.title}</strong><small>{step.detail}</small></div>
            </li>
          ))}
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
