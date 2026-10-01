import { Icon } from "./Icon";

export function OnboardingGuide({ mode, disabled, busy, onComplete, onDismiss, onResume }: {
  mode: "atelier" | "help";
  disabled: boolean;
  busy: boolean;
  onComplete: () => void;
  onDismiss?: () => void;
  onResume?: () => void;
}) {
  const helpMode = mode === "help";

  return (
    <section className={`onboarding-guide ${helpMode ? "" : "onboarding-guide-atelier"}`} aria-labelledby="onboarding-title">
      <div className="onboarding-icon"><Icon name="spark" size={20} /></div>
      <div className="onboarding-copy">
        <span className="section-kicker">{helpMode ? "Guide de démarrage" : "Première utilisation"}</span>
        <h2 id="onboarding-title">{helpMode ? "Reprendre le guide de démarrage" : "Votre première macro, en trois gestes"}</h2>
        <ol className="onboarding-checklist">
          <li><span>01</span><div><strong>Créer ou choisir une macro</strong><small>Donnez-lui un nom que vous retrouverez facilement.</small></div></li>
          <li><span>02</span><div><strong>Enregistrer une séquence</strong><small>La capture commence après 3 s et retire les 3 dernières secondes à l’arrêt.</small></div></li>
          <li><span>03</span><div><strong>Relire depuis l’atelier</strong><small>Les réglages CoC et Telegram restent facultatifs.</small></div></li>
        </ol>
        <p>{helpMode ? "Terminer le guide l’enregistre comme terminé. Les gestes restent accessibles dans les rubriques ci-dessous." : "Terminer le guide le marque comme terminé. Masquer le retire de l’atelier sans le terminer; vous pourrez le reprendre depuis Aide."}</p>
      </div>
      <div className="onboarding-actions">
        {helpMode && <button className="button quiet" type="button" onClick={onResume}>Reprendre dans l’atelier<Icon name="arrow" size={14} /></button>}
        <button className="text-button" type="button" disabled={disabled || busy} onClick={onComplete}>{busy ? "Enregistrement…" : "Terminer le guide"}</button>
      </div>
      {onDismiss && <button className="icon-button subtle onboarding-dismiss" type="button" aria-label="Masquer le guide de démarrage" title="Masquer le guide" onClick={onDismiss}><Icon name="close" size={16} /></button>}
    </section>
  );
}
