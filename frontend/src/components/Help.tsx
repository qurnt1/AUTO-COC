import { useEffect, useState } from "react";
import type { ActionRunner, BackendClient, Diagnostics, HelpDocument, Snapshot } from "../api";
import { Icon } from "./Icon";
import { OnboardingGuide } from "./OnboardingGuide";
import { getOnboardingStage } from "../onboardingStage";
import type { SettingsTab } from "./Settings";

const quickGuides: { title: string; body: string; action: string; destination: "atelier" | "settings"; settingsTab?: SettingsTab }[] = [
  { title: "Créer une macro", body: "Nommez-la, enregistrez plus de 3 s, puis arrêtez. Les 3 dernières secondes sont retirées.", action: "Ouvrir l’atelier", destination: "atelier" as const },
  { title: "Lire une macro", body: "Choisissez une séquence prête. Clash of Clans doit être au premier plan.", action: "Ouvrir l’atelier", destination: "atelier" as const },
  { title: "Piloter avec Telegram", body: "Associez votre bot dans les réglages pour commander les macros à distance.", action: "Ouvrir Telegram", destination: "settings" as const, settingsTab: "telegram" as const },
];

export function Help({ api, snapshot, busy, run, navigate, online, onResumeOnboarding, onCompleteOnboarding }: {
  api: BackendClient;
  snapshot: Snapshot | null;
  busy: string | null;
  run: ActionRunner;
  online: boolean;
  navigate: (view: "atelier" | "settings", settingsTab?: SettingsTab) => void;
  onResumeOnboarding: () => void;
  onCompleteOnboarding: () => void;
}) {
  const [help, setHelp] = useState<HelpDocument | null>(null);
  const [helpError, setHelpError] = useState("");
  const [helpRetry, setHelpRetry] = useState(0);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [diagnosticsError, setDiagnosticsError] = useState("");
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  useEffect(() => {
    if (!online) {
      setHelpError("Réessayez depuis la bannière de connexion pour charger les guides détaillés.");
      return;
    }
    setHelpError("");
    const controller = new AbortController();
    api.getHelp(controller.signal).then(setHelp).catch((error: unknown) => {
      if (!controller.signal.aborted) setHelpError(error instanceof Error ? error.message : "L’aide locale est indisponible.");
    });
    return () => controller.abort();
  }, [api, helpRetry, online]);

  async function loadDiagnostics() {
    setShowDiagnostics(true);
    setDiagnosticsError("");
    if (!online) {
      setDiagnosticsError("Réessayez depuis la bannière de connexion pour charger les diagnostics.");
      return;
    }
    const value = await run("diagnostics", () => api.getDiagnostics());
    if (value) setDiagnostics(value);
    else setDiagnosticsError("Les diagnostics n’ont pas pu être chargés.");
  }

  const keys = snapshot?.settings.shortcuts;
  const onboardingStage = snapshot ? getOnboardingStage(snapshot) : "create";

  return (
    <main className="help-page">
      <header className="page-header">
        <div className="breadcrumbs"><span>Guides</span><Icon name="chevron" size={13} /><span>Aide</span></div>
        <h1 tabIndex={-1}>Que voulez-vous faire ?</h1>
        <p>Choisissez une tâche pour continuer.</p>
      </header>

      {snapshot && !snapshot.onboardingComplete && <OnboardingGuide mode="help" stage={onboardingStage} disabled={!online} busy={busy === "onboarding"} onComplete={onCompleteOnboarding} onResume={onResumeOnboarding} />}

      <div className="help-layout">
        <div className="help-main-column">
          {snapshot?.onboardingComplete !== false && <section className="help-section" aria-labelledby="first-steps-title">
            <div className="section-heading-row"><div><span className="section-kicker">Actions principales</span><h2 id="first-steps-title">Commencer ici</h2></div></div>
            <ol className="help-steps-list">
              {quickGuides.map((item, index) => <li key={item.title}><span className="help-step-index" aria-hidden="true">0{index + 1}</span><div><h3>{item.title}</h3><p>{item.body}</p><button className="text-button" type="button" aria-label={`${item.action} : ${item.title}`} onClick={() => navigate(item.destination, item.settingsTab)}>{item.action}<Icon name="arrow" size={14} /></button></div></li>)}
            </ol>
          </section>}

          <section className="help-section keyboard-reference" aria-labelledby="shortcuts-title">
            <div className="section-heading-row"><div><span className="section-kicker">Contrôle rapide</span><h2 id="shortcuts-title">Vos raccourcis</h2></div><button className="text-button" type="button" disabled={!snapshot} onClick={() => navigate("settings", "shortcuts")}>Modifier<Icon name="arrow" size={14} /></button></div>
            {keys ? <>
              <div className="shortcut-reference-row"><span>Basculer lecture / arrêt</span><kbd>{keys.toggle}</kbd></div>
              <div className="shortcut-reference-row"><span>Lire la macro</span><kbd>{keys.play}</kbd></div>
              <div className="shortcut-reference-row"><span>Arrêter</span><kbd>{keys.stop}</kbd></div>
            </> : <p className="muted-copy">La connexion est nécessaire pour afficher vos raccourcis configurés.</p>}
          </section>

          <section className="help-section local-help-section" aria-labelledby="local-help-title">
            <div className="section-heading-row"><div><span className="section-kicker">Au besoin</span><h2 id="local-help-title">Guides détaillés</h2></div><Icon className="local-help-icon" name="folder" size={19} />
            </div>
            {helpError && <div className="inline-alert error"><Icon name="help" size={17} /><div><strong>L’aide détaillée est indisponible</strong><span>{helpError}</span></div><button className="text-button" type="button" disabled={!online} onClick={() => { setHelp(null); setHelpError(""); setHelpRetry((value) => value + 1); }}>Réessayer</button></div>}
            {help && help.sections.length > 0 && <div className="local-help-list">{help.sections.map((section) => <details key={section.title}><summary>{section.title}<Icon name="chevron" size={15} /></summary><p>{section.body}</p></details>)}</div>}
            {help && !help.sections.length && <p className="muted-copy">Aucun guide complémentaire n’est disponible.</p>}
          </section>
        </div>

        <aside className="help-side-column">
          <section className="help-callout">
            <div className="callout-mark"><Icon name="activity" size={18} /></div>
            <h2>Une macro ne se lance pas ?</h2>
            <p>Vérifiez qu’elle contient des événements. Les diagnostics locaux peuvent préciser le problème.</p>
            <button className="text-button" type="button" disabled={busy !== null} onClick={() => void loadDiagnostics()}>Voir les diagnostics<Icon name="arrow" size={14} /></button>
          </section>
          <section className="help-callout quiet-callout">
            <Icon name="shield" size={18} />
            <h2>Données et sûreté</h2>
            <p>Vos macros et réglages restent sur ce PC. Le token Telegram ne s’affiche pas dans l’aide ni dans les diagnostics.</p>
            <details className="safety-details">
              <summary>Voir les détails de sûreté<Icon name="chevron" size={15} /></summary>
              <div>
                <p>Le replay démarre uniquement si Clash of Clans est au premier plan et reste ciblé sur sa fenêtre. Si le jeu perd le focus, AUTO-COC libère les commandes maintenues puis met en pause ou arrête la lecture selon l’option des réglages. Aucune entrée n’est envoyée à une autre application.</p>
                <p>La capture commence après 3 secondes; à l’arrêt, les 3 dernières secondes sont retirées. Il faut donc laisser enregistrer plus de 3 secondes pour garder une séquence.</p>
                <p>Google Play Games peut ignorer les messages ciblés. Un crash du jeu Android peut rester invisible si sa fenêtre hôte demeure ouverte.</p>
              </div>
            </details>
          </section>
        </aside>
      </div>

      {showDiagnostics && <section className="diagnostics-panel" aria-labelledby="diagnostics-title">
        <div className="diagnostics-head"><div><span className="section-kicker">Informations expurgées</span><h2 id="diagnostics-title">Diagnostics locaux</h2></div><button className="text-button" type="button" onClick={() => setShowDiagnostics(false)}>Masquer</button></div>
        {busy === "diagnostics" && <p className="muted-copy">Lecture des diagnostics…</p>}
        {diagnosticsError && <p className="form-error" role="alert">{diagnosticsError}</p>}
        {diagnostics && <div className="diagnostic-values"><div><span>Version AUTO-COC</span><strong>{diagnostics.appVersion}</strong></div><div><span>Moteur Rust intégré</span><strong>{diagnostics.rustVersion}</strong></div><div><span>État</span><strong>{diagnostics.status}</strong></div><div className="diagnostic-errors"><span>Erreurs signalées</span>{diagnostics.errors.length ? <ul>{diagnostics.errors.map((error, index) => <li key={`${index}-${error}`}>{error}</li>)}</ul> : <strong>Aucune erreur récente</strong>}</div></div>}
      </section>}
    </main>
  );
}
