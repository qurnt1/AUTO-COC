import { useEffect, useState } from "react";
import type { ActionRunner, BackendClient, Diagnostics, HelpDocument, MigrationImportResult, MigrationPreview, MigrationState, Snapshot } from "../api";
import { Icon } from "./Icon";
import { OnboardingGuide } from "./OnboardingGuide";

const quickGuides = [
  { title: "Créer et enregistrer", body: "Créez une macro, sélectionnez-la dans l’atelier, puis choisissez Enregistrer. Une préparation de 3 secondes précède la capture; les 3 dernières secondes sont retirées à l’arrêt. Si la capture dure 3 secondes ou moins, la macro reste vide." },
  { title: "Relire ou arrêter", body: "Choisissez une macro lisible et lancez sa lecture depuis l’atelier. F1 lance la macro sélectionnée à l’arrêt et stoppe une lecture active. Pendant un enregistrement, F1 ne fait rien; le raccourci Lire démarre uniquement à l’arrêt et le raccourci Arrêter termine la lecture ou finalise l’enregistrement." },
  { title: "Régler les raccourcis", body: "Ouvrez Réglages puis Raccourcis. Cliquez sur Modifier, appuyez sur une touche ou une combinaison, puis enregistrez." },
];

export function Help({ api, snapshot, busy, run, navigate, online, onResumeOnboarding, onCompleteOnboarding }: {
  api: BackendClient;
  snapshot: Snapshot | null;
  busy: string | null;
  run: ActionRunner;
  online: boolean;
  navigate: (view: "atelier" | "settings") => void;
  onResumeOnboarding: () => void;
  onCompleteOnboarding: () => void;
}) {
  const [help, setHelp] = useState<HelpDocument | null>(null);
  const [helpError, setHelpError] = useState("");
  const [helpRetry, setHelpRetry] = useState(0);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [diagnosticsError, setDiagnosticsError] = useState("");
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [migration, setMigration] = useState<MigrationState | null>(null);
  const [migrationError, setMigrationError] = useState("");
  const [importResult, setImportResult] = useState<MigrationImportResult | null>(null);
  const migrationAlreadyImported = snapshot?.migration.alreadyImported;

  useEffect(() => {
    if (!online) {
      setHelpError("Reconnectez le service local pour charger les guides complémentaires.");
      return;
    }
    setHelpError("");
    const controller = new AbortController();
    api.getHelp(controller.signal).then(setHelp).catch((error: unknown) => {
      if (!controller.signal.aborted) setHelpError(error instanceof Error ? error.message : "L’aide locale est indisponible.");
    });
    return () => controller.abort();
  }, [api, helpRetry, online]);

  useEffect(() => {
    if (migrationAlreadyImported === undefined || !online) return;
    const controller = new AbortController();
    api.getMigration(controller.signal).then(setMigration).catch((error: unknown) => {
      if (!controller.signal.aborted) setMigrationError(error instanceof Error ? error.message : "L’état de l’import est indisponible.");
    });
    return () => controller.abort();
  }, [api, migrationAlreadyImported, online]);

  async function loadDiagnostics() {
    setShowDiagnostics(true);
    setDiagnosticsError("");
    if (!online) {
      setDiagnosticsError("Reconnectez le service local pour charger les diagnostics.");
      return;
    }
    const value = await run("diagnostics", () => api.getDiagnostics());
    if (value) setDiagnostics(value);
    else setDiagnosticsError("Les diagnostics n’ont pas pu être chargés.");
  }

  const keys = snapshot?.settings.shortcuts;
  async function selectMigrationSource() {
    setMigrationError("");
    const selection = await run("migration-source", () => api.selectMigrationSource());
    if (selection && snapshot) setMigration({ status: { ...snapshot.migration, sourceSelected: true, available: selection.preview.available }, preview: selection.preview });
  }

  async function importSelectedSource() {
    const result = await run("migration-import", () => api.importMigration(), "Import terminé. Le dossier source a été conservé.");
    if (result) {
      setImportResult(result);
      setMigration((current) => current ? { ...current, status: { ...current.status, alreadyImported: true } } : current);
    }
  }

  return (
    <main className="help-page">
      <header className="page-header">
        <div className="breadcrumbs"><span>Guides</span><Icon name="chevron" size={13} /><span>Aide</span></div>
        <h1 tabIndex={-1}>Créer, relire, reprendre la main</h1>
        <p>Des repères simples pour créer votre première séquence et piloter AUTO-COC.</p>
      </header>

      {snapshot && !snapshot.onboardingComplete && <OnboardingGuide mode="help" disabled={!online} busy={busy === "onboarding"} onComplete={onCompleteOnboarding} onResume={onResumeOnboarding} />}

      {snapshot && !snapshot.migration.alreadyImported && <MigrationPanel migration={migration} loading={online && migration === null && !migrationError} error={migrationError} busy={busy} online={online} importResult={importResult} onSelect={() => void selectMigrationSource()} onImport={() => void importSelectedSource()} />}
      {snapshot?.migration.alreadyImported && importResult && <div className="migration-success" role="status"><Icon name="check" size={16} /><span>Import terminé : {importResult.imported.macros} macros, {importResult.imported.settings} réglages.</span></div>}

      <div className="help-layout">
        <div className="help-main-column">
          <section className="help-section" aria-labelledby="first-steps-title">
            <div className="section-heading-row"><div><span className="section-kicker">Bien démarrer</span><h2 id="first-steps-title">Les premiers gestes</h2></div><span className="help-steps-mark" aria-hidden="true"><i /><i /><i /></span></div>
            <ol className="help-steps-list">
              {quickGuides.map((item, index) => <li key={item.title}><span className="help-step-index">{index + 1}</span><div><h3>{item.title}</h3><p>{item.body}</p></div></li>)}
            </ol>
          </section>

          <section className="help-section keyboard-reference" aria-labelledby="shortcuts-title">
            <div className="section-heading-row"><div><span className="section-kicker">Contrôle rapide</span><h2 id="shortcuts-title">Vos raccourcis</h2></div><button className="text-button" type="button" disabled={!snapshot} onClick={() => navigate("settings")}>Modifier<Icon name="arrow" size={14} /></button></div>
            {keys ? <>
              <div className="shortcut-reference-row"><span>Basculer lecture / arrêt</span><kbd>{keys.toggle}</kbd></div>
              <div className="shortcut-reference-row"><span>Lire la macro</span><kbd>{keys.play}</kbd></div>
              <div className="shortcut-reference-row"><span>Arrêter</span><kbd>{keys.stop}</kbd></div>
            </> : <p className="muted-copy">Reconnectez le service local pour afficher vos raccourcis configurés.</p>}
          </section>

          <section className="help-section local-help-section" aria-labelledby="local-help-title">
            <div className="section-heading-row"><div><span className="section-kicker">Documentation intégrée</span><h2 id="local-help-title">Guides AUTO-COC</h2></div><Icon className="local-help-icon" name="folder" size={19} /></div>
            {helpError && <div className="inline-alert error"><Icon name="help" size={17} /><div><strong>L’aide détaillée ne répond pas</strong><span>{helpError}</span></div><button className="text-button" type="button" disabled={!online} onClick={() => { setHelp(null); setHelpError(""); setHelpRetry((value) => value + 1); }}>Réessayer</button></div>}
            {help && help.sections.length > 0 && <div className="local-help-list">{help.sections.map((section) => <details key={section.title}><summary>{section.title}<Icon name="chevron" size={15} /></summary><p>{section.body}</p></details>)}</div>}
            {help && !help.sections.length && <p className="muted-copy">Aucun guide complémentaire n’est disponible pour le moment.</p>}
          </section>
        </div>

        <aside className="help-side-column">
          <section className="help-callout">
            <div className="callout-mark"><Icon name="activity" size={18} /></div>
            <h2>Une macro ne se lance pas ?</h2>
            <p>Vérifiez qu’elle contient des événements et qu’elle est lisible. En cas de problème, ouvrez les diagnostics locaux.</p>
            <button className="text-button" type="button" disabled={busy !== null} onClick={() => void loadDiagnostics()}>Voir les diagnostics<Icon name="arrow" size={14} /></button>
          </section>
          <section className="help-callout quiet-callout">
            <Icon name="shield" size={18} />
            <h2>Vos données restent sur ce PC</h2>
            <p>Les macros et les réglages sont gérés par le service Rust local. Le token Telegram n’est jamais affiché dans l’aide ni dans les diagnostics.</p>
            <p>Pour arrêter les services locaux, utilisez Réglages &gt; Outils de cet ordinateur &gt; Quitter AUTO-COC ou fermez complètement la fenêtre de l’application. Arrêtez tout enregistrement ou replay en cours avant de quitter.</p>
          </section>
        </aside>
      </div>

      {showDiagnostics && <section className="diagnostics-panel" aria-labelledby="diagnostics-title">
        <div className="diagnostics-head"><div><span className="section-kicker">Informations expurgées</span><h2 id="diagnostics-title">Diagnostics locaux</h2></div><button className="text-button" type="button" onClick={() => setShowDiagnostics(false)}>Masquer</button></div>
        {busy === "diagnostics" && <p className="muted-copy">Lecture des diagnostics…</p>}
        {diagnosticsError && <p className="form-error" role="alert">{diagnosticsError}</p>}
        {diagnostics && <div className="diagnostic-values"><div><span>Version AUTO-COC</span><strong>{diagnostics.appVersion}</strong></div><div><span>Service Rust</span><strong>{diagnostics.rustVersion}</strong></div><div><span>État</span><strong>{diagnostics.status}</strong></div><div className="diagnostic-errors"><span>Erreurs signalées</span>{diagnostics.errors.length ? <ul>{diagnostics.errors.map((error, index) => <li key={`${index}-${error}`}>{error}</li>)}</ul> : <strong>Aucune erreur récente</strong>}</div></div>}
      </section>}
    </main>
  );
}

function MigrationPanel({ migration, loading, error, busy, online, importResult, onSelect, onImport }: {
  migration: MigrationState | null;
  loading: boolean;
  error: string;
  busy: string | null;
  online: boolean;
  importResult: MigrationImportResult | null;
  onSelect: () => void;
  onImport: () => void;
}) {
  const preview: MigrationPreview | null = migration?.preview ?? null;
  return (
    <section className="migration-panel" aria-labelledby="migration-title">
      <div className="migration-intro">
        <span className="migration-icon"><Icon name="refresh" size={18} /></span>
        <div><span className="section-kicker">Ancienne installation</span><h2 id="migration-title">Importer vos données</h2><p>Récupérez vos macros et réglages depuis une ancienne version d’AUTO-COC.</p></div>
      </div>
      <p className="migration-safety">Une sauvegarde locale expurgée du token est créée avant l’import. Les fichiers d’origine restent à leur emplacement et ne sont pas modifiés. Le CSV source peut contenir le token historique en clair; il n’est pas copié dans la sauvegarde, et le token importé est stocké séparément de façon protégée.</p>
      {loading && <p className="muted-copy">Vérification de l’état de l’import…</p>}
      {error && <p className="form-error" role="alert">{error}</p>}
      {migration?.status.alreadyImported ? (
        <p className="migration-complete"><Icon name="check" size={15} />Ces données ont déjà été importées.</p>
      ) : preview ? (
        <div className="migration-preview">
          <div className="migration-preview-meta"><span>{preview.macros.length} {preview.macros.length === 1 ? "macro trouvée" : "macros trouvées"}</span><span>{preview.settingsFound ? "Réglages détectés, valeurs masquées" : "Aucun réglage détecté"}</span></div>
          {preview.macros.length > 0 && <ul className="migration-macro-list">{preview.macros.map((macro) => <li key={macro.name}><span className={`migration-readability ${macro.readable ? "" : "unreadable"}`} role="img" aria-label={macro.readable ? "Lisible" : "À vérifier"} /><span className="migration-macro-name">{macro.name}</span><span>{macro.eventCount} événements</span>{!macro.readable && <small>{macro.issue ?? "À vérifier"}</small>}</li>)}</ul>}
          {migration?.status.available ? <button className="button primary" type="button" onClick={onImport} disabled={!online || busy !== null}>{busy === "migration-import" ? "Import en cours…" : "Importer ces données"}</button> : <p className="muted-copy">Aucune donnée compatible à importer depuis le dossier choisi.</p>}
          <button className="text-button migration-reselect" type="button" onClick={onSelect} disabled={!online || busy !== null}>Choisir un autre dossier</button>
        </div>
      ) : (
        <div className="migration-picker">
          <p>{migration?.status.sourceSelected ? "Aucun aperçu disponible. Vous pouvez choisir un autre dossier." : "Choisissez l’ancien dossier du projet. La fenêtre de sélection Windows s’ouvrira."}</p>
          <button className="button quiet" type="button" onClick={onSelect} disabled={!online || busy !== null}>{busy === "migration-source" ? "Ouverture du sélecteur…" : "Choisir un dossier"}</button>
        </div>
      )}
      {importResult && <div className="migration-result" role="status"><strong>Import terminé</strong><span>{importResult.imported.macros} macros et {importResult.imported.settings} réglages importés.</span>{importResult.collisions.length > 0 && <span>{importResult.collisions.length} collision(s) préservée(s), aucun fichier existant remplacé.</span>}{importResult.preserved.length > 0 && <span>{importResult.preserved.length} élément(s) préservé(s).</span>}{importResult.errors.length > 0 && <span>{importResult.errors.length} erreur(s) signalée(s).</span>}</div>}
    </section>
  );
}
