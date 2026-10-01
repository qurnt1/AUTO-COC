import { useEffect, useMemo, useState } from "react";
import type { ActionRunner, ApiClient, MacroDetail, MacroStep, MacroSummary, Snapshot } from "../api";
import { Dialog } from "./Dialog";
import { Icon } from "./Icon";
import { OnboardingGuide } from "./OnboardingGuide";
import { formatDuration } from "./formatDuration";
import {
  getSequencePageBounds,
  getTimelineMarkers,
  nextSequencePageStart,
  previousSequencePageStart,
} from "./sequence";

type DialogState = { kind: "create" } | { kind: "rename"; macro: MacroSummary } | { kind: "delete"; macro: MacroSummary } | null;
type PageProps = { api: ApiClient; snapshot: Snapshot; busy: string | null; run: ActionRunner; online: boolean; showOnboarding: boolean; onDismissOnboarding: () => void; onCompleteOnboarding: () => void };

const eventNames: Record<string, { label: string; icon: "mouse" | "key" | "scroll" | "dots" }> = {
  mouse_move: { label: "Déplacement", icon: "mouse" },
  mouse_click: { label: "Clic", icon: "mouse" },
  scroll: { label: "Défilement", icon: "scroll" },
  key_down: { label: "Touche enfoncée", icon: "key" },
  key_up: { label: "Touche relâchée", icon: "key" },
  nop: { label: "Pause", icon: "dots" },
};

function formatTime(seconds: number): string {
  return `${Math.max(0, seconds).toFixed(2).replace(".", ",")} s`;
}

function MacroDialog({ dialog, busy, onClose, onSubmit }: {
  dialog: Exclude<DialogState, null>;
  busy: boolean;
  onClose: () => void;
  onSubmit: (name: string) => void;
}) {
  const [name, setName] = useState(dialog.kind === "rename" ? dialog.macro.name : "");
  const destructive = dialog.kind === "delete";
  const label = dialog.kind === "create" ? "Nom de la macro" : "Nouveau nom";
  const title = dialog.kind === "create" ? "Créer une macro" : dialog.kind === "rename" ? "Renommer la macro" : "Supprimer cette macro ?";
  const description = destructive
    ? `« ${dialog.macro.name} » sera supprimée de votre bibliothèque. Cette action ne peut pas être annulée.`
    : "Choisissez un nom simple que vous retrouverez facilement.";

  return (
    <Dialog title={title} description={description} onClose={onClose}>
      <form className="dialog-form" onSubmit={(event) => { event.preventDefault(); onSubmit(name.trim()); }}>
        {!destructive && <label className="field-label" htmlFor="macro-name">{label}</label>}
        {!destructive && <input id="macro-name" autoFocus maxLength={80} value={name} onChange={(event) => setName(event.target.value)} placeholder="Ex. Récolte des ressources" />}
        <div className="dialog-actions">
          <button className="button quiet" type="button" onClick={onClose}>Annuler</button>
          <button className={`button ${destructive ? "danger" : "primary"}`} type="submit" disabled={busy || (!destructive && !name.trim())}>
            {busy ? "Patientez…" : destructive ? "Supprimer" : dialog.kind === "create" ? "Créer la macro" : "Enregistrer le nom"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

function Sequence({ detail, summary }: { detail: MacroDetail | null; summary: MacroSummary }) {
  const [pageStart, setPageStart] = useState(0);
  const timelineMarkers = useMemo(() => detail ? getTimelineMarkers(detail.steps) : [], [detail]);
  if (summary.eventCount === 0) {
    return (
      <div className="sequence-empty">
        <div className="sequence-empty-mark" aria-hidden="true"><span /><span /><span /><span /><span /><span /></div>
        <p>Cette macro n’a pas encore de séquence.</p>
        <span>Enregistrez une première action pour la remplir.</span>
      </div>
    );
  }
  if (!detail) return <div className="sequence-loading" role="status" aria-label="Chargement de la séquence"><span /><span /><span /></div>;

  const page = getSequencePageBounds(detail.steps.length, pageStart);
  const steps = detail.steps.slice(page.start, page.end);
  const pageHasInternalScroll = steps.length > 9;
  return (
    <div className="sequence-wrap">
      <div className="trace-heading"><span>Déroulé réel</span><span>{detail.steps.length} événements</span></div>
      <div className="trace-track" aria-hidden="true">
        {timelineMarkers.map(({ step, index, left }) => <span key={`${index}-${step.t}-${step.type}`} className={`trace-node ${step.type === "mouse_click" ? "is-click" : step.type.startsWith("key") ? "is-key" : ""}`} style={{ left }} title={`${index + 1}. ${eventNames[step.type]?.label ?? step.type}`} />)}
      </div>
      {pageHasInternalScroll && <p id="sequence-scroll-hint" className="sequence-scroll-hint">Faites défiler la liste pour parcourir cette page.</p>}
      <ol
        key={page.start}
        className="sequence-list"
        tabIndex={0}
        aria-label={`Événements ${page.start + 1} à ${page.end} sur ${detail.steps.length}`}
        aria-describedby={pageHasInternalScroll ? "sequence-scroll-hint" : undefined}
      >
        {steps.map((step, index) => <SequenceRow key={`${page.start + index}-${step.t}-${step.type}`} step={step} index={page.start + index} />)}
      </ol>
      {detail.steps.length > 8 && (
        <nav className="sequence-more" aria-label="Navigation des événements">
          <button className="text-button" type="button" disabled={page.start === 0} onClick={() => setPageStart(previousSequencePageStart(page.start))}>
            Précédent
          </button>
          <span role="status" aria-live="polite" aria-atomic="true">
            Événements {page.start + 1} à {page.end} sur {detail.steps.length}
          </span>
          <button className="text-button" type="button" disabled={page.end >= detail.steps.length} onClick={() => setPageStart(nextSequencePageStart(detail.steps.length, page.start))}>
            Suivant
          </button>
        </nav>
      )}
    </div>
  );
}

function SequenceRow({ step, index }: { step: MacroStep; index: number }) {
  const event = eventNames[step.type] ?? { label: `Événement : ${step.type}`, icon: "dots" as const };
  return (
    <li className="sequence-row">
      <span className="sequence-index">{String(index + 1).padStart(2, "0")}</span>
      <span className={`sequence-symbol ${step.type === "mouse_click" ? "click" : ""}`}><Icon name={event.icon} size={15} /></span>
      <span className="sequence-event">{event.label}</span>
      <span className="sequence-time">{formatTime(step.t)}</span>
    </li>
  );
}

export function Atelier({ api, snapshot, busy, run, online, showOnboarding, onDismissOnboarding, onCompleteOnboarding }: PageProps) {
  const [query, setQuery] = useState("");
  const [dialog, setDialog] = useState<DialogState>(null);
  const [detail, setDetail] = useState<MacroDetail | null>(null);
  const [detailError, setDetailError] = useState("");
  const [detailLoading, setDetailLoading] = useState(false);
  const selected = snapshot.macros.find((macro) => macro.name === snapshot.selectedMacro) ?? null;
  const active = snapshot.status.kind !== "idle";
  const filtered = snapshot.macros.filter((macro) => macro.name.toLocaleLowerCase("fr").includes(query.trim().toLocaleLowerCase("fr")));
  const recordingSelectedMacro = snapshot.status.kind === "recording" && snapshot.status.macroName === selected?.name;
  const selectionVisible = filtered.some((macro) => macro.name === selected?.name);

  useEffect(() => {
    setDetail(null);
    setDetailError("");
    setDetailLoading(false);
    if (!selected?.readable || !selected.eventCount || recordingSelectedMacro) return;
    const controller = new AbortController();
    setDetailLoading(true);
    api.getMacro(selected.name, controller.signal)
      .then(setDetail)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setDetailError(error instanceof Error ? error.message : "Impossible de lire cette séquence.");
      })
      .finally(() => { if (!controller.signal.aborted) setDetailLoading(false); });
    return () => controller.abort();
  }, [api, selected?.name, selected?.readable, selected?.eventCount, recordingSelectedMacro]);

  async function submitDialog(value: string) {
    if (!dialog) return;
    let result: Snapshot | undefined;
    if (dialog.kind === "delete") {
      result = await run("delete", () => api.deleteMacro(dialog.macro.name), "Macro supprimée.");
    } else if (dialog.kind === "rename") {
      result = await run("rename", () => api.renameMacro(dialog.macro.name, value), "Nom de la macro modifié.");
    } else {
      result = await run("create", () => api.createMacro(value), "Macro créée.");
    }
    if (result) setDialog(null);
  }

  return (
    <>
    {showOnboarding && <OnboardingGuide mode="atelier" disabled={!online} busy={busy === "onboarding"} onComplete={onCompleteOnboarding} onDismiss={onDismissOnboarding} />}
    <main className="workbench">
      <aside className="library" aria-label="Bibliothèque des macros">
        <div className="library-heading">
          <div><span className="section-kicker">Bibliothèque</span><h2>Vos macros</h2></div>
          <button className="icon-button add-button" type="button" aria-label="Créer une macro" title="Créer une macro" onClick={() => setDialog({ kind: "create" })} disabled={!online || active}>
            <Icon name="plus" />
          </button>
        </div>
        <label className="search-box">
          <Icon name="search" size={16} />
          <span className="sr-only">Rechercher une macro</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Rechercher…" />
          {query && <button type="button" className="clear-search" aria-label="Effacer la recherche" onClick={() => setQuery("")}><Icon name="close" size={14} /></button>}
        </label>
        <div className="library-meta"><span>{snapshot.macros.length} {snapshot.macros.length > 1 ? "macros" : "macro"}</span></div>
        <div className="macro-list" role="listbox" aria-label="Macros">
          {filtered.map((macro, index) => {
            const current = macro.name === selected?.name;
            return (
              <button
                className={`macro-item ${current ? "selected" : ""}`}
                id={`macro-${encodeURIComponent(macro.name)}`}
                key={macro.name}
                type="button"
                role="option"
                aria-selected={current}
                tabIndex={current || (!selectionVisible && index === 0) ? 0 : -1}
                onClick={() => run("select", () => api.selectMacro(macro.name))}
                onKeyDown={(event) => {
                  const destination = event.key === "ArrowDown" ? index + 1 : event.key === "ArrowUp" ? index - 1 : event.key === "Home" ? 0 : event.key === "End" ? filtered.length - 1 : -1;
                  if (destination < 0 || destination >= filtered.length) return;
                  event.preventDefault();
                  const target = filtered[destination];
                  void run("select", () => api.selectMacro(target.name));
                  window.requestAnimationFrame(() => document.getElementById(`macro-${encodeURIComponent(target.name)}`)?.focus());
                }}
                disabled={!online || active || busy !== null}
              >
                <span className="macro-item-marker" aria-hidden="true"><span /></span>
                <span className="macro-item-copy">
                  <span className="macro-item-name">{macro.protected && <span className="macro-system-tag">Système</span>}{macro.name}{macro.protected && <Icon className="protected-icon" name="lock" size={13} />}</span>
                  <span className="macro-item-meta">{macro.readable ? `${macro.eventCount} événements` : "À vérifier"}<span className="meta-dot">·</span>{formatDuration(macro.durationSeconds)}</span>
                </span>
                {current && <Icon className="selected-chevron" name="chevron" size={15} />}
              </button>
            );
          })}
          {filtered.length === 0 && (
            <div className="library-empty">
              <Icon name={query ? "search" : "activity"} size={19} />
              <p>{query ? "Aucune macro trouvée" : "Votre bibliothèque est vide"}</p>
              <span>{query ? "Essayez un autre nom." : "Créez une macro pour commencer."}</span>
              {!query && <button type="button" className="text-button" onClick={() => setDialog({ kind: "create" })} disabled={!online || active || busy !== null}>Créer la première</button>}
            </div>
          )}
        </div>
        <button className="library-create" type="button" onClick={() => setDialog({ kind: "create" })} disabled={!online || active}>
          <Icon name="plus" size={16} /><span>Nouvelle macro</span>
        </button>
      </aside>

      <section className="macro-workspace" aria-label="Détail de la macro">
        {!selected ? (
          <div className="welcome-empty">
            <div className="empty-trace" aria-hidden="true"><span /><span /><span /><span /><i /><span /></div>
            <span className="section-kicker">L’atelier commence ici</span>
            <h1 tabIndex={-1}>{snapshot.macros.length ? "Choisissez une macro" : "Créez votre première macro"}</h1>
            <p>{snapshot.macros.length ? "Sélectionnez une macro dans la bibliothèque pour voir sa séquence et la lancer." : "Enregistrez une suite d’actions, puis rejouez-la quand vous en avez besoin."}</p>
            {!snapshot.macros.length && <button className="button primary" type="button" onClick={() => setDialog({ kind: "create" })} disabled={!online || active || busy !== null}><Icon name="plus" size={16} />Créer une macro</button>}
            {snapshot.macros.length > 0 && <div className="empty-tip"><Icon name="keyboard" size={16} />Vous pouvez aussi parcourir la liste avec les flèches du clavier.</div>}
          </div>
        ) : (
          <>
            <div className="workspace-topbar">
              <div className="breadcrumbs"><span>Atelier</span><Icon name="chevron" size={13} /><span>Macro</span></div>
              <div className="workspace-tools">
                <span className={`state-label state-${snapshot.status.kind}`}><i aria-hidden="true" />{statusLabel(snapshot.status)}</span>
                {selected.protected && <span className="topbar-protected"><Icon name="lock" size={13} />Système</span>}
              </div>
            </div>
            <div className="macro-heading">
              <div className="macro-title-line">
                <h1 tabIndex={-1}>{selected.name}</h1>
                {selected.protected && <span className="protected-tag"><Icon name="lock" size={12} />Macro système</span>}
              </div>
              <p>{selected.readable ? "Séquence prête à être examinée ou rejouée." : selected.issue ?? "Cette macro ne peut pas être lue."}</p>
              {!selected.protected && (
                <div className="macro-actions-menu">
                  <button className="text-button" type="button" disabled={!online || active || busy !== null} onClick={() => setDialog({ kind: "rename", macro: selected })}><Icon name="edit" size={15} />Renommer</button>
                  <span className="action-divider" />
                  <button className="text-button danger-text" type="button" disabled={!online || active || busy !== null} onClick={() => setDialog({ kind: "delete", macro: selected })}><Icon name="trash" size={15} />Supprimer</button>
                </div>
              )}
            </div>

            {!selected.readable && <div className="inline-alert error"><Icon name="activity" size={17} /><div><strong>La séquence ne peut pas être lue</strong><span>{selected.issue ?? "Vérifiez le fichier de cette macro ou demandez de l’aide."}</span></div></div>}
            {detailError && <div className="inline-alert error"><Icon name="activity" size={17} /><div><strong>La séquence n’a pas pu être chargée</strong><span>{detailError}</span></div></div>}

            <section className="sequence-section" aria-labelledby="sequence-title">
              <div className="section-heading-row">
                <div><span className="section-kicker">Trace des actions</span><h2 id="sequence-title">La séquence</h2></div>
                <button className="quiet-button" type="button" disabled={!online || !selected.readable || active || busy !== null} onClick={() => run("record", () => api.startRecording(), "Enregistrement démarré.")}><Icon name="record" size={14} />Enregistrer</button>
              </div>
              <p className="recording-note">La préparation dure 3 s. À l’arrêt, les 3 dernières secondes sont retirées; une capture de 3 s ou moins laisse la macro vide.</p>
              <div className="sequence-panel">
                {recordingSelectedMacro ? <div className="recording-progress"><span className={`recording-mark ${snapshot.status.kind === "recording" ? snapshot.status.phase : ""}`}><Icon name="record" size={17} /></span><div><strong>{snapshot.status.kind === "recording" && snapshot.status.phase === "preparing" ? `Préparation · ${snapshot.status.countdownSeconds} s` : "Capture des actions en cours"}</strong><span>La séquence sera actualisée à l’arrêt de l’enregistrement.</span></div></div> : selected.readable ? detailLoading ? <div className="sequence-loading" role="status" aria-label="Chargement de la séquence"><span /><span /><span /></div> : detailError ? <div className="sequence-failed">Impossible de charger le détail des événements.</div> : <Sequence key={selected.name} detail={detail} summary={selected} /> : null}
              </div>
            </section>

            <section className="run-section" aria-label="Commandes de lecture">
              <div className="run-copy">
                <span className="section-kicker">Exécution</span>
                <strong>{snapshot.status.kind === "recording" ? `Enregistrement de « ${snapshot.status.macroName} »` : snapshot.status.kind === "playing" ? `Lecture de « ${snapshot.status.macroName} »` : "Prêt à lancer"}</strong>
                <span>{snapshot.status.kind === "recording" && snapshot.status.phase === "preparing" ? `La capture commence dans ${snapshot.status.countdownSeconds} s.` : snapshot.status.kind !== "idle" ? `Temps écoulé : ${formatDuration(snapshot.status.elapsedSeconds)}` : selected.eventCount ? "Les actions seront rejouées dans leur ordre enregistré." : "Enregistrez une séquence avant de la lancer."}</span>
              </div>
              <div className="run-controls">
                {snapshot.status.kind === "recording" ? (
                  <button className="button copper" type="button" onClick={() => run("stop-recording", () => api.stopRecording(), "Enregistrement terminé. La séquence a été mise à jour.")} disabled={!online || busy !== null}><Icon name="stop" size={15} />Arrêter l’enregistrement</button>
                ) : snapshot.status.kind === "playing" ? (
                  <button className="button stop-button" type="button" onClick={() => run("stop-playback", () => api.stopPlayback(), "Lecture arrêtée.")} disabled={!online || busy !== null}><Icon name="stop" size={15} />Arrêter la lecture</button>
                ) : (
                  <button className="button primary play-button" type="button" onClick={() => run("play", () => api.startPlayback(), "Lecture terminée.")} disabled={!online || !selected.readable || !selected.eventCount || busy !== null}><Icon name="play" size={15} />Lire la macro</button>
                )}
              </div>
            </section>

            <div className="macro-facts" aria-label="Informations sur la macro">
              <div><span>Événements</span><strong>{selected.eventCount.toLocaleString("fr-FR")}</strong></div>
              <div><span>Durée enregistrée</span><strong>{formatDuration(selected.durationSeconds)}</strong></div>
              <label className={`loop-setting ${active ? "is-muted" : ""}`}>
                <span className="loop-text"><span>Boucle</span><small>Répéter la lecture</small></span>
                <input type="checkbox" checked={snapshot.settings.loop} disabled={!online || active || busy !== null} onChange={(event) => run("loop", () => api.updateSettings({ loop: event.target.checked }), event.target.checked ? "Boucle activée." : "Boucle désactivée.")} />
              </label>
            </div>
            <div className="session-facts" aria-label="Statistiques de la session locale">
              <span>{snapshot.status.kind === "playing" ? "Lecture en cours" : "Dernière session de lecture"}</span>
              <div><strong>{snapshot.session.cycles.toLocaleString("fr-FR")}</strong><small>cycles</small></div>
              <div><strong>{formatDuration(snapshot.session.elapsedSeconds)}</strong><small>temps de session</small></div>
            </div>
            <p className="keyboard-note"><Icon name="keyboard" size={14} />Raccourcis globaux disponibles dans les paramètres.</p>
          </>
        )}
      </section>
      {dialog && <MacroDialog dialog={dialog} busy={!online || busy === dialog.kind || busy === "rename" || busy === "delete" || busy === "create"} onClose={() => setDialog(null)} onSubmit={submitDialog} />}
    </main>
    </>
  );
}

function statusLabel(status: Snapshot["status"]): string {
  if (status.kind === "recording") return status.phase === "preparing" ? "Préparation" : "Enregistrement";
  return status.kind === "playing" ? "Lecture en cours" : "En attente";
}
