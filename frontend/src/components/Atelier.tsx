import { useEffect, useMemo, useState } from "react";
import type { ActionRunner, BackendClient, MacroDetail, MacroSummary, Snapshot } from "../api";
import { Icon } from "./Icon";
import { MacroDialog, type MacroDialogState } from "./MacroDialog";
import { OnboardingGuide } from "./OnboardingGuide";
import { formatCountdown, formatDuration, formatElapsedDuration } from "./formatDuration";
import { getOnboardingStage } from "../onboardingStage";
import { getPlaybackPosition, getTimelineMarkers, getTraceNodeClass } from "./sequence";

type DialogState = MacroDialogState | null;
type PageProps = { api: BackendClient; snapshot: Snapshot; busy: string | null; run: ActionRunner; online: boolean; showOnboarding: boolean; onDismissOnboarding: () => void; onCompleteOnboarding: () => void; onClearMacroSearch?: () => void };

const eventNames: Record<string, string> = {
  mouse_move: "Déplacement",
  mouse_click: "Clic",
  scroll: "Défilement",
  key_down: "Touche enfoncée",
  key_up: "Touche relâchée",
  nop: "Pause",
};

export function Sequence({ detail, summary, playback }: {
  detail: MacroDetail | null;
  summary: MacroSummary;
  playback?: { elapsedSeconds: number; isRunning: boolean; looped: boolean };
}) {
  const timelineMarkers = useMemo(() => detail ? getTimelineMarkers(detail.steps) : [], [detail]);
  const playbackPosition = playback
    ? getPlaybackPosition(playback.elapsedSeconds, summary.durationSeconds, playback.looped)
    : null;
  const traceCounts = useMemo(() => timelineMarkers.reduce((counts, { step }) => {
    if (step.type === "mouse_move") counts.movement += 1;
    else if (step.type === "mouse_click") counts.mouse += 1;
    else if (step.type.startsWith("key")) counts.keyboard += 1;
    else counts.other += 1;
    return counts;
  }, { movement: 0, mouse: 0, keyboard: 0, other: 0 }), [timelineMarkers]);
  const traceDescription = `Chronologie condensée, de gauche à droite. Repères affichés : ${traceCounts.movement} fin${traceCounts.movement === 1 ? "" : "s"} de déplacement, ${traceCounts.mouse} clic${traceCounts.mouse === 1 ? "" : "s"} souris, ${traceCounts.keyboard} événement${traceCounts.keyboard === 1 ? "" : "s"} clavier et ${traceCounts.other} autre${traceCounts.other === 1 ? "" : "s"} action${traceCounts.other === 1 ? "" : "s"}.`;
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

  return (
    <div className="sequence-wrap">
      <div className="trace-heading">
        <span>Déroulé réel</span>
        <span>{playbackPosition
          ? `${formatDuration(playbackPosition.elapsedSeconds)} / ${formatDuration(summary.durationSeconds)}`
          : `${timelineMarkers.length} repères`}</span>
      </div>
      <div
        className="trace-track"
        role="img"
        aria-label={playbackPosition
          ? `${traceDescription} Lecture à ${formatDuration(playbackPosition.elapsedSeconds)} sur ${formatDuration(summary.durationSeconds)}.`
          : traceDescription}
      >
        {timelineMarkers.map(({ step, index, left }) => <span key={`${index}-${step.t}-${step.type}`} className={`trace-node ${getTraceNodeClass(step.type)}`} style={{ left }} title={`${index + 1}. ${eventNames[step.type] ?? step.type}`} />)}
        {playbackPosition && <span
          key={playbackPosition.cycle}
          className={playback?.isRunning ? "trace-playhead is-running" : "trace-playhead"}
          style={{ left: `${playbackPosition.percent}%` }}
          aria-hidden="true"
        />}
      </div>
      <ul className="trace-legend" aria-label="Légende de la trace">
        <li><span className="trace-legend-swatch is-move" aria-hidden="true" />Fin d’un déplacement</li>
        <li><span className="trace-legend-swatch is-click" aria-hidden="true" />Clic souris</li>
        <li><span className="trace-legend-swatch is-key" aria-hidden="true" />Touche clavier</li>
        <li><span className="trace-legend-swatch is-other" aria-hidden="true" />Pause ou défilement</li>
      </ul>
    </div>
  );
}

export function Atelier({ api, snapshot, busy, run, online, showOnboarding, onDismissOnboarding, onCompleteOnboarding, onClearMacroSearch }: PageProps) {
  const [dialog, setDialog] = useState<DialogState>(null);
  const [detail, setDetail] = useState<MacroDetail | null>(null);
  const [detailError, setDetailError] = useState("");
  const [detailLoading, setDetailLoading] = useState(false);
  const selected = snapshot.macros.find((macro) => macro.name === snapshot.selectedMacro) ?? null;
  const active = snapshot.status.kind !== "idle";
  const playbackStatus = snapshot.status.kind === "playing" || snapshot.status.kind === "paused"
    ? snapshot.status
    : snapshot.status.kind === "waiting_for_foreground" && snapshot.status.action === "resuming"
      ? snapshot.status
      : null;
  const timelinePlayback = selected && playbackStatus?.macroName === selected.name
    ? {
      elapsedSeconds: snapshot.session.elapsedSeconds,
      isRunning: snapshot.status.kind === "playing",
      looped: snapshot.settings.loop,
    }
    : undefined;
  const waitingForSelectedRecording = snapshot.status.kind === "waiting_for_foreground"
    && snapshot.status.action === "recording"
    && snapshot.status.macroName === selected?.name;
  const recordingSelectedMacro = (snapshot.status.kind === "recording" && snapshot.status.macroName === selected?.name)
    || waitingForSelectedRecording;
  const recordingProgressPhase = snapshot.status.kind === "recording" ? snapshot.status.phase : waitingForSelectedRecording ? "waiting" : "";
  const recordingProgressTitle = waitingForSelectedRecording
    ? "En attente de Clash of Clans"
    : snapshot.status.kind === "recording" && snapshot.status.phase === "preparing"
      ? `Préparation · ${formatCountdown(snapshot.status.countdownSeconds)} s`
      : "Capture des actions en cours";
  const recordingProgressDetail = snapshot.status.kind === "waiting_for_foreground"
    && snapshot.status.action === "recording"
    && snapshot.status.macroName === selected?.name
    ? "La préparation de 3 s commencera quand la fenêtre du jeu sera reconnue. Cette attente expire automatiquement au bout de 30 secondes."
    : "La séquence sera actualisée à l’arrêt de l’enregistrement.";
  const executionCopy = getExecutionCopy(snapshot.status, selected);
  const onboardingStage = getOnboardingStage(snapshot);

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

  function cancelForegroundWait() {
    if (snapshot.status.kind !== "waiting_for_foreground") return;
    if (snapshot.status.action === "resuming") {
      return run("cancel-resume-wait", () => api.cancelResumeWait(), "Reprise annulée.");
    }
    if (snapshot.status.action === "recording") {
      return run("cancel-recording-wait", () => api.stopRecording(), "Attente annulée.");
    }
    return run("cancel-playback-wait", () => api.stopPlayback(), "Attente annulée.");
  }

  return (
    <>
    {showOnboarding && !active && <OnboardingGuide
      mode="atelier"
      stage={onboardingStage}
      disabled={!online || busy !== null}
      busy={busy === "onboarding"}
      onPrimaryAction={() => {
        if (onboardingStage === "create") setDialog({ kind: "create" });
        else if (onboardingStage === "select") {
          const targetMacro = snapshot.macros.find((macro) => macro.readable && macro.name !== selected?.name)
            ?? snapshot.macros[0];
          if (targetMacro) {
            onClearMacroSearch?.();
            window.requestAnimationFrame(() => document.getElementById(`macro-${encodeURIComponent(targetMacro.name)}`)?.focus());
          }
        } else if (onboardingStage === "record") void run("record", () => api.startRecording());
        else document.getElementById("play-macro-button")?.focus();
      }}
      onComplete={onCompleteOnboarding}
      onDismiss={onDismissOnboarding}
    />}
    <main className="workbench">
      <section className="macro-workspace" aria-label="Détail de la macro">
        {!selected ? (
          <div className="welcome-empty">
            <div className="empty-trace" aria-hidden="true"><span /><span /><span /><span /><i /><span /></div>
            <span className="section-kicker">L’atelier commence ici</span>
            <h1 tabIndex={-1}>{snapshot.macros.length ? "Choisissez une macro" : "Créez votre première macro"}</h1>
            <p>{snapshot.macros.length ? "Sélectionnez une macro dans la bibliothèque pour voir sa séquence et la lancer." : "Enregistrez une suite d’actions, puis rejouez-la quand vous en avez besoin."}</p>
            {!snapshot.macros.length && !showOnboarding && <div className="empty-tip"><Icon name="plus" size={16} />Utilisez le bouton + dans la bibliothèque pour commencer.</div>}
            {snapshot.macros.length > 0 && <div className="empty-tip"><Icon name="keyboard" size={16} />Parcourez la liste avec les flèches du clavier, puis Entrée pour choisir.</div>}
          </div>
        ) : (
          <>
            <div className="workspace-topbar">
              <div className="breadcrumbs"><span>Atelier</span><Icon name="chevron" size={13} /><span>Macro</span></div>
              <div className="workspace-tools">
                <span className={`state-label state-${snapshot.status.kind}`}><i aria-hidden="true" />{statusLabel(snapshot.status)}</span>
              </div>
            </div>
            <div className="macro-heading">
              <div className="macro-title-line">
                <h1 tabIndex={-1}>{selected.name}</h1>
              </div>
              {!selected.readable && <p>{selected.issue ?? "Cette macro ne peut pas être lue."}</p>}
              <div className="macro-actions-menu">
                <button className="text-button" type="button" disabled={!online || active || busy !== null} onClick={() => setDialog({ kind: "rename", macro: selected })}><Icon name="edit" size={15} />Renommer</button>
                <span className="action-divider" />
                <button className="text-button danger-text" type="button" disabled={!online || active || busy !== null} onClick={() => setDialog({ kind: "delete", macro: selected })}><Icon name="trash" size={15} />Supprimer</button>
              </div>
            </div>

            {!selected.readable && <div className="inline-alert error"><Icon name="activity" size={17} /><div><strong>La séquence ne peut pas être lue</strong><span>{selected.issue ?? "Vérifiez le fichier de cette macro ou demandez de l’aide."}</span></div></div>}
            {detailError && <div className="inline-alert error"><Icon name="activity" size={17} /><div><strong>La séquence n’a pas pu être chargée</strong><span>{detailError}</span></div></div>}

            <section className="sequence-section" aria-labelledby="sequence-title">
              <div className="section-heading-row">
                <div><span className="section-kicker">Trace des actions</span><h2 id="sequence-title">La séquence</h2></div>
                <button className="quiet-button" type="button" disabled={!online || !selected.readable || active || busy !== null} onClick={() => run("record", () => api.startRecording(), "Demande d’enregistrement envoyée.")}><Icon name="record" size={14} />Enregistrer</button>
              </div>
              <p className="recording-note">Après le compte à rebours, laissez l’enregistrement tourner plus de 3 s avant de l’arrêter. Les 3 dernières secondes seront retirées.</p>
              <div className="sequence-panel">
                {!recordingSelectedMacro && selected.readable && <div className="sequence-summary" role="group" aria-label="Informations sur la macro">
                  <div className="sequence-summary-item"><span>Événements</span><strong>{selected.eventCount.toLocaleString("fr-FR")}</strong></div>
                  <div className="sequence-summary-item"><span>Durée enregistrée</span><strong>{formatDuration(selected.durationSeconds)}</strong></div>
                </div>}
                {recordingSelectedMacro ? (
                  <div className="recording-progress">
                    <span className={`recording-mark ${recordingProgressPhase}`}><Icon name="record" size={17} /></span>
                    <div><strong>{recordingProgressTitle}</strong><span>{recordingProgressDetail}</span></div>
                  </div>
                ) : selected.readable ? detailLoading ? <div className="sequence-loading" role="status" aria-label="Chargement de la séquence"><span /><span /><span /></div> : detailError ? <div className="sequence-failed">Impossible de charger le détail des événements.</div> : <Sequence key={selected.name} detail={detail} summary={selected} playback={timelinePlayback} /> : null}
              </div>
            </section>

            <section className="run-section" aria-label="Commandes de lecture">
              <div
                className="run-copy"
                role={snapshot.status.kind === "waiting_for_foreground" || snapshot.status.kind === "paused" ? "status" : undefined}
                aria-atomic={snapshot.status.kind === "waiting_for_foreground" || snapshot.status.kind === "paused" ? "true" : undefined}
              >
                <span className="section-kicker">Exécution</span>
                <strong>{executionCopy.title}</strong>
                <span>{executionCopy.detail}</span>
              </div>
              <div className="run-controls">
                {snapshot.status.kind === "waiting_for_foreground" ? (
                  snapshot.status.action === "resuming" ? (
                    <button className="button stop-button" type="button" onClick={() => { void cancelForegroundWait(); }} disabled={!online || busy !== null}><Icon name="stop" size={15} />Annuler la reprise</button>
                  ) : (
                    <button className="button stop-button" type="button" onClick={() => { void cancelForegroundWait(); }} disabled={!online || busy !== null}><Icon name="stop" size={15} />Annuler l’attente</button>
                  )
                ) : snapshot.status.kind === "recording" ? (
                  <button className="button copper" type="button" onClick={() => run("stop-recording", () => api.stopRecording(), "Enregistrement terminé. La séquence a été mise à jour.")} disabled={!online || busy !== null}><Icon name="stop" size={15} />Arrêter l’enregistrement</button>
                ) : snapshot.status.kind === "playing" ? (
                  <button className="button stop-button" type="button" onClick={() => run("stop-playback", () => api.stopPlayback(), "Lecture arrêtée.")} disabled={!online || busy !== null}><Icon name="stop" size={15} />Arrêter la lecture</button>
                ) : snapshot.status.kind === "paused" ? (
                  <>
                    <button className="button primary play-button" type="button" onClick={() => run("resume-playback", () => api.startPlayback(), "Reprise de la lecture demandée.")} disabled={!online || busy !== null}><Icon name="play" size={15} />Reprendre</button>
                    <button className="button stop-button" type="button" onClick={() => run("stop-playback", () => api.stopPlayback(), "Lecture arrêtée.")} disabled={!online || busy !== null}><Icon name="stop" size={15} />Arrêter la lecture</button>
                  </>
                ) : (
                  <button id="play-macro-button" className="button primary play-button" type="button" onClick={() => run("play", () => api.startPlayback(), "Demande de lecture envoyée.")} disabled={!online || !selected.readable || !selected.eventCount || busy !== null}><Icon name="play" size={15} />Lire la macro</button>
                )}
              </div>
              {snapshot.status.kind === "idle" && selected.readable && selected.eventCount > 0 && <p className="play-safety-note"><Icon name="shield" size={14} />{snapshot.settings.allowCocBackgroundPlayback ? "Clash of Clans doit être ouvert; le replay reste ciblé sur sa fenêtre." : "Pour lire, Clash of Clans doit être la fenêtre active."}</p>}
            </section>

            <div className="macro-options" role="group" aria-label="Options de la macro">
              <label className={`macro-option ${active ? "is-muted" : ""}`}>
                <span className="loop-text"><span>Boucle</span><small>Répéter la lecture</small></span>
                <input type="checkbox" checked={snapshot.settings.loop} disabled={!online || active || busy !== null} onChange={(event) => run("loop", () => api.updateSettings({ loop: event.target.checked }), event.target.checked ? "Boucle activée." : "Boucle désactivée.")} />
              </label>
              <label className={`macro-option ${active ? "is-muted" : ""}`}>
                <span className="loop-text"><span>Pause hors CoC</span><small>Quand le jeu perd le focus</small></span>
                <input
                  type="checkbox"
                  checked={!snapshot.settings.allowCocBackgroundPlayback}
                  disabled={!online || active || busy !== null}
                  onChange={(event) => run(
                    "focus-guard",
                    () => api.updateSettings({ allowCocBackgroundPlayback: !event.target.checked }),
                    event.target.checked ? "Pause hors CoC activée." : "Replay en arrière-plan activé.",
                  )}
                />
              </label>
            </div>
            <div className="session-facts" aria-label="Statistiques de la session">
              <span>{snapshot.status.kind === "waiting_for_foreground" && snapshot.status.action === "resuming" ? "Reprise en attente" : snapshot.status.kind === "waiting_for_foreground" && snapshot.status.action === "playing" ? "Lecture en attente" : snapshot.status.kind === "playing" ? "Lecture en cours" : snapshot.status.kind === "paused" ? "Lecture en pause" : "Dernière session de lecture"}</span>
              <div><strong>{snapshot.session.cycles.toLocaleString("fr-FR")}</strong><small>cycles</small></div>
              <div><strong>{formatElapsedDuration(snapshot.session.elapsedSeconds)}</strong><small>temps de session</small></div>
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
  if (status.kind === "waiting_for_foreground") return status.action === "resuming" ? "Reprise en attente" : "En attente du jeu";
  if (status.kind === "recording") return status.phase === "preparing" ? "Préparation" : "Enregistrement";
  if (status.kind === "paused") return "En pause";
  return status.kind === "playing" ? "Lecture en cours" : "En attente";
}

function getExecutionCopy(status: Snapshot["status"], selected: MacroSummary | null): { title: string; detail: string } {
  if (status.kind === "waiting_for_foreground") {
    if (status.action === "resuming") {
      return {
        title: `Reprise en attente pour « ${status.macroName} »`,
        detail: "Revenez dans Clash of Clans. La lecture reprendra dès que le jeu sera de nouveau au premier plan. Utilisez « Annuler la reprise » pour annuler cette demande.",
      };
    }
    const action = status.action === "recording" ? "Enregistrement" : "Lecture";
    return {
      title: `${action} en attente pour « ${status.macroName} »`,
      detail: "Aucune action ne démarre avant la détection de Clash of Clans. L’attente expire automatiquement au bout de 30 secondes; utilisez « Annuler l’attente » pour l’interrompre plus tôt.",
    };
  }
  if (status.kind === "recording") {
    return {
      title: `Enregistrement de « ${status.macroName} »`,
      detail: status.phase === "preparing"
        ? `La capture commence dans ${formatCountdown(status.countdownSeconds)} s.`
        : `Temps écoulé : ${formatElapsedDuration(status.elapsedSeconds)}`,
    };
  }
  if (status.kind === "playing") {
    return {
      title: `Lecture de « ${status.macroName} »`,
      detail: `Temps écoulé : ${formatElapsedDuration(status.elapsedSeconds)}`,
    };
  }
  if (status.kind === "paused") {
    return {
      title: "En pause, CoC n'est plus la fenêtre active",
      detail: `Lecture de « ${status.macroName} » suspendue à ${formatElapsedDuration(status.elapsedSeconds)}. Cliquez sur « Reprendre », puis revenez dans Clash of Clans. La lecture reprendra dès que le jeu sera de nouveau au premier plan.`,
    };
  }
  return {
    title: "Prêt à lancer",
    detail: selected?.eventCount
      ? "Les actions seront rejouées dans leur ordre enregistré."
      : "Enregistrez une séquence avant de la lancer.",
  };
}
