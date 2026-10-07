import { useCallback, useEffect, useRef, useState } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { BackendClientError, type Snapshot } from "./api";
import { createBackendClient } from "./backend";
import { Atelier } from "./components/Atelier";
import { Help } from "./components/Help";
import { ActionErrorFeedback } from "./components/ActionErrorFeedback";
import { Icon } from "./components/Icon";
import { Settings } from "./components/Settings";

type View = "atelier" | "settings" | "help";
type ConnectionState = "connecting" | "connected" | "disconnected";

function viewFromPath(path: string): View {
  if (path.startsWith("/settings")) return "settings";
  if (path.startsWith("/help")) return "help";
  return "atelier";
}

function pathForView(view: View): string {
  return view === "settings" ? "/settings" : view === "help" ? "/help" : "/macros";
}

function hasRevision(value: unknown): value is Snapshot {
  return typeof value === "object" && value !== null && "revision" in value && typeof value.revision === "number" && "macros" in value && Array.isArray(value.macros);
}

function snapshotFrom(value: unknown): Snapshot | null {
  if (hasRevision(value)) return value;
  if (typeof value === "object" && value !== null && "snapshot" in value && hasRevision(value.snapshot)) return value.snapshot;
  return null;
}

function statusText(snapshot: Snapshot): string {
  if (snapshot.status.kind === "recording") return "Enregistrement en cours";
  if (snapshot.status.kind === "playing") return "Lecture en cours";
  return "Prêt";
}

export default function App() {
  if (!isTauri()) {
    return (
      <main className="app-frame">
        <section className="page-loading" role="alert">
          <h1>AUTO-COC est une application de bureau</h1>
          <p>Ouvrez AUTO-COC depuis son application de bureau. Pour le développement, exécutez <code>cargo tauri dev</code>.</p>
        </section>
      </main>
    );
  }
  return <DesktopApp />;
}

function DesktopApp() {
  const [api] = useState(() => createBackendClient());
  const [view, setView] = useState<View>(() => viewFromPath(window.location.pathname));
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [connectionError, setConnectionError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState("");
  const [notice, setNotice] = useState("");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [onboardingDismissed, setOnboardingDismissed] = useState(false);
  const mobileMenuButton = useRef<HTMLButtonElement>(null);
  const mobileDrawer = useRef<HTMLElement>(null);
  const mobileDrawerWasOpen = useRef(false);

  const connect = useCallback(async (signal?: AbortSignal) => {
    return api.connect(signal);
  }, [api]);
  const revision = snapshot?.revision;

  useEffect(() => () => api.dispose(), [api]);
  useEffect(() => api.listenShutdownErrors((error) => {
    setNotice("");
    setActionError(error.message);
  }), [api]);
  useEffect(() => {
    if (snapshot?.lastError) setActionError(snapshot.lastError);
  }, [snapshot?.lastError]);

  useEffect(() => {
    if (window.location.pathname === "/") window.history.replaceState({}, "", "/macros");
    const onPopState = () => setView(viewFromPath(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setConnection("connecting");
    connect(controller.signal).then((initial) => {
      if (controller.signal.aborted) return;
      setSnapshot(initial);
      setConnection("connected");
      setConnectionError("");
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setConnection("disconnected");
      setConnectionError(error instanceof Error ? error.message : "AUTO-COC ne répond pas.");
    });
    return () => controller.abort();
  }, [connect]);

  useEffect(() => {
    if (revision === undefined) return;
    const controller = new AbortController();
    const { signal } = controller;
    let currentRevision = revision;
    let retryTimer: number | undefined;

    const waitBeforeRetry = () => new Promise<void>((resolve) => {
      const finish = () => {
        if (retryTimer !== undefined) window.clearTimeout(retryTimer);
        signal.removeEventListener("abort", finish);
        resolve();
      };
      retryTimer = window.setTimeout(finish, 1200);
      signal.addEventListener("abort", finish, { once: true });
    });

    async function listen() {
      while (!signal.aborted) {
        try {
          const updated = await api.waitForSnapshot(currentRevision, signal);
          currentRevision = updated.revision;
          setSnapshot((current) => current && updated.revision < current.revision ? current : updated);
          setConnection("connected");
          setConnectionError("");
        } catch (error) {
          if (signal.aborted) return;
          setConnection("disconnected");
          setConnectionError(error instanceof Error ? error.message : "La connexion locale a été interrompue.");
          await waitBeforeRetry();
        }
      }
    }
    void listen();
    return () => { controller.abort(); if (retryTimer !== undefined) window.clearTimeout(retryTimer); };
  }, [api, connect, revision]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 3600);
    return () => window.clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (!mobileNavOpen) {
      if (mobileDrawerWasOpen.current) {
        mobileDrawerWasOpen.current = false;
        const trigger = mobileMenuButton.current;
        if (trigger?.getClientRects().length) trigger.focus();
        else document.querySelector<HTMLElement>("#main-content h1")?.focus({ preventScroll: true });
      }
      return;
    }
    mobileDrawerWasOpen.current = true;
    const drawer = mobileDrawer.current;
    if (!drawer) return;
    const viewport = window.matchMedia("(max-width: 850px)");
    const onViewportChange = () => { if (!viewport.matches) setMobileNavOpen(false); };
    const getFocusable = () => Array.from(drawer.querySelectorAll<HTMLElement>(
      'a[href], button:not(:disabled), input:not(:disabled), [tabindex]:not([tabindex="-1"])',
    )).filter((element) => element.getClientRects().length > 0);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setMobileNavOpen(false);
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = getFocusable();
      const first = focusable[0];
      const last = focusable.at(-1);
      if (!first || !last) { event.preventDefault(); return; }
      if (event.shiftKey && (document.activeElement === first || !drawer.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !drawer.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    viewport.addEventListener("change", onViewportChange);
    window.requestAnimationFrame(() => {
      if (drawer.classList.contains("mobile-open")) getFocusable()[0]?.focus();
    });
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      viewport.removeEventListener("change", onViewportChange);
    };
  }, [mobileNavOpen]);

  useEffect(() => {
    setMobileNavOpen(false);
    window.requestAnimationFrame(() => document.querySelector<HTMLElement>("#main-content h1")?.focus({ preventScroll: true }));
  }, [view]);

  function navigate(next: View) {
    setMobileNavOpen(false);
    if (next !== view) window.history.pushState({}, "", pathForView(next));
    setView(next);
  }

  const run = useCallback(async <T,>(label: string, action: () => Promise<T>, message?: string): Promise<T | undefined> => {
    if (connection !== "connected") {
      setActionError("AUTO-COC ne répond pas. Les commandes sont temporairement indisponibles.");
      return undefined;
    }
    setBusy(label);
    setActionError("");
    setNotice("");
    try {
      const value = await action();
      const updated = snapshotFrom(value);
      if (updated) {
        setSnapshot((current) => current && updated.revision < current.revision ? current : updated);
      }
      if (message) setNotice(message);
      return value;
    } catch (error) {
      const messageText = error instanceof Error ? error.message : "L’action n’a pas abouti.";
      setActionError(messageText);
      if (error instanceof BackendClientError && (
        error.status === 409
        || error.code === "conflict"
        || error.code === "invalid_state"
        || error.code === "shortcut_unavailable"
      )) {
        try {
          const latest = await api.getSnapshot();
          setSnapshot((current) => current && latest.revision < current.revision ? current : latest);
        } catch { /* The original conflict remains the useful message. */ }
      }
      return undefined;
    } finally {
      setBusy(null);
    }
  }, [api, connection]);

  function resumeOnboarding() {
    setOnboardingDismissed(false);
    navigate("atelier");
  }

  function dismissOnboarding() {
    setOnboardingDismissed(true);
    window.requestAnimationFrame(() => document.querySelector<HTMLElement>("#main-content h1")?.focus({ preventScroll: true }));
  }

  async function completeOnboarding() {
    const updated = await run("onboarding", () => api.completeOnboarding(), "Guide de démarrage terminé.");
    if (updated) window.requestAnimationFrame(() => document.querySelector<HTMLElement>("#main-content h1")?.focus({ preventScroll: true }));
  }
  const online = connection === "connected";

  const macroState = snapshot ? statusText(snapshot) : connection === "connected" ? "Chargement" : "AUTO-COC indisponible";
  const pageLabel = view === "atelier" ? "Atelier" : view === "settings" ? "Réglages" : "Aide";

  return (
    <div className="app-frame">
      <a className="skip-link" href="#main-content">Aller au contenu</a>
      <header className="mobile-topbar">
        <button className="brand-lockup" type="button" onClick={() => navigate("atelier")} aria-label="AUTO-COC, ouvrir l’atelier"><BrandMark /><span>AUTO-COC</span></button>
        <button ref={mobileMenuButton} className="icon-button subtle" type="button" aria-label={mobileNavOpen ? "Fermer le menu" : "Ouvrir le menu"} aria-expanded={mobileNavOpen} aria-controls="primary-navigation" onClick={() => setMobileNavOpen((open) => !open)}><Icon name={mobileNavOpen ? "close" : "grid"} /></button>
      </header>
      {mobileNavOpen && <button className="nav-backdrop" type="button" aria-label="Fermer le menu" onClick={() => setMobileNavOpen(false)} />}
      <aside ref={mobileDrawer} className={`sidebar ${mobileNavOpen ? "mobile-open" : ""}`} id="primary-navigation" role={mobileNavOpen ? "dialog" : undefined} aria-label={mobileNavOpen ? "Menu principal" : undefined} aria-modal={mobileNavOpen ? true : undefined}>
        <button className="brand-lockup desktop-brand" type="button" onClick={() => navigate("atelier")} aria-label="AUTO-COC, ouvrir l’atelier"><BrandMark /><span>AUTO-COC</span></button>
        <span className="sidebar-caption">Votre atelier</span>
        <nav className="primary-nav" aria-label="Navigation principale">
          <NavButton view="atelier" current={view} onNavigate={navigate} icon="grid">Atelier</NavButton>
          <NavButton view="settings" current={view} onNavigate={navigate} icon="sliders">Réglages</NavButton>
          <NavButton view="help" current={view} onNavigate={navigate} icon="help">Aide</NavButton>
        </nav>
        <div className="sidebar-rule" />
        <div className="sidebar-library-label"><span>Bibliothèque</span><span className="library-count">{snapshot?.macros.length ?? "—"}</span></div>
        {snapshot && snapshot.macros.length > 0 ? (
          <nav className="sidebar-macros" aria-label="Accès aux macros">
            {snapshot.macros.slice(0, 5).map((macro) => <button key={macro.name} type="button" className={`sidebar-macro ${macro.name === snapshot.selectedMacro ? "selected" : ""}`} onClick={() => { navigate("atelier"); void run("select", () => api.selectMacro(macro.name)); }} disabled={!online || snapshot.status.kind !== "idle" || busy !== null}><span aria-hidden="true" />{macro.name}{macro.protected && <Icon name="lock" size={11} />}</button>)}
            {snapshot.macros.length > 5 && <button className="sidebar-more" type="button" onClick={() => navigate("atelier")}>Voir les {snapshot.macros.length - 5} autres</button>}
          </nav>
        ) : <p className="sidebar-empty">Vos macros apparaîtront ici.</p>}
        <div className="sidebar-spacer" />
        <button className="local-status" type="button" onClick={() => navigate("help")} aria-label={`État d’AUTO-COC : ${connection === "connected" ? macroState : connection === "connecting" ? "démarrage en cours" : "ne répond pas"}`}>
          <span className={`connection-mark ${connection}`}><i /></span><span className="local-status-copy"><strong>{connection === "connected" ? macroState : connection === "connecting" ? "Démarrage en cours" : "AUTO-COC ne répond pas"}</strong><small>{connection === "connected" ? "Ce PC · local" : connection === "connecting" ? "Démarrage en cours" : "Voir l’aide"}</small></span><Icon name="chevron" size={14} />
        </button>
        <div className="sidebar-version">AUTO-COC <span>·</span> outil local</div>
      </aside>

      <div className="page-shell" id="main-content" aria-label={pageLabel} tabIndex={-1}>
        {(connection === "disconnected" || connection === "connecting") && !snapshot && (
          <div className={`connection-banner ${connection}`} role={connection === "disconnected" ? "alert" : "status"}>
            <span className="connection-mark"><i /></span><div><strong>{connection === "disconnected" ? "AUTO-COC ne répond pas" : "Démarrage en cours…"}</strong><span>{connection === "disconnected" ? connectionError || "Relancez AUTO-COC depuis son raccourci." : "L’atelier et ses commandes apparaîtront une fois le démarrage terminé."}</span></div>
            {connection === "disconnected" && <button className="text-button" type="button" onClick={() => { setConnection("connecting"); void connect().then((value) => { setSnapshot(value); setConnection("connected"); setConnectionError(""); }).catch((error: unknown) => { setConnection("disconnected"); setConnectionError(error instanceof Error ? error.message : "Connexion impossible."); }); }}>Réessayer<Icon name="refresh" size={14} /></button>}
          </div>
        )}
        {actionError && <ActionErrorFeedback message={actionError} onDismiss={() => setActionError("")} />}
        {notice && <div className="action-feedback success" role="status"><Icon name="check" size={16} /><span>{notice}</span></div>}
        {connection === "disconnected" && snapshot && <div className="reconnecting-strip" role="status"><span className="connection-mark disconnected"><i /></span>AUTO-COC ne répond pas. Les commandes sont temporairement indisponibles.</div>}

        {snapshot && view === "atelier" && <Atelier api={api} snapshot={snapshot} busy={busy} run={run} online={online} showOnboarding={!snapshot.onboardingComplete && !onboardingDismissed} onDismissOnboarding={dismissOnboarding} onCompleteOnboarding={() => { void completeOnboarding(); }} />}
        {snapshot && view === "settings" && <Settings api={api} snapshot={snapshot} busy={busy} run={run} online={online} />}
        {view === "help" && <Help api={api} snapshot={snapshot} busy={busy} run={run} navigate={navigate} online={online} onResumeOnboarding={resumeOnboarding} onCompleteOnboarding={() => { void completeOnboarding(); }} />}
        {!snapshot && connection !== "disconnected" && <div className="page-loading" role="status"><div className="loading-trace"><i /><i /><i /></div><span>Chargement de votre atelier</span></div>}
      </div>
    </div>
  );
}

function NavButton({ view, current, onNavigate, icon, children }: {
  view: View;
  current: View;
  onNavigate: (view: View) => void;
  icon: "grid" | "sliders" | "help";
  children: string;
}) {
  return <button className={`primary-nav-item ${current === view ? "active" : ""}`} type="button" aria-current={current === view ? "page" : undefined} onClick={() => onNavigate(view)}><Icon name={icon} size={17} />{children}{current === view && <span className="nav-indicator" />}</button>;
}

function BrandMark() {
  return <span className="brand-mark" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><path d="M5 7.5 16 3l11 4.5v13L16 29 5 20.5z"/><path d="M10 10.5 16 8l6 2.5v8L16 23l-6-4.5z"/><path d="m16 12 3 2v3.5l-3 2-3-2V14z"/></svg></span>;
}
