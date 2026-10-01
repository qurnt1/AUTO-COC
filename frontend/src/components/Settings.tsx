import { useEffect, useState } from "react";
import type { ActionRunner, ApiClient, PairingPreparation, ShutdownPreparation, Snapshot } from "../api";
import { Dialog } from "./Dialog";
import { Icon } from "./Icon";
import { normalizeShortcut } from "./normalizeShortcut";

type Props = { api: ApiClient; snapshot: Snapshot; busy: string | null; run: ActionRunner; online: boolean };
type SettingsTab = "general" | "shortcuts" | "telegram" | "system";
type ShortcutName = keyof Snapshot["settings"]["shortcuts"];
type PairingCode = Pick<PairingPreparation, "code" | "expiresAt">;

const shortcutLabels: Record<ShortcutName, string> = { toggle: "Basculer lecture / arrêt", play: "Lire la macro", stop: "Arrêter" };
const statusCopy: Record<Snapshot["settings"]["telegram"]["status"], string> = {
  not_configured: "Non configuré",
  disconnected: "Déconnecté",
  waiting_pairing: "En attente de l’appairage",
  connected: "Connecté",
  error: "Erreur de connexion",
};

function telegramTone(status: Snapshot["settings"]["telegram"]["status"]): string {
  return status === "connected" ? "good" : status === "error" ? "bad" : status === "waiting_pairing" ? "copper" : "neutral";
}

export function Settings({ api, snapshot, busy, run, online }: Props) {
  const [tab, setTab] = useState<SettingsTab>("general");
  const [path, setPath] = useState(snapshot.settings.cocPath);
  const [shortcuts, setShortcuts] = useState(snapshot.settings.shortcuts);
  const [capturing, setCapturing] = useState<ShortcutName | null>(null);
  const [shortcutError, setShortcutError] = useState("");
  const [token, setToken] = useState("");
  const [telegramMessage, setTelegramMessage] = useState("");
  const [pairing, setPairing] = useState<PairingCode | null>(snapshot.settings.telegram.pairingCode && snapshot.settings.telegram.pairingExpiresAt ? { code: snapshot.settings.telegram.pairingCode, expiresAt: snapshot.settings.telegram.pairingExpiresAt } : null);
  const [screenshotUrl, setScreenshotUrl] = useState("");
  const [shutdown, setShutdown] = useState<ShutdownPreparation | null>(null);
  const [showTokenRemoval, setShowTokenRemoval] = useState(false);
  const [shutdownError, setShutdownError] = useState("");

  useEffect(() => setPath(snapshot.settings.cocPath), [snapshot.settings.cocPath]);
  const { toggle, play, stop } = snapshot.settings.shortcuts;
  useEffect(() => setShortcuts({ toggle, play, stop }), [toggle, play, stop]);
  useEffect(() => {
    if (snapshot.settings.telegram.pairingCode && snapshot.settings.telegram.pairingExpiresAt) {
      setPairing({ code: snapshot.settings.telegram.pairingCode, expiresAt: snapshot.settings.telegram.pairingExpiresAt });
    } else if (!snapshot.settings.telegram.paired) setPairing(null);
  }, [snapshot.settings.telegram.pairingCode, snapshot.settings.telegram.pairingExpiresAt, snapshot.settings.telegram.paired]);
  useEffect(() => {
    if (!screenshotUrl) return;
    return () => URL.revokeObjectURL(screenshotUrl);
  }, [screenshotUrl]);

  useEffect(() => {
    if (!capturing) return;
    if (!online || snapshot.status.kind !== "idle" || busy !== null) {
      setCapturing(null);
      return;
    }
    const handle = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (event.key === "Escape") { setCapturing(null); setShortcutError(""); return; }
      const value = normalizeShortcut(event);
      if (!value) { setShortcutError("Choisissez une touche seule F1 à F12 ou une combinaison avec Ctrl, Alt ou Maj."); return; }
      setShortcuts((current) => ({ ...current, [capturing]: value }));
      setShortcutError("");
      setCapturing(null);
    };
    window.addEventListener("keydown", handle, true);
    return () => window.removeEventListener("keydown", handle, true);
  }, [capturing, online, snapshot.status.kind, busy]);

  const telegram = snapshot.settings.telegram;
  const shortcutsEditable = online && snapshot.status.kind === "idle" && busy === null;
  const changedShortcuts = (Object.keys(shortcuts) as ShortcutName[]).some((key) => shortcuts[key] !== snapshot.settings.shortcuts[key]);
  const duplicated = new Set(Object.values(shortcuts).map((value) => value.toLocaleLowerCase("fr"))).size !== Object.values(shortcuts).length;

  async function saveShortcuts() {
    if (!shortcutsEditable) return;
    if (duplicated) { setShortcutError("Chaque action doit utiliser une combinaison différente."); return; }
    const saved = await run("shortcuts", () => api.updateShortcuts(shortcuts), "Raccourcis enregistrés.");
    if (saved) setShortcutError("");
  }

  async function saveToken() {
    if (!token.trim()) return;
    const saved = await run("telegram-token", () => api.saveTelegramToken(token.trim()), "Token Telegram enregistré dans le stockage sécurisé.");
    if (saved) setToken("");
    setTelegramMessage(saved ? "Le token est enregistré. Il n’est plus conservé dans ce champ." : "Le token n’a pas été enregistré.");
  }

  async function startPairing() {
    const result = await run("telegram-pairing", () => api.startTelegramPairing(), "Code d’appairage créé.");
    if (result) setPairing(result);
  }

  async function copyPairingCode() {
    if (!pairing) return;
    try {
      await navigator.clipboard.writeText(`/start ${pairing.code}`);
      setTelegramMessage("Commande copiée. Envoyez-la dans une conversation privée avec le bot.");
    } catch {
      setTelegramMessage("Copie impossible dans ce navigateur. Sélectionnez la commande et copiez-la.");
    }
  }

  async function takeScreenshot() {
    const blob = await run("screenshot", () => api.getScreenshot(), "Capture réalisée sur cet ordinateur.");
    if (!blob) return;
    setScreenshotUrl((current) => { if (current) URL.revokeObjectURL(current); return URL.createObjectURL(blob); });
  }

  async function prepareShutdown() {
    if (!online || snapshot.status.kind !== "idle" || busy !== null) return;
    setShutdownError("");
    const prepared = await run("shutdown-prepare", () => api.prepareShutdown());
    if (prepared) setShutdown(prepared);
  }

  async function confirmShutdown() {
    if (!shutdown) return;
    if (!online || snapshot.status.kind !== "idle" || busy !== null) {
      setShutdownError("Arrêtez d’abord l’enregistrement ou la lecture avant de confirmer l’arrêt du PC.");
      return;
    }
    setShutdownError("");
    const result = await run("shutdown-confirm", () => api.confirmShutdown(shutdown.confirmationId));
    if (result) setShutdown(null);
    else setShutdownError("La confirmation n’a pas pu être vérifiée. Regardez l’état du PC avant de réessayer.");
  }

  return (
    <main className="settings-page">
      <header className="page-header">
        <div className="breadcrumbs"><span>Préférences</span><Icon name="chevron" size={13} /><span>Réglages</span></div>
        <h1 tabIndex={-1}>Réglages</h1>
        <p>Adaptez AUTO-COC à votre façon de jouer.</p>
      </header>
      <div className="settings-layout">
        <nav className="settings-nav" aria-label="Rubriques des réglages">
          {(["general", "shortcuts", "telegram", "system"] as SettingsTab[]).map((key) => {
            const labels: Record<SettingsTab, string> = { general: "Général", shortcuts: "Raccourcis", telegram: "Telegram", system: "Actions locales" };
            const icons: Record<SettingsTab, "sliders" | "keyboard" | "activity" | "shield"> = { general: "sliders", shortcuts: "keyboard", telegram: "activity", system: "shield" };
            return <button key={key} type="button" className={`settings-nav-item ${tab === key ? "active" : ""}`} aria-pressed={tab === key} onClick={() => setTab(key)}><Icon name={icons[key]} size={16} />{labels[key]}{key === "telegram" && telegram.status === "error" && <i aria-hidden="true" />}</button>;
          })}
        </nav>

        <section className="settings-content">
          {tab === "general" && (
            <>
              <div className="settings-section-head"><span className="section-kicker">Votre environnement</span><h2>Général</h2><p>Le chemin est utilisé uniquement pour lancer l’application que vous configurez ici.</p></div>
              <form className="setting-row path-setting" onSubmit={(event) => { event.preventDefault(); void run("path", () => api.updateSettings({ cocPath: path.trim() }), "Chemin enregistré."); }}>
                <div className="setting-row-copy"><strong>Chemin de Clash of Clans</strong><span>Chemin local vers l’application ou son raccourci.</span><label className="path-input"><Icon name="folder" size={16} /><input value={path} onChange={(event) => setPath(event.target.value)} placeholder="Ex. C:\\…\\Clash of Clans.lnk" aria-label="Chemin de Clash of Clans" /></label></div>
                <button className="button quiet" type="submit" disabled={!online || busy !== null || path.trim() === snapshot.settings.cocPath}>{busy === "path" ? "Enregistrement…" : "Enregistrer"}</button>
              </form>
              <div className="setting-row">
                <div className="setting-row-copy"><strong>Répéter les macros</strong><span>La boucle s’applique au prochain lancement.</span></div>
                <label className="switch-control"><span className="sr-only">Activer la boucle</span><input type="checkbox" checked={snapshot.settings.loop} disabled={!online || snapshot.status.kind !== "idle" || busy !== null} onChange={(event) => void run("loop-setting", () => api.updateSettings({ loop: event.target.checked }), event.target.checked ? "Boucle activée." : "Boucle désactivée.")} /><i aria-hidden="true" /></label>
              </div>
              <div className="setting-row coc-launch-row">
                <div className="setting-row-copy"><strong>Lancer Clash of Clans</strong><span>{snapshot.settings.cocPath ? "Utilise le chemin enregistré ci-dessus." : "Ajoutez d’abord le chemin de l’application."}</span></div>
                <button className="button quiet" type="button" disabled={!online || !snapshot.settings.cocPath || busy !== null || snapshot.status.kind !== "idle"} onClick={() => void run("launch-coc", () => api.launchCoc(), "Demande de lancement envoyée.")}><Icon name="arrow" size={15} />Lancer</button>
              </div>
            </>
          )}

          {tab === "shortcuts" && (
            <>
              <div className="settings-section-head"><span className="section-kicker">Commandes clavier</span><h2>Raccourcis</h2><p>Ils fonctionnent lorsque AUTO-COC n’est pas au premier plan. Choisissez une combinaison différente pour chaque action. Leur modification est possible lorsque l’atelier est au repos.</p></div>
              <div className="shortcut-list">
                {(Object.keys(shortcutLabels) as ShortcutName[]).map((key) => (
                  <div className={`shortcut-row ${key === "stop" ? "stop-shortcut" : ""}`} key={key}>
                    <div className="setting-row-copy"><strong>{shortcutLabels[key]}</strong><span>{key === "toggle" ? "Passe de la lecture à l’arrêt." : key === "play" ? "Lance la macro sélectionnée." : "Interrompt l’action en cours."}</span></div>
                    <div className="shortcut-edit">
                      <kbd className={capturing === key ? "capture-active" : ""}>{capturing === key ? "Appuyez sur les touches…" : shortcuts[key]}</kbd>
                      <button className="button quiet" type="button" onClick={() => { setShortcutError(""); setCapturing(key); }} disabled={!shortcutsEditable || capturing !== null}>{capturing === key ? "Écoute…" : "Modifier"}</button>
                    </div>
                  </div>
                ))}
              </div>
              {capturing && <p className="capture-hint"><Icon name="keyboard" size={15} />Appuyez sur la combinaison souhaitée. Échap annule l’écoute en cours; les combinaisons déjà modifiées restent en brouillon.</p>}
              {shortcutError && <p className="form-error" role="alert">{shortcutError}</p>}
              {duplicated && <p className="form-error" role="alert">Deux actions utilisent la même combinaison. Modifiez-les avant d’enregistrer.</p>}
              <div className="settings-footer"><span>Le raccourci Arrêter reste configuré en permanence.</span><button className="button primary" type="button" onClick={() => void saveShortcuts()} disabled={!shortcutsEditable || !changedShortcuts || duplicated || capturing !== null}>Enregistrer les raccourcis</button></div>
            </>
          )}

          {tab === "telegram" && (
            <>
              <div className="settings-section-head"><span className="section-kicker">Connexion facultative</span><h2>Telegram</h2><p>Contrôlez les macros à distance après avoir relié votre propre bot.</p></div>
              <div className="telegram-state-row">
                <div className="setting-row-copy"><strong>État du bot</strong><span>{telegram.paired ? "Le chat Telegram est associé à cette installation." : telegram.tokenConfigured ? "Le token est enregistré, mais aucun chat n’est associé." : "Ajoutez le token de votre bot pour commencer."}</span></div>
                <span className={`status-chip ${telegramTone(telegram.status)}`}><i aria-hidden="true" />{statusCopy[telegram.status] ?? "État inconnu"}</span>
              </div>
              <div className="telegram-token-block">
                <label className="field-label" htmlFor="telegram-token">Token du bot</label>
                <div className="token-entry"><input id="telegram-token" type="password" autoComplete="new-password" value={token} onChange={(event) => setToken(event.target.value)} placeholder={telegram.tokenConfigured ? "Token enregistré · saisir pour remplacer" : "Coller le token fourni par BotFather"} /><button className="button quiet" type="button" disabled={!online || !token.trim() || busy !== null} onClick={() => void saveToken()}>{busy === "telegram-token" ? "Enregistrement…" : "Enregistrer"}</button></div>
                <span className="field-note"><Icon name="shield" size={14} />Le token est stocké par le service Rust dans le coffre Windows. Il ne revient jamais dans cette page.</span>
                {telegram.tokenConfigured && <button className="text-button danger-text remove-token" type="button" disabled={!online || busy !== null} onClick={() => setShowTokenRemoval(true)}><Icon name="trash" size={14} />Retirer le token enregistré</button>}
              </div>
              <div className="telegram-pairing-block">
                <div className="setting-row-copy"><strong>Associer une conversation</strong><span>Le code ne fonctionne que pour une courte durée et dans une conversation privée avec votre bot.</span></div>
                {pairing ? (
                  <div className="pairing-code-block"><span className="section-kicker">Commande à envoyer au bot</span><div className="pairing-code-line"><code>/start {pairing.code}</code><button className="icon-button subtle" type="button" aria-label="Copier la commande d’appairage" onClick={() => void copyPairingCode()}><Icon name="keyboard" size={16} /></button></div><span className="field-note">Le code expire le {new Date(pairing.expiresAt).toLocaleString("fr-FR")}.</span><button className="text-button" type="button" disabled={!online || busy !== null} onClick={() => void run("telegram-cancel-pairing", () => api.cancelTelegramPairing(), "Code d’appairage annulé.").then((result) => { if (result) setPairing(null); })}>Annuler le code</button></div>
                ) : (
                  <button className="button quiet" type="button" disabled={!online || !telegram.tokenConfigured || telegram.paired || busy !== null} onClick={() => void startPairing()}>{busy === "telegram-pairing" ? "Création du code…" : "Créer un code d’appairage"}</button>
                )}
              </div>
              <div className="settings-footer telegram-test-footer"><span>{telegramMessage || "Le test met à jour l’état du bot affiché ci-dessus."}</span><button className="button primary" type="button" disabled={!online || !telegram.tokenConfigured || busy !== null} onClick={() => void run("telegram-test", () => api.testTelegram(), "Vérification Telegram terminée.")}>{busy === "telegram-test" ? "Vérification…" : "Tester la connexion"}</button></div>
            </>
          )}

          {tab === "system" && (
            <>
              <div className="settings-section-head"><span className="section-kicker">Outils de cet ordinateur</span><h2>Actions locales</h2><p>Ces actions sont exécutées par le service local sur ce PC.</p></div>
              <div className="setting-row system-action-row">
                <div className="setting-row-copy"><strong>Capture d’écran</strong><span>L’aperçu reste dans cette page. Aucune image n’est envoyée à Telegram.</span></div>
                <button className="button quiet" type="button" disabled={!online || busy !== null} onClick={() => void takeScreenshot()}><Icon name="activity" size={15} />{busy === "screenshot" ? "Capture…" : "Capturer l’écran"}</button>
              </div>
              {screenshotUrl && <figure className="screenshot-preview"><img src={screenshotUrl} alt="Capture d’écran réalisée sur cet ordinateur" /><figcaption><span>Aperçu local</span><a className="text-button" href={screenshotUrl} download="auto-coc-capture.png">Enregistrer l’image</a></figcaption></figure>}
              <div className="setting-row system-action-row">
                <div className="setting-row-copy"><strong>Dossier des macros</strong><span>Ouvre le dossier de données utilisé par AUTO-COC.</span></div>
                <button className="button quiet" type="button" disabled={!online || busy !== null} onClick={() => void run("macros-folder", () => api.openMacrosFolder(), "Dossier des macros ouvert.")}><Icon name="folder" size={15} />Ouvrir le dossier</button>
              </div>
              <div className="shutdown-warning">
                <div className="setting-row-copy"><strong>Arrêter l’ordinateur</strong><span>{snapshot.status.kind === "idle" ? "L’arrêt demande une préparation puis une confirmation séparée. Ne continuez que si vous êtes devant ce PC." : "Terminez d’abord l’enregistrement ou la lecture depuis l’atelier."}</span></div>
                <button className="button danger-outline" type="button" disabled={!online || snapshot.status.kind !== "idle" || busy !== null} onClick={() => void prepareShutdown()}>{busy === "shutdown-prepare" ? "Préparation…" : "Préparer l’arrêt"}</button>
              </div>
              {shutdownError && <p className="form-error" role="alert">{shutdownError}</p>}
              <div className="quit-application-row">
                <div className="setting-row-copy"><strong>Quitter AUTO-COC</strong><span>{snapshot.status.kind === "idle" ? "Ferme le service local et ses raccourcis globaux." : "Arrêtez d’abord l’enregistrement ou la lecture depuis l’atelier."}</span></div>
                <button className="button quiet" type="button" disabled={!online || snapshot.status.kind !== "idle" || busy !== null} onClick={() => void run("application-quit", () => api.quitApplication(), "Demande de fermeture envoyée au service local.")}>Quitter l’application</button>
              </div>
            </>
          )}
        </section>
      </div>
      {showTokenRemoval && <Dialog title="Retirer le token Telegram ?" description="Le bot sera déconnecté de cette installation. Vous pourrez ajouter un nouveau token plus tard." onClose={() => setShowTokenRemoval(false)}><div className="dialog-actions"><button className="button quiet" type="button" onClick={() => setShowTokenRemoval(false)}>Garder le token</button><button className="button danger" type="button" disabled={!online || busy !== null} onClick={() => { void run("telegram-remove", () => api.clearTelegramToken(), "Token Telegram retiré.").then((result) => { if (result) { setShowTokenRemoval(false); setTelegramMessage("Token retiré du stockage sécurisé."); } }); }}>Retirer le token</button></div></Dialog>}
      {shutdown && <Dialog title="Confirmer l’arrêt du PC" description={snapshot.status.kind === "idle" ? `L’ordinateur sera arrêté. La demande expire le ${new Date(shutdown.expiresAt).toLocaleString("fr-FR")}. Fermez d’abord les autres applications et vérifiez que vous êtes devant ce PC.` : "Une activité est en cours. Arrêtez-la depuis l’atelier avant de confirmer l’arrêt du PC."} onClose={() => setShutdown(null)}><div className="dialog-actions"><button className="button quiet" type="button" onClick={() => setShutdown(null)}>Annuler</button><button className="button danger" type="button" disabled={!online || snapshot.status.kind !== "idle" || busy !== null} onClick={() => void confirmShutdown()}>{busy === "shutdown-confirm" ? "Confirmation…" : "Arrêter l’ordinateur"}</button></div>{shutdownError && <p className="form-error" role="alert">{shutdownError}</p>}</Dialog>}
    </main>
  );
}
