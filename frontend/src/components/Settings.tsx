import { useEffect, useState } from "react";
import type { ActionRunner, BackendClient, PairingPreparation, ShutdownPreparation, Snapshot } from "../api";
import { Dialog } from "./Dialog";
import { Icon } from "./Icon";
import { normalizeShortcut } from "./normalizeShortcut";

type Props = { api: BackendClient; snapshot: Snapshot; busy: string | null; run: ActionRunner; online: boolean; initialTab?: SettingsTab };
export type SettingsTab = "general" | "shortcuts" | "telegram" | "system";
type ShortcutName = keyof Snapshot["settings"]["shortcuts"];
type PairingCode = Pick<PairingPreparation, "code" | "expiresAt">;

const shortcutLabels: Record<ShortcutName, string> = { toggle: "Basculer lecture / arrêt", play: "Lire la macro", stop: "Arrêter" };
const statusCopy: Record<Snapshot["settings"]["telegram"]["status"], string> = {
  not_configured: "Non configuré",
  disconnected: "Déconnecté",
  waiting_pairing: "En attente de l’appairage",
  connected: "Connecté",
  token_rejected: "Token refusé",
  network_error: "Connexion impossible",
  error: "Erreur Telegram",
};

function telegramTone(status: Snapshot["settings"]["telegram"]["status"]): string {
  if (status === "connected") return "good";
  if (status === "error" || status === "token_rejected" || status === "network_error") return "bad";
  return status === "waiting_pairing" ? "copper" : "neutral";
}

function formatTelegramHandle(username: string): string | null {
  const trimmed = username.trim();
  if (!trimmed) return null;
  return trimmed.startsWith("@") ? trimmed : `@${trimmed}`;
}

export function Settings({ api, snapshot, busy, run, online, initialTab = "general" }: Props) {
  const [tab, setTab] = useState<SettingsTab>(initialTab);
  const [path, setPath] = useState(snapshot.settings.cocPath);
  const [shortcuts, setShortcuts] = useState(snapshot.settings.shortcuts);
  const [capturing, setCapturing] = useState<ShortcutName | null>(null);
  const [shortcutError, setShortcutError] = useState("");
  const [telegramBotUsername, setTelegramBotUsername] = useState("");
  const [botFatherCopyMessage, setBotFatherCopyMessage] = useState("");
  const [token, setToken] = useState("");
  const [editingTelegramToken, setEditingTelegramToken] = useState(!snapshot.settings.telegram.tokenConfigured);
  const [tokenMessage, setTokenMessage] = useState("");
  const [verificationMessage, setVerificationMessage] = useState("");
  const [copyMessage, setCopyMessage] = useState("");
  const [pairing, setPairing] = useState<PairingCode | null>(snapshot.settings.telegram.pairingCode && snapshot.settings.telegram.pairingExpiresAt ? { code: snapshot.settings.telegram.pairingCode, expiresAt: snapshot.settings.telegram.pairingExpiresAt } : null);
  const [now, setNow] = useState(Date.now());
  const [screenshotUrl, setScreenshotUrl] = useState("");
  const [shutdown, setShutdown] = useState<ShutdownPreparation | null>(null);
  const [showTokenRemoval, setShowTokenRemoval] = useState(false);
  const [shutdownError, setShutdownError] = useState("");
  const pairingExpiresAt = pairing?.expiresAt;

  useEffect(() => setPath(snapshot.settings.cocPath), [snapshot.settings.cocPath]);
  const { toggle, play, stop } = snapshot.settings.shortcuts;
  useEffect(() => setShortcuts({ toggle, play, stop }), [toggle, play, stop]);
  useEffect(() => {
    if (snapshot.settings.telegram.paired) {
      setPairing(null);
      return;
    }
    if (snapshot.settings.telegram.pairingCode && snapshot.settings.telegram.pairingExpiresAt) {
      setPairing({ code: snapshot.settings.telegram.pairingCode, expiresAt: snapshot.settings.telegram.pairingExpiresAt });
    } else setPairing(null);
  }, [snapshot.settings.telegram.pairingCode, snapshot.settings.telegram.pairingExpiresAt, snapshot.settings.telegram.paired]);
  useEffect(() => {
    if (!pairingExpiresAt) return;
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [pairingExpiresAt]);
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
  const telegramVerificationFailed = telegram.status === "error" || telegram.status === "token_rejected" || telegram.status === "network_error";
  const telegramTokenVerified = !telegramVerificationFailed && (telegram.status === "connected" || telegram.status === "waiting_pairing" || telegram.paired);
  const pairingExpired = pairing !== null && Date.parse(pairing.expiresAt) <= now;
  const botHandle = formatTelegramHandle(telegramBotUsername);
  const shouldEditTelegramToken = !telegram.tokenConfigured || editingTelegramToken;
  let pairingStepStatus = "Après l’étape 2";
  let pairingStepDescription = "Cette étape s’ouvrira après la vérification du token.";
  if (telegram.paired && telegramTokenVerified) {
    pairingStepStatus = "Conversation associée";
    pairingStepDescription = "Votre conversation privée est reliée.";
  } else if (telegramVerificationFailed) {
    pairingStepStatus = "Vérification requise";
    pairingStepDescription = "Vérifiez la connexion Telegram avant de reprendre le contrôle à distance.";
  } else if (pairing && !pairingExpired) {
    pairingStepStatus = "Code en attente";
    pairingStepDescription = "Envoyez la commande au bot, puis revenez dans AUTO-COC.";
  } else if (telegramTokenVerified) {
    pairingStepStatus = "À faire";
    pairingStepDescription = "Générez une commande, puis envoyez-la au bot dans Telegram.";
  }
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
    setTokenMessage("");
    setVerificationMessage("");
    const saved = await run("telegram-token", () => api.saveTelegramToken(token.trim()));
    if (saved) {
      setToken("");
      setEditingTelegramToken(false);
    }
    setTokenMessage(saved ? "Token enregistré. Vérifiez-le à l’étape suivante; le champ a été effacé." : "Le token n’a pas été enregistré.");
  }

  async function verifyTelegramToken() {
    setVerificationMessage("");
    const result = await run("telegram-test", () => api.testTelegram());
    if (!result) {
      setVerificationMessage("La vérification n’a pas abouti dans AUTO-COC. Réessayez.");
      return;
    }
    const { status } = result.settings.telegram;
    if (status === "token_rejected") setVerificationMessage("Telegram a refusé le token. Comparez-le avec celui fourni par BotFather.");
    else if (status === "network_error") setVerificationMessage("La connexion à Telegram n’a pas abouti. Vérifiez l’accès à Internet puis réessayez.");
    else if (status === "error") setVerificationMessage("Telegram a renvoyé une erreur lors de la vérification. Réessayez plus tard.");
    else setVerificationMessage("Token vérifié. Aucun message Telegram n’a été envoyé.");
  }

  async function copyBotFatherCommand() {
    setBotFatherCopyMessage("");
    try {
      await navigator.clipboard.writeText("/newbot");
      setBotFatherCopyMessage("/newbot copié. Collez-le dans votre conversation avec @BotFather.");
    } catch {
      setBotFatherCopyMessage("Copie impossible. Envoyez /newbot à @BotFather manuellement.");
    }
  }

  async function startPairing() {
    setCopyMessage("");
    const result = await run("telegram-pairing", () => api.startTelegramPairing(), "Code d’appairage créé.");
    if (result) {
      setPairing(result);
    }
  }

  async function copyPairingCode() {
    if (!pairing) return;
    setCopyMessage("");
    try {
      await navigator.clipboard.writeText(`/start ${pairing.code}`);
      setCopyMessage("Commande copiée. Envoyez-la dans la conversation privée avec votre bot.");
    } catch {
      setCopyMessage("Copie impossible. Sélectionnez la commande affichée et copiez-la manuellement.");
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
            const labels: Record<SettingsTab, string> = { general: "Général", shortcuts: "Raccourcis", telegram: "Telegram", system: "Système" };
            const icons: Record<SettingsTab, "sliders" | "keyboard" | "activity" | "shield"> = { general: "sliders", shortcuts: "keyboard", telegram: "activity", system: "shield" };
            return <button key={key} type="button" className={`settings-nav-item ${tab === key ? "active" : ""}`} aria-pressed={tab === key} onClick={() => setTab(key)}><Icon name={icons[key]} size={16} />{labels[key]}{key === "telegram" && telegramVerificationFailed && <i aria-hidden="true" />}</button>;
          })}
        </nav>

        <section className="settings-content">
          {tab === "general" && (
            <>
              <div className="settings-section-head"><span className="section-kicker">Votre environnement</span><h2>Général</h2><p>Configurez le chemin de lancement ou copiez l’URI Google Play Games affichée sous le champ.</p></div>
              <form className="setting-row path-setting" onSubmit={(event) => { event.preventDefault(); void run("path", () => api.updateSettings({ cocPath: path.trim() }), "Chemin enregistré."); }}>
                <div className="setting-row-copy"><strong>Cible de lancement de Clash of Clans</strong><span>Chemin absolu vers l’application ou son raccourci.</span><label className="path-input"><Icon name="folder" size={16} /><input value={path} onChange={(event) => setPath(event.target.value)} placeholder={"Ex. C:\\…\\Clash of Clans.lnk"} aria-label="Cible de lancement de Clash of Clans" /></label><div className="path-uri-note"><span>URI Google Play Games. Faites Tab puis Ctrl+C, ou sélectionnez le texte à la souris.</span><textarea className="path-uri-copy" aria-label="URI Google Play Games à copier" readOnly rows={3} value="googleplaygames://launch/?id=com.supercell.clashofclans&lid=1&pid=1" onFocus={(event) => event.currentTarget.select()} /></div></div>
                <button className="button quiet" type="submit" disabled={!online || busy !== null || path.trim() === snapshot.settings.cocPath}>{busy === "path" ? "Enregistrement…" : "Enregistrer"}</button>
              </form>
              <div className="setting-row">
                <div className="setting-row-copy"><strong>Répéter les macros</strong><span>La boucle s’applique au prochain lancement.</span></div>
                <label className="switch-control"><span className="sr-only">Activer la boucle</span><input type="checkbox" checked={snapshot.settings.loop} disabled={!online || snapshot.status.kind !== "idle" || busy !== null} onChange={(event) => void run("loop-setting", () => api.updateSettings({ loop: event.target.checked }), event.target.checked ? "Boucle activée." : "Boucle désactivée.")} /><i aria-hidden="true" /></label>
              </div>
              <div className="setting-row">
                <div className="setting-row-copy"><strong>Pause automatique si Clash of Clans perd le premier plan</strong><span>Le replay et l’enregistrement démarrent seulement si CoC est au premier plan. Activée, l’option attend jusqu’à 30 s que le jeu soit actif, puis met le replay en pause si le focus est perdu. Désactivée, CoC doit déjà être actif pour démarrer; une perte de focus arrête le replay. Les entrées restent ciblées sur la fenêtre du jeu. La capture est limitée à CoC; les actions souris sont gardées uniquement dans sa zone de jeu.</span></div>
                <label className="switch-control"><span className="sr-only">Activer la pause automatique Clash of Clans</span><input type="checkbox" checked={snapshot.settings.requireCocForeground} disabled={!online || snapshot.status.kind !== "idle" || busy !== null} onChange={(event) => void run("coc-foreground-setting", () => api.updateSettings({ requireCocForeground: event.target.checked }), event.target.checked ? "Pause automatique activée." : "Pause automatique désactivée.")} /><i aria-hidden="true" /></label>
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
                      <button className="button quiet" type="button" aria-label={`Modifier le raccourci ${shortcutLabels[key]}`} onClick={() => { setShortcutError(""); setCapturing(key); }} disabled={!shortcutsEditable || capturing !== null}>{capturing === key ? "Écoute…" : "Modifier"}</button>
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
              <div className="settings-section-head"><span className="section-kicker">Connexion facultative</span><h2>Telegram</h2><p>Reliez votre bot pour piloter les macros à distance. L’atelier sur ce PC fonctionne sans Telegram.</p></div>
              <ol className="telegram-steps" aria-label="Étapes pour connecter Telegram">
                <li className="telegram-step">
                  <div className="telegram-step-heading"><span className="telegram-step-number">01</span><div><h3>Créer votre bot</h3><p>Dans Telegram, ouvrez @BotFather et envoyez /newbot. Choisissez un nom visible, puis un nom d’utilisateur se terminant par bot. Copiez le token qu’il vous donne.</p></div><span className={`telegram-step-state ${telegram.tokenConfigured ? "done" : ""}`}>{telegram.tokenConfigured ? "Token enregistré" : "À faire"}</span></div>
                  <div className="telegram-step-content">
                    {shouldEditTelegramToken && <>
                      <button className="button quiet" type="button" onClick={() => void copyBotFatherCommand()}><Icon name="copy" size={15} />Copier /newbot</button>
                      <p className="telegram-feedback" role="status">{botFatherCopyMessage}</p>
                    </>}
                    <label className="field-label" htmlFor="telegram-bot-username">Nom d’utilisateur du bot (facultatif)</label>
                    <input id="telegram-bot-username" autoComplete="off" spellCheck={false} value={telegramBotUsername} onChange={(event) => setTelegramBotUsername(event.target.value)} placeholder="Ex. @mon_bot" />
                    <span className="field-note">BotFather vous le donne après la création. Vous n’avez pas besoin de le saisir ici; il sert seulement à retrouver le bot plus facilement à l’étape 3.</span>
                    {shouldEditTelegramToken ? <>
                      <label className="field-label" htmlFor="telegram-token">Token du bot</label>
                      <div className="token-entry"><input id="telegram-token" type="password" autoComplete="new-password" value={token} onChange={(event) => { setToken(event.target.value); setTokenMessage(""); }} placeholder={telegram.tokenConfigured ? "Saisir le nouveau token" : "Coller le token de BotFather"} /><button className="button quiet" type="button" disabled={!online || !token.trim() || busy !== null} onClick={() => void saveToken()}>{busy === "telegram-token" ? "Enregistrement…" : "Enregistrer le token"}</button></div>
                      <p className="telegram-feedback" role="status">{tokenMessage}</p>
                      <span className="field-note"><Icon name="shield" size={14} />Stocké dans le coffre Windows. Le token ne sera pas réaffiché.</span>
                      {telegram.tokenConfigured && <button className="text-button" type="button" onClick={() => { setToken(""); setEditingTelegramToken(false); }}>Annuler le remplacement</button>}
                    </> : <>
                      <span className="muted-copy">Token enregistré et protégé. Il reste masqué.</span>
                      <p className="telegram-feedback" role="status">{tokenMessage}</p>
                      <button className="text-button" type="button" onClick={() => setEditingTelegramToken(true)}>Remplacer le token</button>
                      <button className="text-button danger-text remove-token" type="button" disabled={!online || busy !== null} onClick={() => setShowTokenRemoval(true)}><Icon name="trash" size={14} />Retirer le token enregistré</button>
                    </>}
                  </div>
                </li>
                <li className="telegram-step">
                  <div className="telegram-step-heading"><span className="telegram-step-number">02</span><div><h3>Vérifier le token</h3><p>Vérifiez que le bot répond. Cette vérification n’envoie aucun message.</p></div><span className={`telegram-step-state ${telegramTokenVerified ? "done" : ""}`}>{!telegram.tokenConfigured ? "Après l’étape 1" : telegramTokenVerified ? "Vérifié" : telegram.status === "token_rejected" ? "Token refusé" : telegram.status === "network_error" ? "Connexion impossible" : "À vérifier"}</span></div>
                  {telegram.tokenConfigured && <div className="telegram-step-content telegram-verify-row">
                      <span className="muted-copy">{telegramTokenVerified ? "Le bot répond à Telegram." : "Cette étape confirme que le token est valide."}</span>
                      <button className="button quiet" type="button" disabled={!online || busy !== null} onClick={() => void verifyTelegramToken()}>{busy === "telegram-test" ? "Vérification…" : telegramTokenVerified ? "Revérifier le token" : "Vérifier le token"}</button>
                      <p className="telegram-feedback verification-feedback" role="status">{verificationMessage}</p>
                  </div>}
                </li>
                <li className="telegram-step">
                  <div className="telegram-step-heading"><span className="telegram-step-number">03</span><div><h3>Associer votre conversation</h3><p>{pairingStepDescription}</p></div><span className={`telegram-step-state ${telegramVerificationFailed ? "" : telegram.paired ? "done" : pairing && !pairingExpired ? "pending" : ""}`}>{pairingStepStatus}</span></div>
                  {telegramTokenVerified && <div className="telegram-step-content">
                    {telegram.paired ? <p className="telegram-paired-copy">Cette conversation peut maintenant contrôler AUTO-COC.</p> : pairing && !pairingExpired ? (
                      <div className="pairing-code-block"><span className="section-kicker">Dans Telegram, ouvrez une conversation privée avec {botHandle ?? "le nom du bot donné par BotFather"}, puis envoyez</span><div className="pairing-code-line"><code>/start {pairing.code}</code><button className="text-button copy-pairing-button" type="button" onClick={() => void copyPairingCode()}><Icon name="copy" size={15} />Copier la commande</button></div><p className="telegram-feedback copy-feedback" role="status">{copyMessage}</p><span className="field-note">Revenez dans AUTO-COC : la confirmation apparaîtra ici automatiquement. Le code expire le {new Date(pairing.expiresAt).toLocaleString("fr-FR")}.</span><button className="text-button" type="button" disabled={!online || busy !== null} onClick={() => void run("telegram-cancel-pairing", () => api.cancelTelegramPairing(), "Code d’appairage annulé.").then((result) => { if (result) setPairing(null); })}>Annuler le code</button></div>
                    ) : (
                      <div className="telegram-pairing-action">
                        {pairingExpired && <p className="form-hint" role="status">Le code a expiré. Générez-en un nouveau pour reprendre l’association.</p>}
                        <button className="button quiet" type="button" disabled={!online || telegram.paired || busy !== null} onClick={() => void startPairing()}>{busy === "telegram-pairing" ? "Création du code…" : pairingExpired ? "Générer un nouveau code" : "Générer une commande d’appairage"}</button>
                      </div>
                    )}
                  </div>}
                </li>
              </ol>
              <div className="telegram-status-summary"><span className={`status-chip ${telegramTone(telegram.status)}`}><i aria-hidden="true" />{telegram.paired && telegramTokenVerified ? "Conversation associée" : telegramTokenVerified ? "Bot vérifié" : telegram.tokenConfigured ? statusCopy[telegram.status] : "Non configuré"}</span><span>Les macros sur ce PC fonctionnent sans Telegram.</span></div>
            </>
          )}

          {tab === "system" && (
            <>
              <div className="settings-section-head"><span className="section-kicker">Outils de cet ordinateur</span><h2>Système</h2><p>Ces actions sont exécutées par AUTO-COC sur ce PC.</p></div>
              <div className="setting-row system-action-row">
                <div className="setting-row-copy"><strong>Capture d’écran</strong><span>L’aperçu reste dans cette page. Aucune image n’est envoyée à Telegram.</span></div>
                <button className="button quiet" type="button" disabled={!online || busy !== null} onClick={() => void takeScreenshot()}><Icon name="activity" size={15} />{busy === "screenshot" ? "Capture…" : "Capturer l’écran"}</button>
              </div>
              {screenshotUrl && <figure className="screenshot-preview"><img src={screenshotUrl} alt="Capture d’écran réalisée sur cet ordinateur" /><figcaption><span>Aperçu</span><a className="text-button" href={screenshotUrl} download="auto-coc-capture.png">Enregistrer l’image</a></figcaption></figure>}
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
                <div className="setting-row-copy"><strong>Quitter AUTO-COC</strong><span>{snapshot.status.kind === "idle" ? "Ferme l’application et désactive ses raccourcis globaux." : "Arrêtez d’abord l’enregistrement ou la lecture depuis l’atelier."}</span></div>
                <button className="button quiet" type="button" disabled={!online || snapshot.status.kind !== "idle" || busy !== null} onClick={() => void run("application-quit", () => api.quitApplication(), "Demande de fermeture envoyée à AUTO-COC.")}>Quitter l’application</button>
              </div>
            </>
          )}
        </section>
      </div>
      {showTokenRemoval && <Dialog title="Retirer le token Telegram ?" description="Le bot sera déconnecté de cette installation. Vous pourrez ajouter un nouveau token plus tard." onClose={() => setShowTokenRemoval(false)}><div className="dialog-actions"><button className="button quiet" type="button" onClick={() => setShowTokenRemoval(false)}>Garder le token</button><button className="button danger" type="button" disabled={!online || busy !== null} onClick={() => { void run("telegram-remove", () => api.clearTelegramToken()).then((result) => { if (result) { setShowTokenRemoval(false); setTokenMessage("Token retiré du stockage sécurisé."); setVerificationMessage(""); } }); }}>Retirer le token</button></div></Dialog>}
      {shutdown && <Dialog title="Confirmer l’arrêt du PC" description={snapshot.status.kind === "idle" ? `L’ordinateur sera arrêté. La demande expire le ${new Date(shutdown.expiresAt).toLocaleString("fr-FR")}. Fermez d’abord les autres applications et vérifiez que vous êtes devant ce PC.` : "Une activité est en cours. Arrêtez-la depuis l’atelier avant de confirmer l’arrêt du PC."} onClose={() => setShutdown(null)}><div className="dialog-actions"><button className="button quiet" type="button" onClick={() => setShutdown(null)}>Annuler</button><button className="button danger" type="button" disabled={!online || snapshot.status.kind !== "idle" || busy !== null} onClick={() => void confirmShutdown()}>{busy === "shutdown-confirm" ? "Confirmation…" : "Arrêter l’ordinateur"}</button></div>{shutdownError && <p className="form-error" role="alert">{shutdownError}</p>}</Dialog>}
    </main>
  );
}
