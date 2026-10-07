# AUTO-COC

AUTO-COC enregistre et rejoue des macros clavier/souris sous Windows. L’interface React est hébergée dans une fenêtre desktop Tauri/WebView2; son backend Rust est intégré à l’application et appelé par commandes Tauri. Aucun service séparé, navigateur externe ou terminal n’est à lancer en production.

## Prérequis Windows

- Windows 10 ou 11 x64.
- Rust installé avec [rustup](https://rustup.rs/) et le toolchain MSVC. Tauri utilise les [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) : installez la charge **Desktop development with C++** et le Windows SDK. Si nécessaire, sélectionnez le toolchain avec `rustup default stable-msvc`.
- Node.js `22.12+` et npm pour le frontend.
- Le runtime [Microsoft Edge WebView2 Evergreen](https://developer.microsoft.com/microsoft-edge/webview2/). Il est fourni sur la plupart des versions récentes de Windows 10/11. Si le runtime manque, l’installateur actuel télécharge son bootstrapper, ce qui nécessite une connexion Internet.
- Le CLI Tauri 2, installé une fois :

```powershell
cargo install tauri-cli --version "^2.0.0" --locked
```

Les prérequis Windows détaillés sont décrits dans la [documentation Tauri](https://v2.tauri.app/start/prerequisites/) et la [documentation Microsoft WebView2](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution).

## Développement

Depuis la racine du dépôt, installez les dépendances puis démarrez la fenêtre Tauri :

```powershell
cd frontend
npm ci
cargo tauri dev
```

La configuration Tauri lance Vite sur `127.0.0.1:5173` et ouvre l’interface dans la fenêtre de développement. Pour quitter, fermez la fenêtre ou utilisez `Ctrl+C` dans le terminal de développement.

Les vérifications du frontend sont disponibles depuis `frontend` :

```powershell
npm run lint
npm run typecheck
npm test
npm run build
```

Vérifications Rust du backend, depuis la racine :

```powershell
cargo fmt --manifest-path backend/Cargo.toml --check
cargo clippy --manifest-path backend/Cargo.toml --all-targets
cargo test --manifest-path backend/Cargo.toml
```

## Construire l’installateur Windows

Depuis `frontend`, exécutez les vérifications frontend, puis construisez l’installateur NSIS x64 :

```powershell
npm ci
npm run lint
npm run typecheck
npm test
cargo tauri build --bundles nsis --target x86_64-pc-windows-msvc
```

`cargo tauri build` exécute aussi `npm run build` selon `frontend/src-tauri/tauri.conf.json`. L’installateur est produit sous `frontend/src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/` avec un nom se terminant par `-setup.exe`. L’option `--bundles nsis` limite cette commande à NSIS, même si la configuration permet aussi MSI.

L’installateur NSIS utilise le mode d’installation par utilisateur par défaut. Les données AUTO-COC sont conservées dans `%LOCALAPPDATA%\AUTO-COC`; le token Telegram est protégé pour le compte Windows courant. Le runtime WebView2 peut être installé pendant l’installation si nécessaire.

## Utiliser AUTO-COC

Créez une macro dans l’atelier, sélectionnez-la, puis démarrez l’enregistrement. La capture commence après trois secondes de préparation. À l’arrêt, les trois dernières secondes sont retirées; un enregistrement de trois secondes ou moins produit une macro vide. La lecture peut être interrompue depuis l’atelier ou avec un raccourci.

Les raccourcis initiaux sont `F1` pour démarrer la macro sélectionnée ou arrêter une lecture, `Ctrl+Shift+1` pour démarrer la lecture et `Ctrl+Shift+0` pour arrêter la lecture ou finaliser l’enregistrement. Ils peuvent être personnalisés dans Réglages. Windows doit accepter chaque combinaison; si une combinaison est déjà utilisée, l’application reste accessible et permet d’en choisir une autre.

Telegram est facultatif. Configurez votre bot dans Réglages, créez un code d’appairage, puis envoyez `/start CODE` au bot depuis une conversation privée. Les commandes texte incluent `stop`, `go`, `menu`, `capture`, `shutdown`, `relancer` et `launch`. La commande `gif` indique que la capture GIF n’est pas prise en charge.

La page Aide décrit les commandes, les raccourcis et la migration. Pour migrer, choisissez le dossier de l’ancienne installation ou son sous-dossier `config`. Une sauvegarde CSV expurgée du token est créée avant l’import; les fichiers source restent à leur emplacement. Les macros illisibles ou incompatibles sont signalées et ne sont pas rejouées. L’identifiant de conversation Telegram historique n’est pas importé : après la migration, appairez à nouveau votre conversation depuis Réglages > Telegram.

La [matrice de régression desktop](docs/desktop-regression-matrix.md) distingue les vérifications automatisées des parcours Windows qui restent à valider manuellement.

## CI et publication

Le workflow `.github/workflows/desktop.yml` vérifie le frontend et le backend sur Windows x64, construit l’installateur NSIS et exécute un smoke test d’installation, de lancement, d’actions accessibles dans l’interface et de fermeture. Après réussite des vérifications et du smoke test, l’installateur est joint à l’exécution de pull request sous `auto-coc-windows-x64-nsis`; la rétention configurée est de 14 jours, sous réserve des politiques du dépôt ou de l’organisation. Un push sur `main` construit et vérifie le paquet, mais ne conserve pas d’artefact. Aucun GitHub Release n’est créé et l’application n’est pas publiée.

### Générer et valider un installateur candidat

La version applicative est actuellement `4.0.0` dans `frontend/src-tauri/tauri.conf.json`, `frontend/src-tauri/Cargo.toml` et `backend/Cargo.toml`. Depuis `frontend`, lancez la commande de build de la section précédente. Le fichier `*-setup.exe` se trouve dans `frontend/src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/`.

Pour valider ce paquet localement, lancez le smoke test depuis une session Windows interactive, avec un compte de test où `%LOCALAPPDATA%\com.autococ.desktop` n’existe pas. Depuis la racine du dépôt, lancez le test sur un dossier temporaire existant :

```powershell
$env:RUNNER_TEMP = $env:TEMP
.\.github\scripts\windows-runtime-smoke.ps1 -BundleDirectory .\frontend\src-tauri\target\x86_64-pc-windows-msvc\release\bundle\nsis
```

Le script installe le paquet dans un espace temporaire et tente de le désinstaller à la fin. Il vérifie le démarrage de la fenêtre, la création et le renommage d’une macro via l’interface accessible, la persistance du fichier, le lancement d’une seconde instance et la fermeture propre. Il prend un instantané des processus Chrome/Edge avant le démarrage, puis vérifie après les actions qu’aucun nouveau processus n’est encore actif; un navigateur lancé puis fermé avant ce contrôle ne serait pas détecté. Il refuse d’exécuter le test si le profil Tauri existe déjà. Ce smoke test ne valide pas le rendu visuel, les raccourcis globaux, ni l’enregistrement et la lecture réels; effectuez aussi les contrôles manuels ci-dessous avant toute distribution.

Avant de distribuer une version, installez l’artefact sur une machine Windows propre, avec un compte utilisateur standard, et vérifiez le démarrage, l’absence de terminal et de navigateur externe, les raccourcis globaux, l’enregistrement/la lecture, la fermeture, ainsi que la conservation des macros et réglages après mise à niveau. Testez aussi l’installation lorsque WebView2 est absent, avec et sans accès réseau selon le mode de distribution choisi.

Avant une release publique, mettez à jour de façon cohérente les versions ci-dessus, configurez la signature Authenticode en gardant certificat et clé privée hors du dépôt, puis ajoutez et validez une procédure de publication qui crée une GitHub Release et y attache l’installateur. Le workflow actuel n’a que la permission `contents: read` et ne réalise aucune de ces étapes. Sans signature, Windows SmartScreen peut avertir l’utilisateur; une signature ne garantit pas qu’un nouveau certificat aura immédiatement une réputation SmartScreen. L’[updater Tauri](https://v2.tauri.app/plugin/updater/) n’est pas configuré et n’est pas fourni par le workflow actuel.

## Structure

```text
backend/            logique métier Rust, stockage, accès Windows et tests
frontend/           interface React/TypeScript
frontend/src-tauri/ fenêtre desktop, commandes Tauri et packaging Windows
config/             icône et ressources suivies
```
