# AUTO-COC

AUTO-COC aide à rejouer des gestes clavier et souris répétitifs dans Clash of Clans lancé avec Google Play Games sur Windows. L’application s’adresse aux joueurs qui veulent enregistrer une séquence une fois, la retrouver dans leur bibliothèque et la relancer sans refaire chaque action manuellement.

Les macros et réglages sont conservés sur l’ordinateur. L’application réunit l’enregistrement, le replay, les raccourcis personnalisables et des commandes d’arrêt dans une fenêtre desktop. Telegram est une option séparée pour piloter certaines commandes à distance.

## Utiliser l’application

1. Dans l’Atelier, créez ou sélectionnez une macro.
2. Ouvrez Clash of Clans au premier plan et démarrez l’enregistrement. La capture commence après trois secondes de préparation. À l’arrêt, les trois dernières secondes sont retirées; il faut donc enregistrer plus de trois secondes pour conserver des actions.
3. Examinez le résumé et le déroulé visuel de la séquence, puis lancez ou arrêtez le replay depuis l’Atelier ou avec vos raccourcis.
4. Dans Réglages, personnalisez les raccourcis, activez la répétition et choisissez si le replay peut continuer en arrière-plan.

Par défaut, `F1` bascule entre lecture et arrêt, `Ctrl+Shift+1` démarre le replay et `Ctrl+Shift+0` arrête l’opération en cours. Modifiez-les dans Réglages > Raccourcis. Si Windows refuse une combinaison déjà utilisée, choisissez-en une autre.

## Contrôle du replay et limites

**Autoriser le replay en arrière-plan** est activé par défaut lorsqu’aucune valeur n’est enregistrée. Quand ce réglage est activé, AUTO-COC cherche une fenêtre visible reconnue de Clash of Clans et adresse les actions à cette fenêtre, même si une autre application est au premier plan. Si la fenêtre ciblée ou son processus disparaît, le replay s’arrête. En cas de cible absente ou ambiguë, le démarrage échoue plutôt que d’envoyer les actions ailleurs.

Si le réglage est désactivé, un lancement depuis l’Atelier peut attendre jusqu’à 30 secondes que CoC passe au premier plan; les raccourcis et Telegram ne patientent pas. Une perte de focus pendant la lecture met ensuite le replay en pause; il faut reprendre depuis l’Atelier et revenir dans le jeu. L’enregistrement reste toujours limité au premier plan et s’arrête si CoC perd le focus.

Le ciblage d’une fenêtre ne garantit pas que Google Play Games transmette ou que le jeu traite chaque action. AUTO-COC ne peut pas toujours détecter un crash interne de Clash of Clans si la fenêtre hôte de Google Play Games reste ouverte avec le même titre. Pour les nouvelles macros, les coordonnées suivent CoC quand sa fenêtre bouge; les anciennes macros conservent leurs coordonnées écran. Les déplacements hors de la zone du jeu sont ignorés, tandis qu’un clic ou un défilement hors zone arrête le replay.

L’aide intégrée présente les parcours de base, les raccourcis configurés, les précautions de lecture et les diagnostics de l’application. Le token Telegram n’est pas affiché dans l’aide ni dans les diagnostics. Telegram est facultatif : il faut configurer un bot et appairer une conversation privée avant de l’utiliser. Il prend notamment en charge l’arrêt, le lancement et la sélection de macros; `gif` indique que la capture GIF n’est pas prise en charge.

La [matrice de régression desktop](docs/desktop-regression-matrix.md) distingue la couverture automatisée des vérifications qui nécessitent une session Windows et le jeu réel.

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

## CI et publication

Le workflow [Windows desktop CI](.github/workflows/desktop.yml) vérifie le frontend et le backend sur Windows x64, construit l’installateur NSIS et exécute un smoke test d’installation, de lancement, d’actions accessibles dans l’interface et de fermeture. Après réussite des vérifications et du smoke test, l’installateur est joint à l’exécution de pull request sous `auto-coc-windows-x64-nsis`; la rétention configurée est de 14 jours, sous réserve des politiques du dépôt ou de l’organisation. Un push sur `main` construit et vérifie le paquet, mais ne conserve pas d’artefact. Aucun GitHub Release n’est créé et l’application n’est pas publiée.

### Générer et valider un installateur candidat

La version applicative est actuellement `4.0.0` dans `frontend/src-tauri/tauri.conf.json`, `frontend/src-tauri/Cargo.toml` et `backend/Cargo.toml`. Depuis `frontend`, lancez la commande de build de la section précédente. Le fichier `*-setup.exe` se trouve dans `frontend/src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/`.

Pour valider ce paquet localement, lancez le smoke test depuis une session Windows interactive, avec un compte de test où `%LOCALAPPDATA%\com.autococ.desktop` n’existe pas. Depuis la racine du dépôt, lancez le test sur un dossier temporaire existant :

```powershell
$env:RUNNER_TEMP = $env:TEMP
.\.github\scripts\windows-runtime-smoke.ps1 -BundleDirectory .\frontend\src-tauri\target\x86_64-pc-windows-msvc\release\bundle\nsis
```

Le script installe le paquet dans un espace temporaire et tente de le désinstaller à la fin. Il vérifie le démarrage de la fenêtre, la création et le renommage d’une macro via l’interface accessible, la persistance du fichier, le lancement d’une seconde instance et la fermeture propre. Il prend un instantané des processus Chrome/Edge avant le démarrage, puis vérifie après les actions qu’aucun nouveau processus n’est encore actif; un navigateur lancé puis fermé avant ce contrôle ne serait pas détecté. Il refuse d’exécuter le test si le profil Tauri existe déjà. Ce smoke test ne valide pas le rendu visuel, les raccourcis globaux, ni l’enregistrement et la lecture réels; effectuez aussi les contrôles manuels de la matrice avant toute distribution.

Avant de distribuer une version, installez l’artefact sur une machine Windows propre, avec un compte utilisateur standard, et vérifiez le démarrage, l’absence de terminal et de navigateur externe, les raccourcis globaux, l’enregistrement/la lecture, la fermeture, ainsi que la conservation des macros et réglages après mise à niveau. Testez aussi l’installation lorsque WebView2 est absent, avec et sans accès réseau selon le mode de distribution choisi.

Avant une release publique, mettez à jour de façon cohérente les versions ci-dessus, configurez la signature Authenticode en gardant certificat et clé privée hors du dépôt, puis ajoutez et validez une procédure de publication qui crée une GitHub Release et y attache l’installateur. Le workflow actuel n’a que la permission `contents: read` et ne réalise aucune de ces étapes. Sans signature, Windows SmartScreen peut avertir l’utilisateur; une signature ne garantit pas qu’un nouveau certificat aura immédiatement une réputation SmartScreen. L’[updater Tauri](https://v2.tauri.app/plugin/updater/) n’est pas configuré et n’est pas fourni par le workflow actuel.

## Structure du dépôt

```text
backend/            logique métier Rust, stockage, accès Windows et tests
frontend/           interface React/TypeScript
frontend/src-tauri/ fenêtre desktop, commandes Tauri et packaging Windows
config/             icône et ressources suivies
```
