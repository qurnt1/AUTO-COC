# AUTO-COC

AUTO-COC enregistre et rejoue des macros clavier/souris sur Windows. L’interface React s’ouvre dans le navigateur; un service Rust local gère les macros, les raccourcis globaux, Telegram et les accès Windows.

## Prérequis

- Windows 10 ou 11.
- Rust et Cargo, installés avec [rustup](https://rustup.rs/).
- Node.js `20.19+` ou `22.12+`, avec npm, pour construire l’interface React.

## Construire et lancer

Depuis la racine du dépôt, construire d’abord les fichiers React, puis lancer le backend Rust en mode release :

```powershell
cd frontend
npm ci
npm run build
cd ..
cargo run --release --manifest-path backend/Cargo.toml
```

Le backend intègre `frontend/dist` dans l’exécutable, choisit un port local sur `127.0.0.1` et ouvre l’interface dans le navigateur par défaut. Fermer l’onglet ne termine pas le service Rust. Laissez la fenêtre du terminal ouverte pendant l’utilisation; pour quitter, utilisez Réglages > Actions locales > Quitter AUTO-COC lorsque l’application est au repos, ou `Ctrl+C` dans le terminal.

Pour compiler sans lancer l’application :

```powershell
cargo build --release --manifest-path backend/Cargo.toml
```

La commande release exige que `frontend/dist` existe. En mode debug, `cargo run --manifest-path backend/Cargo.toml` fonctionne aussi sans build React, mais le backend sert alors une page indiquant qu’il faut construire l’interface.

## Développement de l’interface

Lancer d’abord le backend Rust en mode debug et relever l’adresse `127.0.0.1:PORT` affichée dans le terminal. Dans un second terminal PowerShell :

```powershell
cd frontend
npm ci
$env:AUTO_COC_DEV_BACKEND_URL = "http://127.0.0.1:PORT"
npm run dev
```

Remplacez `PORT` par le port affiché par le backend, puis ouvrez l’adresse Vite indiquée par npm (par défaut `http://127.0.0.1:5173`). Le proxy Vite transmet `/api` au backend. Cette origine de développement est acceptée uniquement par un backend compilé en mode debug.

## Utiliser AUTO-COC

Créez une macro dans l’atelier, sélectionnez-la, puis démarrez l’enregistrement. La capture commence après trois secondes de préparation. À l’arrêt, les trois dernières secondes sont retirées; un enregistrement de trois secondes ou moins produit une macro vide. La lecture reprend les événements enregistrés et peut être interrompue depuis l’atelier ou avec un raccourci.

Les raccourcis initiaux sont :

| Raccourci | Action |
| --- | --- |
| `F1` | Démarrer la macro sélectionnée au repos ou arrêter une lecture en cours |
| `Ctrl+Shift+1` | Démarrer la lecture |
| `Ctrl+Shift+0` | Arrêter la lecture ou finaliser l’enregistrement |

Les raccourcis peuvent être modifiés dans Réglages. Windows doit accepter chaque combinaison; si une combinaison est déjà utilisée, l’application reste accessible et permet d’en choisir une autre.

Telegram est facultatif. Configurez votre bot dans Réglages, créez un code d’appairage, puis envoyez `/start CODE` au bot depuis une conversation privée. L’ancien identifiant de chat n’est pas repris comme autorisation. Les commandes texte disponibles incluent `stop`, `go`, `menu`, `capture`, `shutdown`, `relancer` et `launch`. La commande `gif` reçoit une réponse indiquant que la capture GIF n’est pas prise en charge.

La page Aide de l’application décrit les commandes, les raccourcis et la migration des données d’une ancienne installation. Pour migrer, choisissez le dossier de l’ancien projet ou son sous-dossier `config`. Une sauvegarde CSV expurgée du token est créée avant l’import; les fichiers source restent à leur emplacement. Les macros illisibles ou incompatibles sont signalées et ne sont pas rejouées. Les données de la version Rust sont stockées sous `%LOCALAPPDATA%\AUTO-COC`; le token Telegram y est conservé séparément et protégé pour l’utilisateur Windows courant.

## Vérifications Rust

Depuis le dossier `backend` :

```powershell
cargo fmt --check
cargo check --all-targets
cargo test
```

## Structure

```text
backend/    service Rust/Axum, accès Windows et tests
frontend/   interface React/TypeScript
config/     ressources suivies et fichiers locaux historiques ignorés par Git
```
