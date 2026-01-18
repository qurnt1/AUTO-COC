# 🎮 AUTO-COC v3.0

<p align="center">
  <img src="config/image.png" alt="AUTO-COC" width="120"/>
</p>

<p align="center">
  <strong>Application d'automatisation macros avec contrôle Telegram</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/CustomTkinter-5.2+-green" alt="CTk"/>
  <img src="https://img.shields.io/badge/Telegram-Bot%20API-0088cc?logo=telegram" alt="Telegram"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## 📋 Description

**AUTO-COC** est une application d'enregistrement et lecture de macros (clavier + souris) avec une interface moderne et un **contrôle à distance via Telegram**. Idéale pour automatiser des tâches répétitives sur Clash of Clans ou tout autre jeu/application.

### ✨ Points forts

| Fonctionnalité | Description |
|----------------|-------------|
| 🎯 **Précision temporelle** | Timing haute-précision avec `perf_counter` |
| 🤖 **Contrôle Telegram** | Lancez, stoppez, capturez l'écran depuis votre téléphone |
| 🔄 **Mode boucle** | Répétition automatique des macros |
| 💾 **Sauvegarde atomique** | Aucune perte de données en cas de crash |
| 🌙 **Interface sombre** | Design moderne avec CustomTkinter |

---

## 🚀 Installation

### Prérequis

- **Python 3.10+** 
- **Windows 10/11** (macOS/Linux non testé)

### Installation rapide

```bash
# Cloner ou télécharger le projet
cd "AUTO-COC"

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python main.py
```

### Dépendances principales

| Package | Utilisation |
|---------|-------------|
| `customtkinter` | Interface graphique moderne |
| `python-telegram-bot` | API Telegram asynchrone |
| `pynput` | Capture clavier/souris |
| `keyboard` | Raccourcis globaux |
| `Pillow` | Captures d'écran |

---

## 🎮 Utilisation

### Interface locale

1. **Créer une macro** → Bouton `Nouveau`
2. **Enregistrer** → Cliquez `Enregistrer`, attendez le bip, effectuez vos actions
3. **Stopper** → Cliquez `Stopper` (les 3 dernières secondes sont auto-coupées)
4. **Lire** → Sélectionnez la macro et cliquez `Lire`

### Raccourcis clavier

| Raccourci | Action |
|-----------|--------|
| `F1` | Toggle lecture/arrêt |
| `Ctrl+Shift+1` | Lancer la macro |
| `Ctrl+Shift+0` | Stopper |

### Contrôle Telegram

```
┌─────────────────┬─────────────┐
│ Paramètres ⚙️   │ Capture 📸  │
├─────────────────┴─────────────┤
│         Lancer CoC            │
├───────────────┬───────────────┤
│   Lancer ✅   │   Stop ❌     │
└───────────────┴───────────────┘
```

**Commandes texte :** `stop`, `go`, `menu`, `capture`, `shutdown`

---

## ⚙️ Configuration

### 1. Configurer Telegram

1. Créez un bot avec [@BotFather](https://t.me/BotFather)
2. Copiez le **Token**
3. Envoyez un message à votre bot, puis récupérez votre **Chat ID**
4. Dans l'app : `Paramètres → Configurer Telegram...`

> 💡 Un guide HTML détaillé est inclus : `Paramètres → Ouvrir le guide`

### 2. Chemin CoC (optionnel)

Pour le bouton "Lancer CoC", renseignez le chemin vers :
- L'exécutable `.exe` du jeu, **ou**
- Un raccourci `.lnk`

---

## 📁 Architecture v3.0

```
Macro_COC/
├── main.py                 # Point d'entrée
├── requirements.txt
├── models/
│   └── macro.py            # Dataclasses Step/Macro
├── services/
│   ├── telegram_service.py # Bot Telegram async
│   ├── recorder_service.py # Recorder/Player
│   └── vision_service.py   # Computer Vision (v3.1)
├── gui/
│   ├── app.py              # Fenêtre principale
│   ├── dialogs.py          # Popups
│   ├── components.py       # Widgets
│   └── theme.py            # Couleurs
├── utils/
│   ├── logger.py           # Logging rotatif
│   ├── config.py           # I/O CSV/JSON
│   └── system.py           # Processus, shutdown
└── config/
    ├── macros/             # Fichiers JSON
    ├── data.csv            # Configuration
    └── app.log             # Logs
```

---

## 🆕 Nouveautés v3.0

- **Architecture modulaire** — Code séparé en packages maintenables
- **Telegram async** — Migration vers `python-telegram-bot` v21+
- **Dataclasses** — Modèles typés pour les macros
- **Écriture atomique** — Sauvegarde sécurisée des fichiers
- **Communication thread-safe** — Queue entre Telegram et GUI
- **Selftests intégrés** — `python main.py --selftest`

---

## 🧪 Tests

```bash
# Lancer les tests internes
python main.py --selftest
```

---

## 📜 Licence

**MIT** — Libre d'utilisation et modification.

---

<p align="center">
  <sub>Made with ❤️ for automation enthusiasts</sub>
</p>
