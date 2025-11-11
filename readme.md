# 🎮 **Macro COC v2.1.0**

> Macro COC est une application de macro-enregistrement (recording) et de lecture (playback) conçue pour automatiser des tâches. Elle combine une interface graphique de bureau (UI) pour l'enregistrement et la gestion locale, avec un bot Telegram puissant pour le contrôle à distance. Lancez vos macros, démarrez le jeu, ou même éteignez votre PC depuis n'importe où via de simples commandes Telegram. 

---

## 🚀 Installation

### 🧩 **Prérequis**
* **Python 3.x** installé sur votre système  
* **pip** (installateur de paquets Python)

### 📦 **Installation des dépendances**
Le projet inclut un fichier `requirements.txt` contenant toutes les dépendances nécessaires.  
Exécutez simplement la commande suivante :

```bash
pip install -r requirements.txt
```

### ▶️ **Lancement de l’application**
Une fois les dépendances installées, lancez l’application via :

```bash
python macro.py
```

---

## ✨ Fonctionnalités

### 🧠 **Gestion complète des macros (UI)**
* Créer, renommer, dupliquer, importer et exporter vos macros.  
* Interface graphique moderne basée sur **CustomTkinter**.  
* Sauvegarde automatique des macros en JSON local.  

### 🖱️ **Enregistrement et lecture**
* Enregistre précisément vos actions clavier et souris.  
* Lecture fidèle et ajustée au temps réel.  
* Possibilité de **lecture en boucle** (activable dans l’UI et sur Telegram).  

### 🔁 **Mode Boucle**
* Un **toggle unique** permet d’activer/désactiver le mode boucle.  
* L’état est synchronisé entre l’UI et Telegram.  

### 🤖 **Contrôle à distance (Bot Telegram)**
L’application peut être entièrement pilotée depuis votre téléphone :
* **Démarrer / Stopper** la macro en cours.  
* **Choisir** la macro à exécuter.  
* **Basculer le mode boucle**.  
* **Prendre une capture d’écran** avec `/capture`.  
* **Éteindre le PC** à distance (`📴 Éteindre PC`).  
* **Recharger le jeu** via la macro spéciale `🔃 Recharger COC`.  

Le clavier Telegram est dynamique et clair :
```
[Paramètres ⚙️] [Capture 📸]
[Lancer COC]
[Go ✅] [Stop ❌]
```

### 🕹️ **Lancement automatique du jeu**
* Un bouton dans l’UI permet de lancer **Clash of Clans** directement.  
* Le bouton Telegram s’adapte automatiquement :  
  `COC lancé ✅` si le processus du jeu est détecté.  

### 🧼 **Maintenance automatique**
* À chaque démarrage, le bot **purge les anciens messages Telegram**.  
* Les fichiers de logs de plus de **24 h** sont automatiquement supprimés.  

---

## ⚙️ Configuration

### 1️⃣ **Chemin de lancement CoC**
* Renseignez le chemin vers l’exécutable ou le raccourci `.lnk` du jeu.  
* Obligatoire pour que le bouton **Lancer COC** fonctionne.  

### 2️⃣ **Connexion à Telegram**
* Fournissez votre **Token de Bot** et votre **Chat ID**.  
* L’application propose un **guide complet en HTML** :  
  `Paramètres → Configurer Telegram... → Ouvrir le guide`.  
  Ce guide explique pas à pas comment :
  * Créer un bot avec `@BotFather`  
  * Récupérer votre **Token**
  * Trouver votre **Chat ID**

---

## 🕹️ Utilisation

1. **Créer une macro :**
   * Cliquez sur `Nouveau`, nommez-la, puis sur `Enregistrer`.  
   * Attendez le décompte, effectuez vos actions, puis `Stopper`.  

2. **Lire une macro (localement) :**
   * Sélectionnez une macro.
   * Activez la boucle si nécessaire.
   * Cliquez sur `Lire la macro`.  

3. **Lire une macro (Telegram) :**
   * Utilisez les boutons `Go ✅` et `Stop ❌` sur votre téléphone.  

---

## ⌨️ Raccourcis Clavier

| Raccourci | Action |
|------------|--------|
| `F1` | Démarrer / Stopper la macro selectionnée|

---

## 📁 Structure du projet

```
Macro_COC/
├── config/
│   ├── macros/
│   │   ├── macro1.json
│   │   ├── macro2.json
│   │   └── etc etc
│   ├── data.csv
│   ├── app.log
│   ├── icon.ico
│   └── image.png
│    
├── requirements.txt
└── macro_coc_v2.py
```


## 🧾 Licence

Projet open-source sous licence **MIT**.  
Libre de l’utiliser, modifier (sur votre pc) et redistribuer.  

---

