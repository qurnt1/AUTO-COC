Macro COC (v2.1.0)

Macro COC est une application de macro-enregistrement (recording) et de lecture (playback) conçue pour automatiser des tâches dans Clash of Clans. Elle combine une interface graphique de bureau (UI) pour l'enregistrement et la gestion locale, avec un bot Telegram puissant pour le contrôle à distance.

Lancez vos macros, démarrez le jeu, ou même éteignez votre PC depuis n'importe où via de simples commandes Telegram.

1. Installation

Prérequis

Python 3.7+

Un fichier requirements.txt (que vous avez mentionné être dans votre projet)

Étapes d'installation

Clonez ce dépôt (ou dézippez vos fichiers) dans un dossier.

Ouvrez un terminal ou une invite de commande dans ce dossier.

Installez les dépendances Python nécessaires en utilisant le fichier requirements.txt :

pip install -r requirements.txt


Une fois l'installation terminée, lancez l'application :

python macro_coc_v2.py


2. Présentation et Fonctionnement

L'application vous permet d'enregistrer une séquence d'actions (clics de souris, mouvements, frappes au clavier) et de la sauvegarder en tant que "macro". Vous pouvez ensuite rejouer cette macro à volonté, soit depuis l'interface, soit depuis Telegram.

Fonctionnalités principales

Interface de bureau (UI): Une interface claire (basée sur CustomTkinter) pour créer, renommer, supprimer et gérer vos macros.

Enregistrement/Lecture: Enregistrement précis des événements pynput avec gestion des délais.

Mode Boucle: Un bouton unique (dans les Paramètres et sur Telegram) vous permet d'activer ou de désactiver la lecture en boucle de votre macro.

Contrôle via Telegram: Un bot Telegram vous donne un contrôle total à distance.

Lancement de CoC: Un bouton dans l'UI et sur Telegram vous permet de lancer Clash of Clans. Le bouton Telegram se met à jour intelligemment pour afficher "COC lancé ✅" lorsque le jeu est détecté.

Actions à distance:

Démarrer / Stopper la macro.

Choisir quelle macro exécuter.

Prendre une capture d'écran (/capture).

Éteindre votre PC (📴 Éteindre PC).

Recharger le jeu (🔃 Recharger COC - lance une macro protégée).

Gestion de la vie privée: L'application purge les anciens messages du bot dans votre conversation Telegram à chaque démarrage.

Maintenance: Les fichiers de logs de plus de 24 heures sont automatiquement supprimés au démarrage et à la fermeture.

3. Configuration

Pour une utilisation complète, deux éléments doivent être configurés via le bouton Paramètres dans l'interface principale.

A. Chemin de lancement CoC

Où : Paramètres -> Chemin de lancement CoC

Quoi : Indiquez le chemin complet vers l'exécutable (.exe) de Clash of Clans ou, de préférence, vers son raccourci (.lnk).

Pourquoi : Permet à l'application (et à Telegram) de lancer ou de relancer le jeu.

B. Connexion à Telegram

Pour connecter l'application à Telegram, vous avez besoin de deux choses : un Token de Bot et votre Chat ID.

La méthode la plus simple est d'utiliser le guide intégré à l'application :

Dans l'application de bureau, cliquez sur Paramètres.

Cliquez sur Configurer Telegram....

Dans la nouvelle fenêtre, cliquez sur Ouvrir le guide (page HTML locale).

Ce guide HTML (stocké localement dans votre dossier config/) vous expliquera pas à pas comment :

Parler à @BotFather sur Telegram pour créer votre propre bot et obtenir un Token (ex: 123456:ABC-DEF1234...).

Envoyer un message à votre nouveau bot pour trouver votre Chat ID (ex: 987654321).

Une fois ces deux informations obtenues, copiez-les dans les champs "Bot token" et "Chat ID" de la fenêtre de configuration et cliquez sur "Enregistrer".

Si tout est correct, le statut dans les paramètres passera à "Connecté" et vous recevrez un message de démarrage (Macro COC v2.1.0 lancée.) sur votre téléphone.

4. Utilisation

Enregistrer une Macro:

Lancez l'application.

Cliquez sur Nouveau, donnez un nom à votre macro (ex: "Collecter ressources").

Cliquez sur Enregistrer.

... (Attendez le décompte de 3 secondes) ...

Effectuez vos actions dans le jeu.

Cliquez sur Stopper la macro (ou F1) pour terminer l'enregistrement.

Jouer une Macro (Local):

Assurez-vous que la macro est sélectionnée dans la liste de gauche.

Activez la boucle si nécessaire (Paramètres -> Lecture en boucle).

Cliquez sur Lire la macro (ou F1).

Jouer une Macro (Telegram):

Ouvrez la conversation avec votre bot sur Telegram.

Utilisez les boutons Lancer ✅ ou Stop ❌.

Pour changer de macro, allez dans Paramètres ⚙️ -> Choisir macro.

5. Raccourcis Clavier

F1 : Démarrer / Stopper la macro sélectionnée.

Ctrl+Shift+1 : Démarrer la macro.

Ctrl+Shift+0 : Stopper la macro (arrêt d'urgence).