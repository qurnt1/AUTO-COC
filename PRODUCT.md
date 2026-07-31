# Product

<!-- impeccable:product-schema 1 -->

## Platform

adaptive

## Users

Utilisateur Windows qui lance régulièrement des macros clavier/souris pendant qu’une autre application est au premier plan.

## Product Purpose

AUTO-COC enregistre, stocke et rejoue des séquences clavier/souris, avec contrôle local et distant via Telegram. Le succès se mesure à la fiabilité de l’exécution, à la visibilité immédiate de l’état et à la capacité d’arrêter une macro sans ambiguïté.

## Positioning

Un outil d’automatisation local orienté opérateur, avec timing précis, commandes globales et télécommande Telegram.

## Operating Context

L’application est utilisée sur Windows, souvent en arrière-plan pendant que l’utilisateur interagit avec Clash of Clans ou une autre application. Les macros sont stockées localement en JSON et les paramètres en CSV.

## Capabilities and Constraints

- Enregistrement clavier/souris via `pynput`.
- Lecture à timing absolu, boucle optionnelle et arrêt d’urgence.
- Macros système protégées.
- Raccourcis globaux `F1`, `Ctrl+Shift+1` et `Ctrl+Shift+0`.
- Contrôle Telegram asynchrone dans un thread séparé.
- Capture d’écran, lancement de CoC et extinction Windows.
- Profil de lancement CoC configurable par chemin, processus et titres de fenêtre.
- Détection de présence CoC et safeguard qui arrête la lecture après plusieurs contrôles absents.
- L’interface finale utilise PyQt6 exclusivement.

## Brand Commitments

Le nom AUTO-COC et les assets présents dans `config/` sont conservés. L’interface adopte une console opérateur sombre, sobre et très lisible.

## Evidence on Hand

- Modèles de macros dans `models/macro.py`.
- Services d’enregistrement, de lecture et Telegram dans `services/`.
- Macros existantes dans `config/macros/`.

## Product Principles

1. L’état d’exécution et la présence de CoC doivent être compréhensibles en moins d’une seconde.
2. L’arrêt doit rester disponible et sans ambiguïté.
3. Les opérations distantes et locales doivent suivre le même modèle d’état.
4. Les données existantes ne doivent pas être réécrites dans un format incompatible.
5. Les routines système ne doivent pas être confondues avec les macros éditables de l’utilisateur.

## Accessibility & Inclusion

Navigation clavier, focus visible, contraste renforcé, taille de texte compatible avec le scaling Windows et messages d’erreur explicites.
