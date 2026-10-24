# Plan frontend UX — AUTO-COC

## Cap et faits vérifiés

**État vérifié au 14 octobre 2026 :** le frontend est une interface React/Vite hébergée dans la fenêtre Windows Tauri 2. Les commandes passent par l’IPC Tauri vers le runtime/backend Rust; l’application ne sert pas un site dans un navigateur et n’utilise pas Axum ni des routes HTTP locales.

L’atelier actuel contient une bibliothèque filtrable, la création, le renommage et la suppression des macros, ainsi que l’enregistrement, la lecture et l’arrêt. Aucune macro système n’est créée ou protégée. « Valider arrivée » n’est pas une action active. Le callback Telegram `RELOAD_COC` appelle le rechargement natif Rust : si une fenêtre CoC détectée est ouverte, le backend lui envoie une demande de fermeture, attend sa disparition puis lance le chemin configuré; sans fenêtre détectée, il lance directement ce chemin. Il ne tue pas l’hôte partagé Google Play Games. La commande Tauri `launch_coc` lance le chemin configuré.

Le processus Tauri/Rust gère les accès natifs, les hooks, les raccourcis, les fichiers et les intégrations. L’interface reprend l’icône réelle de l’application et une palette parchemin, forêt et laiton inspirée de Clash of Clans. L’aide et les paramètres restent dans l’application desktop.

**Statut des sections suivantes :** les routes `/macros`, `/settings` et `/help`, le wireframe et les choix de style détaillés plus bas sont des propositions UX, pas des descriptions de l’interface actuelle. Elles restent des pistes de refonte future à évaluer dans le shell Tauri existant.

## Structure et parcours

- Proposition : « Atelier » avec bibliothèque recherchable, macro sélectionnée et commandes au centre. Les macros ordinaires restent renommables et supprimables; les commandes natives CoC ne sont pas des macros.
- `/settings` avec onglets « Général », « Raccourcis » et « Telegram » : boucle, chemin CoC, personnalisation des touches, secrets masqués et état réel de Telegram. État système/logs accessibles depuis « Diagnostic ».
- `/help` : démarrage rapide, création d’une macro, raccourcis par défaut et personnalisés, connexion Telegram et résolution des erreurs fréquentes. Aide contextuelle dans les états vides et à côté des réglages complexes.
- Premier lancement : checklist facultative, fermable et reprenable. Elle mène à une première macro, explique le délai de préparation et la coupe des 3 secondes finales, puis propose CoC et Telegram comme étapes facultatives. Aucun compte, token ou service distant n’est requis pour démarrer.

## Direction artistique

Une console de travail précise inspirée du vocabulaire d’un poste de contrôle de village : surfaces mates, lignes de séparation fines, rythme typographique net et une seule signature graphique, le repère de séquence qui s’allume pendant l’enregistrement/lecture. Éviter le camouflage, le néon gamer, les gradients décoratifs et les grilles de cartes identiques. Aligner le contenu à gauche; réserver le centrage aux confirmations courtes.

Palette de base :

- Basalte, fond : `#11191B`
- Ardoise, surface : `#1D292B`
- Lichen, séparation et éléments secondaires : `#3A4A49`
- Craie, texte principal : `#E8E5DA`
- Sauge, action et sélection : `#A8C4B3`
- Cuivre, enregistrement/attention : `#D28B69`

Les couleurs d’état d’erreur et de réussite restent sémantiques; vérifier tous les couples de texte et de focus en WCAG AA avant validation. Typo proposée : IBM Plex Sans pour l’interface, IBM Plex Mono uniquement pour durées, cycles et raccourcis. Réutiliser l’icône AUTO-COC existante si elle convient au nouveau shell. Libellés en casse phrase, verbes explicites (« Enregistrer », « Lire », « Arrêter »).

Wireframe desktop (≥1280 px) :

```text
┌───────────────┬────────────────────┬───────────────────────────────────┐
│ AUTO-COC      │ Atelier            │ Prêt                    Paramètres│
│ Atelier       │ [Rechercher…]       │ Macro sélectionnée       [Boucle]│
│ Paramètres    │ Macro personnelle  │                                   │
│ Aide          │ Macro personnelle  │       [Enregistrer] [Lire] [Stop] │
│               │ …                  │                                   │
│               │ [+ Nouvelle macro] │ Durée   Événements   Cycles   Temps│
│ statut local  │                    │ Aide rapide / diagnostic contextuel│
└───────────────┴────────────────────┴───────────────────────────────────┘
```

La zone centrale garde les commandes visibles et hiérarchisées; « Stop » reste immédiatement repérable. Les statistiques ne deviennent jamais des chiffres décoratifs : afficher les valeurs réellement connues et leur période. Prévoir une prévisualisation compacte de la séquence issue des événements stockés (temps/type), seulement si elle aide à comprendre une macro et sans la faire passer pour un éditeur; sinon l’omettre du premier lot.

## États et micro-interactions

Proposition : concevoir explicitement le chargement initial; aucune macro; aucune sélection; macro vide; recherche sans résultat; enregistrement en attente puis actif; lecture et boucle; arrêt; Telegram non configuré/en attente/connecté/en erreur; chemin CoC manquant; erreur de sauvegarde ou de lancement. Chaque écran vide indique la prochaine action. Les erreurs nomment le problème et sa correction. Ne pas afficher « CoC lancé » ou « Telegram connecté » sans confirmation du runtime.

Pendant une action, désactiver les modifications incompatibles comme aujourd’hui, expliquer brièvement pourquoi et garder Stop disponible. À l’enregistrement, indiquer le compte à rebours de préparation et rappeler la coupe automatique finale. À l’arrêt, confirmer l’enregistrement et la durée réelle. La boucle est décrite comme s’appliquant au prochain lancement si elle ne peut pas modifier la lecture en cours.

Raccourcis : préremplir F1 (basculer lecture/arrêt), Ctrl+Shift+1 (lire), Ctrl+Shift+0 (arrêter). Dans les paramètres, clic sur « Modifier », capture d’une combinaison, aperçu lisible, validation des doublons internes et retour clair si l’hôte refuse l’enregistrement. Ne jamais permettre de retirer la dernière commande globale d’arrêt; toujours garder le bouton Stop visible. Ignorer les raccourcis pendant la saisie/capture d’une touche. Tester les conflits OS en environnement réel.

Mouvement ciblé : transition courte uniquement après une action (bouton qui devient « Arrêter », confirmation de sauvegarde, panneau d’aide qui s’ouvre); un signal discret pendant l’enregistrement/lecture; insertion de la nouvelle macro dans la liste. Pas de cascade de panneaux animés ni d’autoplay décoratif. Respecter « réduire les animations » et rendre l’état compréhensible sans mouvement.

## Responsive et accessibilité

Cible desktop d’abord, en vérifiant au minimum 1024×640 (taille minimale actuelle), 1280×800 et 1600×900. À largeur réduite, transformer la bibliothèque en panneau repliable; garder Stop et le statut accessibles, empiler les statistiques et permettre le scroll du contenu. Ne pas réduire les commandes à des icônes sans libellé.

Navigation intégrale au clavier, ordre de focus stable, focus visible, noms accessibles pour les boutons/icônes, contrastes AA, annonces `aria-live` sobres pour les changements d’état, dialogues avec focus contenu et retour au déclencheur. Ne pas dépendre de la couleur seule pour distinguer Prêt/Enregistrement/Lecture/Erreur. Les modales destructives (suppression, extinction) expliquent l’effet et demandent une confirmation explicite.

## Priorité et acceptation

1. **Socle et atelier (proposition)** : shell Tauri/React, bibliothèque, sélection, création/renommage/suppression, lecture/enregistrement/arrêt, boucle, indicateurs et états vides. Acceptation : les macros restent des données utilisateur ordinaires; les actions natives comme recharger CoC ne sont pas représentées par une macro; les valeurs d’état suivent le backend Rust.
2. **Réglages utiles** : conserver les raccourcis par défaut et ajouter leur édition avec validation; régler boucle/chemin CoC; configurer Telegram avec token masqué; afficher son état et diagnostics. Acceptation : redémarrer l’app conserve les réglages; aucune valeur secrète n’apparaît dans le rendu, les logs UI ou l’aide.
3. **Onboarding et aide** : checklist facultative, guide local cohérent, conseils liés aux états et erreurs compréhensibles. Acceptation : utilisateur neuf peut créer puis arrêter une macro sans Telegram; il peut retrouver les aides ensuite.
4. **Finition** : responsive, mouvement, accessibilité, revue visuelle et corrections. Acceptation : tailles cibles lisibles, parcours clavier complet, reduced-motion respecté, contraste contrôlé.

Validation à prévoir à l’implémentation : vérification visuelle des tailles ci-dessus; parcours Playwright des routes, états vides, paramètres/raccourcis et confirmations; tests d’accessibilité/clavier; puis smoke test Windows avec une macro courte dans une application sans risque, focus ailleurs pour les hooks globaux, commandes Telegram si configuré, lancement CoC et arrêt sécurisé. Les tests UI ne prouvent pas le fonctionnement des hooks ou de Telegram; ceux-ci nécessitent ces vérifications réelles.

Référence de navigation (principe seulement) : [Perplexity décrit la bibliothèque, les fils récents et les raccourcis de sa barre latérale Web](https://www.perplexity.ai/changelog/what-we-shipped---december-5th). Le lien Mobbin fourni n’est pas accessible par le lecteur de cette session; ses écrans précis restent à revoir directement avant l’implémentation visuelle.
