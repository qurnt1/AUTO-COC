# Plan maître — Refonte AUTO-COC React et Rust

> **État vérifié au 14 octobre 2026.** AUTO-COC est maintenant une application Windows de bureau avec interface React/Vite hébergée par Tauri 2. Le shell et les commandes natives sont en Rust; l’interface appelle le backend via IPC Tauri, sans serveur Axum, routes HTTP locales ni navigateur externe. Le contrôleur Rust conserve macros, réglages, raccourcis, hooks et intégrations.
>
> Le rechargement Telegram est une action native `RELOAD_COC`, pas une macro. Si une fenêtre CoC connue est détectée, Rust lui demande de fermer, attend sa disparition puis lance le chemin CoC configuré; sans fenêtre détectée, il lance directement ce chemin. L’hôte partagé Google Play Games n’est pas tué. La commande Tauri `launch_coc` lance le chemin configuré. Aucune macro système « Valider arrivée » n’est active; au démarrage, les anciens placeholders « Valider arrivée » et « Recharger COC » ne sont retirés que si leur manifeste, leur empreinte et leur contenu vide prouvent qu’ils sont intacts. Les macros créées par l’utilisateur restent des macros ordinaires, quel que soit leur nom.
>
> **Statut du document.** Les sections qui suivent décrivent une proposition antérieure de refonte navigateur + Axum. Elles ne décrivent pas l’architecture actuelle. Les conserver comme proposition alternative uniquement; tout choix futur doit être présenté comme tel.

## Objectif

Refaire AUTO-COC comme application de bureau plus simple à apprendre et à piloter, avec React pour l’interface et Rust pour toute la logique et les accès système. Conserver les parcours utiles de l’application actuelle, rendre les macros et réglages existants récupérables, personnaliser les raccourcis, ajouter une aide intégrée, et ouvrir une PR révisable après validation.

Ce document est le plan maître. Les choix UX et les contraintes d’intégration détaillés sont dans [frontend-ux.md](frontend-ux.md), le portage du noyau est dans [backend-rust.md](backend-rust.md), et les portes de validation et la critique indépendante sont dans [integration-review.md](integration-review.md).

## État constaté

- Dépôt audité sur `main`, commit `9db7951`, identique à `origin/main` au 30 septembre 2026. Aucun code applicatif n'a été modifié pendant l'audit initial.
- Entrée réelle `main.pyw`, interface CustomTkinter; aucun projet React/Vite ou Cargo n'était suivi à la date de l'audit.
- Le parcours fonctionnel principal est création/sélection d’une macro, enregistrement clavier/souris, lecture et arrêt. Les steps JSON utilisent `t`, `type`, `data`. La préparation d’enregistrement dure 3 secondes et l’arrêt enlève les 3 dernières secondes.
- Faits de la baseline historique : raccourcis `F1`, `Ctrl+Shift+1` et `Ctrl+Shift+0`; l’ancienne version contenait des macros système « Recharger COC » et « Valider arrivée ». Ces noms ne décrivent plus des macros système actives.
- `config/data.csv` et les fichiers de macro sont des données ignorées par Git. Ils peuvent contenir des réglages et un secret Telegram en clair. Il faut concevoir l’import autour de fichiers utilisateur réels sans afficher leurs valeurs dans les logs ou documents.
- `VisionService` est un squelette; il n’existe pas de moteur de vision à préserver. Certains éléments annoncés ou présents dans l’UI sont inachevés (menu contextuel, purge Telegram, `gif`). Faire l’inventaire des usages avant de les supprimer ou de les promettre.
- `build/` et `dist/` sont déjà non suivis dans le checkout; ils ne font pas partie du plan ni d’une future PR.

## Proposition historique d’architecture — navigateur + Axum, non retenue

Cette section conserve une ancienne proposition d’architecture. Elle n’est ni la cible confirmée ni une description du produit courant. L’architecture courante est Tauri 2 + React/Vite + backend Rust via IPC, telle que résumée en tête de document.

Un port loopback n'est pas une frontière d'authentification : des pages tierces peuvent tenter des requêtes vers des services locaux, et un site hostile peut tenter une attaque DNS rebinding. Le spike doit valider strictement l'en-tête `Host`, l'origine exacte `http://127.0.0.1:<port>`, une session locale liée au processus, et la protection de chaque route qui change l'état (authentification et contrôle anti-CSRF). Refuser les origines absentes ou tierces pour les mutations, ne pas activer CORS générique, ne pas exposer le jeton dans une query string, un log ou un referer, limiter taille/type des requêtes, et appliquer une CSP sans ressources tierces. Vérifier aussi l'absence de bind sur `0.0.0.0`/IPv6 externe, d'API de shell générique ou de chemin arbitraire.

Décisions de produit recommandées pour préparer la phase 0 :

1. Cibler Windows 10/11 x64 au premier lancement; reporter macOS/Linux et ARM64 tant qu’un besoin n’est pas exprimé.
2. Conserver les fonctions réellement actives de l’app (macros, boucle, raccourcis, chemin CoC, captures, Telegram, diagnostic), avec aide et messages d’erreur améliorés. Garder l’extinction distante derrière une confirmation explicite. Ne pas annoncer la vision ou `gif` avant leur implémentation.
3. Garder le format JSON des macros (`name`, `updated_at`, `sha1`, `steps`) et les champs CSV existants comme format d’import/export jusqu’à validation avec les données installées. Ne jamais réécrire silencieusement une macro inconnue en `nop`.
4. Migrer le token Telegram depuis l’ancien CSV sans en révéler la valeur, puis le garder dans le stockage sécurisé Windows (Credential Manager ou DPAPI, choix à arrêter dans le spike). Le frontend ne reçoit que son état masqué/configuré.
5. Remplacer l’attribution Telegram au premier expéditeur par un appairage/Chat ID explicite. Définir en phase 0 la règle pour groupes (ID du chat et utilisateurs autorisés sont deux identités différentes).
6. Pour la distribution, comparer exécutable portable et installateur utilisateur Windows; vérifier l'ouverture dans les navigateurs supportés, le démarrage/arrêt du service et la préservation des données. Signature et installation hors ligne restent à valider selon le public visé.

## Direction du produit

L’inspiration Perplexity porte sur la hiérarchie et le rythme : navigation latérale stable, une tâche principale mise au centre, recherche immédiatement disponible, commandes proches du contenu, et aide consultable sans masquer le travail. AUTO-COC ne devient pas un produit de questions-réponses : le centre reste une macro, sa séquence et les actions Enregistrer/Lire/Arrêter. L’identité proposée par le plan UX est un atelier sombre mat, avec sauge et cuivre employés avec mesure; pas de fausses données, de cartes identiques, de glow ou d’animations permanentes. Le premier écran doit rendre évident comment créer puis exécuter une macro.

L’URL Mobbin fournie n’était pas accessible au lecteur dans cette session; aucune capture particulière ne sera présentée comme inspectée. L’orientation est tirée du brief et d’un article public de Perplexity sur son menu latéral, puis adaptée au vocabulaire et aux actions d’AUTO-COC. [Mobbin, bibliothèque d’écrans et de parcours](https://mobbin.com/), [Perplexity, annonce de la navigation latérale et de la bibliothèque Web](https://www.perplexity.ai/changelog/what-we-shipped---december-5th).

## Découpage de réalisation

Chaque porte bloque la suivante si elle n’est pas franchie. Les estimations calendaires sont omises faute de capacité et d’échéance fournies.

| Phase | Responsable proposé | Travail | Porte de sortie |
|---|---|---|---|
| 0. Contrat produit et inventaire | Orchestrateur + reviewer | Proposition alternative : établir une matrice des fonctions, données et intégrations pour un éventuel client navigateur + Axum. Ne pas remplacer le contrat IPC Tauri existant sans décision produit explicite. | Proposition évaluée séparément de l’architecture en service. |
| 1. Spike d’une éventuelle version navigateur | Backend Rust + reviewer | Proposition alternative uniquement : évaluer les risques d’un listener loopback, d’origines web et du cycle de vie navigateur. Les tests de sécurité HTTP ne s’appliquent pas au runtime IPC Tauri actuel. | Décision d’architecture séparée avant toute implémentation. |
| 2. Squelette et contrats | Frontend + backend Rust | Proposition alternative : construire React avec un backend Axum/HTTP. Le produit actuel utilise déjà les commandes IPC Tauri et le runtime Rust; ne pas créer de second canal de commande sans besoin établi. | Toute évolution conserve un seul chemin d’autorité vers le contrôleur Rust. |
| 3. Données et migration | Backend Rust + reviewer | Import non destructif des macros JSON, du macro legacy et des réglages CSV depuis les emplacements confirmés. Créer sauvegarde préalable, rapport par fichier, import idempotent, collisions visibles; migration token vers coffre sécurisé. | Comparaison du nombre de macros et steps, ordre/deltas/types, hashes et réglages; deuxième import sans doublons; fichier invalide préservé et erreur compréhensible. Ancienne source conservée. |
| 4. Noyau Rust | Backend Rust | Proposition de portage historique : le contrôleur Rust et les fonctions natives sont déjà présents. Ne pas réintroduire de macros système protégées; les actions CoC doivent rester des commandes natives dédiées. | Toute évolution vérifie les parcours existants et préserve les données utilisateur. |
| 5. Atelier et aide React | Frontend | Construire navigation Atelier/Paramètres/Aide, recherche et bibliothèque, états vides et erreur, activité en cours, statistiques utiles, onboarding reprenable, guide/troubleshooting, diagnostic avec logs expurgés, personnalisateur de raccourcis. | Parcours clavier seul; tests UI Playwright des créations, recherches, réglages et confirmations; écrans vérifiés aux dimensions et DPI cibles; focus, contraste et mouvement réduit validés. |
| 6. Intégrations et sécurité | Frontend + backend Rust + reviewer | Relier raccourcis, Telegram, CoC, captures et boucle à la même machine d’état. Mettre en place l’autorisation Telegram explicite et la confirmation d’actions destructives. Gérer les changements Telegram sans redémarrage ambigu. | Tests de conflits de raccourci et de restauration des défauts; scénarios de bot autorisé/refusé, déconnexion et arrêt; validation avec un bot de test et un PC Windows dédié. |
| 7. Distribution et PR | Orchestrateur + reviewer | Proposition navigateur/Axum : remplacer le lancement natif par un service local et ouvrir un navigateur. Cette voie n’est pas en cours; l’application livrée utilise Tauri. Toute proposition de remplacement doit d’abord comparer packaging, migration et cycle de vie aux capacités Tauri présentes. | Analyse et décision distinctes avant de toucher au lancement, à l’installation ou aux données. |

### Dépendances critiques

- La capture clavier/souris et les raccourcis personnalisés dépendent du spike Windows; les hotkeys assignés ne remplacent pas un enregistreur global de tous les événements.
- Le noyau Rust doit posséder timers et entrées afin qu’une animation ou un re-render React ne retarde pas la macro.
- Le contrôleur d’état précède les branchements de l’interface, des hotkeys et de Telegram; ces surfaces ne doivent pas implémenter chacune leurs propres règles lecture/arrêt.
- La migration et le rollback sont acceptés avant toute suppression de Python ou changement de chemin de stockage.
- Les intégrations réelles Telegram, raccourcis et inputs système exigent une vérification sous Windows; les tests du frontend ou Rust seuls ne les prouvent pas.

## Qualité et validation requises

**Macros et sécurité des entrées :** préserver ordre et timing, mouvements, clics down/up, touches maintenues, scroll et événements spéciaux; stopper à toute erreur et relâcher les entrées même lors d’une fermeture. Valider noms, délais, coordonnées et steps avant sauvegarde ou injection. Les callbacks système mettent à jour le backend et envoient des événements UI; aucun thread hook n’écrit directement dans l’interface.

**Raccourcis :** défauts actuels préremplis, capture explicite, syntaxe claire, conflit interne et refus OS signalés, annulation sans changement, restauration par défaut. Garder un arrêt accessible même si la combinaison globale est indisponible. Les déclenchements sont vérifiés au premier plan, fenêtre minimisée, pendant enregistrement et lecture.

**Aide et ergonomie :** premier parcours sans compte ni Telegram, arrêt toujours visible, expliquer les 3 secondes de préparation et la coupe finale, CTA concrets, états non configuré/connecté/erreur vrais. Aide atteignable depuis navigation et états contextuels. Copie de diagnostic expurgée de tokens et chemins privés.

**Accessibilité et visuel :** focus visible, labels, navigation intégrale au clavier, annonces de changement d’état mesurées, contraste WCAG AA contrôlé, aucun statut par la seule couleur, taille de fenêtre réduite et DPI 100/125/150/200 vérifiés. Réduire/désactiver les animations système, mouvements uniquement quand ils montrent le résultat d’une action.

**Validation automatique prévue :** `cargo fmt --check`, Clippy sans avertissements nouveaux, tests Rust ciblés, vérification TypeScript/formatteur/linter/build React, tests navigateur/UI et build du binaire Rust Windows. Ajouter une matrice pour formats/collisions/machine d'état; utiliser un injecteur de test et une horloge contrôlée pour éviter des mouvements réels pendant les tests. Les vérifications d'intégration décrites ici sont des critères de réalisation, pas des résultats déjà obtenus.

**Validation manuelle prévue :** installation/lancement en contexte Windows utilisateur standard; essai de la page dans un navigateur réel, permissions/refus Telegram, raccourcis personnalisés, capture et lecture sur application inoffensive, boucle et arrêt, chemin `.exe`/`.lnk`, persistance des données. Tester les protections depuis une page tierce et rapporter les limites OS séparément des tests CI.

Les seuils de timing et de performance seront définis après le spike sur le PC cible. Le selftest Python actuel ne mesure qu’une attente de cinq secondes à dérive globale <5 %; il ne prouve pas la précision d’injection Windows.

## Rollback et publication

- Copier les anciennes données et produire une sauvegarde avant toute écriture; ne jamais déplacer/supprimer l’original automatiquement.
- Garder l’ancien lancement Python possible pendant la validation bêta, mais ne pas faire tourner simultanément deux pollers Telegram ou deux systèmes de hooks sur la même session Windows.
- S’assurer que la nouvelle version sait relire/exporter les JSON historiques sans enrichissement obligatoire. Si un format évolue plus tard, utiliser une migration versionnée et une sauvegarde restaurable séparée.
- La PR peut supprimer le code Python après validation de la migration, mais ne doit jamais inclure `build/`, `dist/`, `config/data.csv`, macros locales, secrets, caches ou binaires générés.
- Ne pas commit/push implicitement. La création de PR intervient à la phase 7, conformément à la demande, après inspection du diff et des vérifications. Aucune publication ou release signée n’est comprise dans le plan sans décision distincte.

## Risques qui peuvent arrêter ou réorienter le projet

1. **Capture ou injection incompatible avec un jeu/configuration élevée :** reproduire `pynput` au spike sans demander une élévation permanente; si impossible, expliciter la limite avant le portage complet.
2. **Altération d’une macro :** conserver les JSON source et rejeter les steps inconnus plutôt que les normaliser silencieusement; vérifier les collisions à l’import et au renommage.
3. **Attribution Telegram ouverte :** exiger appairage explicite; séparer ID de conversation et ID d’utilisateur autorisé.
4. **Secret local révélé :** aucun token dans l’état React, rapports de diagnostic ou log; coffre Windows testé sur un compte standard.
5. **Boucle ou arrêt incohérent entre surfaces :** une seule machine d’état Rust reçoit les commandes UI, hotkey et Telegram.
6. **Service local ou navigateur indisponible :** signaler clairement si le backend ne démarre pas ou si le navigateur ne peut pas s'ouvrir; permettre un lancement manuel à l'URL locale sans exposer le service au réseau. Valider le comportement sur les navigateurs Windows ciblés.
7. **Perte du contrôle pendant arrêt/plantage :** test de toutes les voies de sortie (stop UI/raccourci, erreur, fermeture) et release garanti des touches/boutons.
8. **Dérive de périmètre sous l’étiquette “100 %” :** traduire cette demande en matrice de fonctions effectivement actives; traiter vision et callbacks morts comme choix nouveaux, jamais comme parité implicite.

## Arbitrages confirmés et restant à préciser

Le runtime vérifié est React/Vite dans Tauri 2, avec backend Rust et IPC Tauri. Le navigateur + Axum décrit plus haut reste une proposition alternative historique. Les détails du packaging et les scénarios d’autorisation Telegram qui ne sont pas couverts par le code restent à vérifier; ils ne changent pas l’architecture constatée.

## Sources vérifiées

- [Axum, serveur HTTP Rust asynchrone](https://docs.rs/axum/latest/axum/)
- [Tokio, port localhost éphémère avec `TcpListener`](https://docs.rs/tokio/latest/tokio/net/struct.TcpListener.html)
- [Microsoft, hooks bas niveau `SetWindowsHookEx`](https://learn.microsoft.com/fr-fr/windows/win32/api/winuser/nf-winuser-setwindowshookexw)
- [Microsoft, injection clavier/souris et limites UIPI de `SendInput`](https://learn.microsoft.com/fr-fr/windows/win32/api/winuser/nf-winuser-sendinput)
