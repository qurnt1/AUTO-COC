# Revue d'intégration et essai de découverte AUTO-COC

## État actuel vérifié

Au 14 octobre 2026, AUTO-COC est une application Windows Tauri 2 avec frontend React/Vite et backend entièrement Rust. Le frontend parle au runtime Rust par IPC Tauri; l’application n’utilise ni serveur Axum, ni API HTTP loopback, ni navigateur externe.

Le callback et la commande texte Telegram `relancer` utilisent `RELOAD_COC`. Le backend Rust demande la fermeture de la fenêtre CoC reconnue, attend sa disparition, puis lance le chemin configuré; s’il ne trouve aucune fenêtre CoC, il lance directement ce chemin. Il ne tue pas le processus partagé de l’hôte Google Play Games. La commande Tauri `launch_coc` lance le chemin configuré. Il n’existe plus de macro système active « Valider arrivée » ni de macro système « Recharger COC ». La migration de démarrage peut supprimer les anciens placeholders de ces noms seulement si leur manifeste, empreinte et contenu vide le prouvent; les données utilisateur modifiées ou importées sont préservées.

Les parcours HTTP et navigateur de ce document sont des propositions de revue héritées d’une architecture alternative. Ils ne sont pas des validations requises du runtime Tauri actuel.

## Base de revue

Les faits ci-dessous viennent de l'audit statique initial du dépôt `main`, commit `9db7951d418012c951cf301bf8e6c7b225d076d5`, au 30 septembre 2026. Ce commit contenait une application Windows Python 3.10+, entrée réelle `main.pyw`, UI CustomTkinter, capture/lecture `pynput`, hotkeys via `keyboard`, bot Telegram et fichiers sous `config/`. Le README dit `python main.py`, mais ce fichier d'entrée n'existe pas.

- Le dépôt initial n'avait pas de sources de tests suivies, pas de configuration Cargo/npm et pas de pipeline CI. Les quatre « selftests » de `main.pyw` couvrent un modèle, des fichiers temporaires, une attente réelle de cinq secondes et le contraste d'une seule paire de couleurs. Ils ne démontrent pas l'exactitude des hooks Windows.
- À cette ancienne baseline seulement, les macros étaient des JSON `{name, updated_at, sha1, steps}` et deux macros système protégées portaient les noms « Recharger COC » et « Valider arrivée ». Cette description historique ne correspond plus au produit actuel : ces noms ne sont pas des macros système actives.
- Les raccourcis observés dans le code : `F1` bascule lecture/arrêt, `Ctrl+Shift+1` lit, `Ctrl+Shift+0` arrête. L'enregistrement clavier/souris est distinct des raccourcis d'action et démarre après trois secondes; l'arrêt coupe les trois dernières secondes.
- Le CSV local ignoré par Git peut contenir `telegram_bot_token` en clair et le Chat ID. Les macros locales sont également ignorées. Les contenus installés ne sont pas disponibles dans un checkout propre; ils ne doivent pas être imprimés, ajoutés aux fixtures ou inclus dans la PR.
- Le premier `/start` ou message Telegram peut actuellement associer un chat, tandis que les checks des callbacks et messages n'utilisent pas la même identité. Certaines commandes distantes lancent/arrêtent des macros, prennent une capture, lancent CoC ou éteignent le PC.
- `VisionService` ne fait pas de détection; le menu contextuel est vide; le bouton « Purger backlog » n'a pas d'effet; `gif` n'est pas traité dans l'UI. Ces éléments ne sont pas des fonctionnalités validées à conserver tacitement.
- Le lien Mobbin partagé n'était pas accessible au lecteur pendant l'audit. La direction visuelle décrite est tirée du brief, non d'une inspection des écrans Mobbin.

## Architecture courante et frontières de cette proposition

L’architecture vérifiée est React/Vite dans la fenêtre de bureau Tauri 2, avec commandes natives IPC et runtime/backend Rust. Les hooks Windows, raccourcis globaux, fichiers, Telegram et actions CoC sont gérés côté Rust. Les paragraphes ci-dessous qui proposent un service Axum, une UI dans un navigateur ou des routes HTTP décrivent une alternative future, non l’architecture confirmée.

Les raccourcis existants doivent rester les valeurs initiales et l'utilisateur doit pouvoir les personnaliser dans les paramètres. Le parcours doit inclure une vraie aide et faciliter les premières actions sans Telegram. Ces décisions ne sont pas des questions en attente. La réécriture complète, l'usage de sous-agents et la création d'une PR ont été demandés; le travail de code est autorisé.

**Proposition alternative navigateur + Axum :** si cette architecture devait être réévaluée, le site ne devrait pas être exposé sur le réseau et l’analyse loopback ci-dessous devrait alors être reprise. Ces contrôles HTTP ne sont pas une description de l’IPC Tauri actuel.

Les décisions de distribution (portable ou installateur, signature) et les scénarios Telegram de groupe restent à mesurer/définir pendant le travail. Elles ne bloquent pas le squelette ni les essais navigateur. Les comportements de hooks face aux privilèges, DPI et multi-écrans sont des faits à vérifier sur Windows, pas des hypothèses certifiées par le plan.

## Risques de la proposition navigateur + Axum (non applicables au runtime actuel)

Ne traiter cette grille comme une porte de livraison que si la proposition navigateur + Axum est explicitement réouverte.

| Risque | Pourquoi il est réel ici | Contrôle attendu avant parité |
|---|---|---|
| Requête hostile vers le serveur local | Le backend a des commandes puissantes alors que le navigateur peut adresser des ports loopback. CORS n'est pas un contrôle d'autorisation complet. | Tests depuis une origine externe, soumission formulaire, Origin absent/falsifié, Host hostile, port inattendu et essai DNS rebinding; toutes les mutations refusées sans session/origine valides. Le serveur écoute uniquement sur l'interface loopback. |
| Migration ou perte de données | CSV/JSON ignorés par Git, JSON sans version, chemins/noms Unicode, grosse séquence d'événements, lecteur Python qui peut retourner une macro vide pour un fichier invalide. | Copie horodatée, import non destructif et idempotent, collisions explicites, comparaison nombre/ordre/durée/hash des steps, source originale conservée, erreur par fichier, restauration testée. Aucun test sur le profil vivant sans copie. |
| Révélation de secret | Ancien token Telegram en clair dans CSV; les erreurs ou journaux peuvent répéter l'URL/API. | Migration sans afficher la valeur, DPAPI/Credential Manager validé sur compte standard, réponse UI limitée à « configuré », redaction des logs/export; confirmer qu'aucun secret n'entre dans le diff. |
| Capture/replay Windows différente | `SendInput` est soumis à UIPI, peut ne pas injecter le nombre demandé et ne remet pas à zéro les touches déjà tenues. Les hooks exigent un thread/pompe et un retrait propre. | POC manuel Windows, cible inoffensive, AZERTY/QWERTY, app standard/élevée, plusieurs écrans et DPI; arrêt UI/hotkey/erreur/fermeture doit relâcher toute entrée et retirer les hooks. Aucune élévation permanente présentée comme solution implicite. |
| Trois mécanismes de commande divergents | L'UI, les hotkeys et Telegram existent comme déclencheurs parallèles. | Une seule machine d'état Rust; chaque source emprunte la même validation. Test croisé lecture/enregistrement/arrêt et double commande concurrente. |
| Fonction présentée mais vide | Vision, `gif`, menu contextuel et purge ont des chemins factices ou absents dans la baseline. | Retirer, différer ou signaler « indisponible »; ne pas les décrire comme déjà supportés dans l'onboarding/aide. |
| Fermeture ou seconde instance ambiguë | Un site dans un navigateur n'a pas le cycle de vie d'une fenêtre native. | Documenter/tester qui possède l'arrêt du processus, comment la fermeture d'onglet est détectée ou gérée, comment Stop reste atteignable, et l'effet d'un second lancement. |

## Grille de découverte par un nouvel utilisateur

Cette proposition de grille UX est à jouer sur un build desktop Tauri utilisable, pas dans un navigateur externe. Le testeur démarre comme une personne qui n'a pas lu le README, ne connaît pas le dépôt et ne reçoit ni visite guidée ni noms de boutons à l'avance. Il suit les tâches, observe sans corriger, puis consigne les hésitations, les clics manqués, les retours en arrière, les demandes d'aide, les termes incompris et les blocages. Ne jamais utiliser une vraie macro de jeu, un vrai token ou une vraie action d'extinction pendant cette session.

| Étape | Tâche donnée au testeur, sans lui indiquer le contrôle | Ce qu'on doit observer/valider |
|---|---|---|
| 1. Installer et lancer | Depuis l'artefact distribué, installer/lancer l'application avec un compte Windows standard, réduire sa fenêtre puis la retrouver; tester une seconde ouverture. | Fenêtre Tauri compréhensible; fermeture et relance suivent le cycle de vie de l’application; une seconde ouverture focalise l’instance existante; aucune demande admin/compte/token injustifiée. Si l’installateur n’est pas choisi, utiliser un build local et marquer l’installation comme non vérifiée. |
| 2. Comprendre l'écran initial | « Tu veux automatiser une petite séquence inoffensive. Commence par trouver ce qu'il faut faire. » | L'Atelier, l'état initial et la prochaine action sont clairs; recherche et création ne sont pas confondues; l'aide et les réglages sont repérables sans tooltip obligatoire. |
| 3. Créer une première macro | Créer une macro avec un nom simple, enregistrer une courte séquence de touches dans une application de test, attendre la fin du délai initial, puis arrêter et retrouver la macro. | Compréhension du compte à rebours de 3 s, de la capture globale et de la coupe des 3 s finales; Stop accessible; confirmation de sauvegarde et durée réelle; aucun événement perdu sans message. Ne pas lancer la macro dans le jeu. |
| 4. Lire, boucler et arrêter | Relire la séquence dans l'application de test, activer la boucle, puis arrêter la lecture en cours par le moyen le plus évident. | L'état Playing et la boucle sont distincts et visibles; arrêt possible par bouton et hotkey; aucun bouton ne laisse croire à un succès avant retour backend; touches libérées à l'arrêt. |
| 5. Trouver l'aide | « Tu ne te rappelles plus le raccourci par défaut pour arrêter. Trouve la réponse; ensuite explique comment interrompre une macro. » | L'aide retourne une réponse exacte et cohérente avec Paramètres; différence entre raccourci de navigation dans l’interface et raccourci global de macro clarifiée; retour facile à l'Atelier. |
| 6. Personnaliser une touche | Dans les paramètres, remplacer le raccourci de lecture par une combinaison facile à retenir; annuler une capture puis enregistrer, redémarrer, vérifier la persistance et tester la nouvelle combinaison. | Capture de combinaison claire, Escape annule, conflit interne détecté, refus OS expliqué sans perdre l'ancien raccourci, nouveau binding réellement opérationnel; si le hotkey global d'arrêt échoue, une procédure de secours est clairement affichée et le bouton Stop reste visible. |
| 7. Naviguer au clavier seul | Depuis une page rechargée, accomplir navigation Atelier → Paramètres → Aide, puis lancer/annuler une boîte de dialogue avec Tab/Shift+Tab/Entrée/Escape. | Ordre logique, focus visible, aucun piège clavier, noms accessibles, focus renvoyé au déclencheur et annonce d'état mesurée. Aucun état communiqué uniquement par couleur ou animation. |
| 8. Comprendre les erreurs et s'arrêter | Avec Telegram non configuré, trouver comment comprendre son état sans saisir de secret. Puis entrer un chemin CoC volontairement invalide et trouver comment corriger/effacer le réglage. Pendant un replay inoffensif, provoquer Stop et vérifier l'état final. | Aide liée au contexte, erreur avec action de correction, token jamais requis pour le cœur de l'app, arrêt bien distingué de quitter le serveur, aucun état fantaisiste « connecté/lancé ». |
| 9. Vérifier les données existantes | Sur un profil de test cloné, lancer l'import de macros/paramètres historiques puis annuler/relancer l'import si le produit le permet. | L'import dit ce qui va être copié, où sauvegarder/restaurer, conserve l'original, explique collisions et invalides; second essai ne duplique pas. Exécuter sur copie synthétique ou anonymisée uniquement. |
| 10. Tester les mots ambigus | Demander au testeur de reformuler ce que « Lire », « Enregistrer », « Stop », « Boucle », « Lancer CoC » et « couper les 3 dernières secondes » vont faire avant de cliquer. | Consigner les interprétations littérales et mauvaises surprises. Clarifier les libellés qui font confondre lire une macro et lire de la documentation, arrêter un replay et quitter le programme, activer la boucle actuelle et la prochaine, ou enregistrer et sauvegarder. |

Pour chaque étape, noter : réussi sans aide / réussi après aide / abandonné, temps approximatif, où le testeur regarde, vocabulaire mal compris, erreurs de focus, et phrase exacte qui bloque. Après la session, classer les problèmes par gravité utilisateur et corriger d'abord les blocages du premier macro, du Stop et de la récupération. Rejouer les tâches touchées après correction. Prendre des captures d'écran seulement avec données synthétiques; pas de capture des secrets/profils réels.

Les essais doivent inclure la fenêtre Tauri aux dimensions 1024×640, 1280×800 et 1600×900, le DPI représentatif, le mouvement réduit et le clavier seul. Compléter séparément les tests E2E du frontend et les essais natifs Windows/Telegram. Une UI E2E qui passe ne prouve ni les hooks, ni l'injection, ni l'appairage Telegram réel.

## Ordre de revue et critères de livraison

1. Pour le runtime actuel, figer les contrats de commandes IPC Tauri et de stockage Rust. N’exécuter les contrôles HTTP/localhost que si la proposition Axum est réouverte.
2. Comparer les copies de données anciennes et la migration; garder l'original intact et prouver le retour arrière.
3. Vérifier la machine d'état commune et les essais natifs de hook/injection sur Windows avant toute annonce de parité.
4. Lancer la grille de découverte dans l’application desktop sur un build utilisable; laisser les problèmes d'usage guider les corrections UX, sans déléguer l'acceptation au simple test automatique.
5. Relire le diff final : pas de `build/`, `dist/`, profils, tokens, caches, ou binaires locaux; commandes, tests exécutés, et limites manuelles sont indiqués dans la PR.

La distribution, les hooks sur une application élevée, la capture multi-écran, le token réel Telegram et l'usage sous une vraie session de jeu ne sont pas prouvés par une compilation ou un test UI. Ils doivent apparaître comme « non testés » tant qu'une validation représentative n'a pas été effectuée.
