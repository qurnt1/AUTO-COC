# Plan d'implémentation : backend Rust et site React local pour AUTO-COC

## Architecture et comportement actuels vérifiés

Au 14 octobre 2026, le produit est une application desktop Windows : frontend React/Vite dans Tauri 2, backend et commandes système en Rust, communication frontend→backend par IPC Tauri. Il n’y a pas de serveur Axum, de contrat HTTP local, ni de site ouvert dans un navigateur externe.

Le callback Telegram `RELOAD_COC` et l’alias texte `relancer` appellent le rechargement natif Rust. Le backend cherche les fenêtres CoC reconnues, leur envoie `WM_CLOSE`, attend leur disparition puis lance le chemin configuré; s’il ne trouve aucune fenêtre CoC, il lance directement ce chemin. Il ne termine pas l’hôte partagé Google Play Games. La commande Tauri `launch_coc` lance le chemin configuré. « Valider arrivée » et « Recharger COC » ne sont pas des macros système actives. Les anciens placeholders de ces deux noms ne peuvent être supprimés au démarrage que si le manifeste et l’empreinte prouvent qu’ils sont encore vides et intacts; les fichiers utilisateur modifiés ou importés sont préservés.

**Statut du reste de ce document :** les détails de portage et d’architecture ci-dessous sont une proposition technique historique, notamment l’option navigateur + Axum. Ils ne décrivent pas l’implémentation actuelle. Les garder comme idées à réévaluer, pas comme tâches restantes confirmées.

## Baseline historique : dépôt Python observé

Audit statique du dépôt initial `main`, commit `9db7951d418012c951cf301bf8e6c7b225d076d5`. Au moment de cet audit, aucun fichier de code n'avait été modifié ni aucune suite exécutée; le dépôt ne contenait ni `Cargo.toml` ni `package.json`. `tests/` ne contenait pas de tests source. La seule vérification intégrée était `run_selftest()` dans `main.pyw:42-166`; elle comprenait notamment une lecture de macro et une attente de cinq secondes.

La version observée à cette baseline historique était une application Windows en CustomTkinter : `main.pyw:169-265` démarrait Telegram dans un thread asyncio puis la fenêtre Tkinter; `gui/app.py:62-120` définissait l'application, son état `IDLE/RECORDING/PLAYING` et sa fenêtre de 1200×720. La fonction `gui/app.py:130-270` construisait une liste latérale, la zone principale, les statistiques et les commandes. Le README décrivait un démarrage `python main.py`, alors que le point d'entrée présent était `main.pyw`.

À la baseline Python historique, les macros utilisaient des fichiers JSON et les paramètres un CSV. L’ancienne version contenait des macros système protégées, notamment « Recharger COC » et « Valider arrivée ». Cette protection ne fait plus partie du produit actuel : les macros actuelles sont des données utilisateur ordinaires et le rechargement CoC est une commande native distincte.

À cette baseline Python, les données runtime n'étaient pas versionnées : `.gitignore:1-10` ignorait `config/data.csv`, `config/macros/`, `config/macro.json` et les logs. Le CSV de cette ancienne version contenait `auto_loop`, `coc_path`, `coc_safeguard`, `last_macro`, `last_macro_played`, `system_reload_coc_macro`, `telegram_bot_token` et `telegram_chat_id`; son token était stocké en clair. Ces champs décrivent les anciennes données importables, pas les réglages runtime actuels. Les macros installées localement n'étaient pas suivies par Git.

`services/recorder_service.py:106-239` enregistre globalement clavier et souris, attend trois secondes avant le premier événement, puis stocke des temps relatifs. `services/recorder_service.py:246-388` rejoue sur des échéances absolues calculées avec `perf_counter` et permet d'interrompre l'attente. `services/recorder_service.py:275-313` tente de relâcher les touches et boutons au stop. `gui/app.py:544-558` enregistre les raccourcis F1, Ctrl+Shift+1 et Ctrl+Shift+0. Ces raccourcis sont distincts de la capture clavier/souris complète pendant l'enregistrement.

À la baseline, Telegram acceptait des commandes texte et des boutons (`services/telegram_service.py:44-63`, `:253-348`, `:414-591`) et transmettait les actions à l'interface par une queue. Le premier `/start` pouvait auto-associer un chat dans `services/telegram_service.py:253-267`; les callbacks et les messages texte vérifiaient des identités différentes (`:274-332`). Ces constats historiques ne décrivent pas le bot actuel. Le bot de cette baseline exposait aussi une extinction confirmée, la capture d'écran, le lancement du jeu et la sélection de macros.

Autres limites observées à garder en tête : `services/vision_service.py:32-154` est un squelette et n'effectue pas de détection; `utils/system.py:37-172` contient des aides de résolution/arrêt/vérification de processus importées par l'interface, mais aucune invocation n'existe ailleurs dans le dépôt; `services/telegram_service.py:51` mappe `gif` vers `CAPTURE_GIF`, sans traitement correspondant dans `gui/app.py:300-342`. Ne pas présenter ces trois fonctions comme opérationnelles pendant la parité.

## Proposition d’architecture alternative — navigateur + Axum (non retenue)

Cette proposition ancienne décrit un client web servi par Axum. Elle n’a pas été retenue; l’application actuelle utilise Tauri 2 et IPC, comme indiqué au début du document. Les analyses loopback, origines web, sessions HTTP et CSP ci-dessous ne sont à reprendre que si cette architecture alternative est explicitement réévaluée.

```text
Navigateur Windows
  └─ React / TypeScript : vues, navigation clavier, aide, paramètres
       ├─ HTTP/JSON : commandes explicites et réponses typées
       └─ flux état/progression : détails fixés dans le contrat API
                    │ même origine loopback, session locale
Rust / Axum / Tokio
  ├─ contrôleur unique : Idle | Recording | Playing
  ├─ macros, settings, import et stockage utilisateur
  ├─ hooks Windows, replay et raccourcis globaux
  ├─ Telegram async + canal vers le contrôleur
  └─ captures, lancement du jeu, aide/diagnostics et fermeture
                    │
Windows : user32, DPAPI utilisateur, système de fichiers utilisateur
```

Un serveur loopback est une surface attaquable par le navigateur. Lier à `127.0.0.1` ne suffit pas à empêcher une page tierce d'envoyer des requêtes, et CORS seul ne protège pas les actions avec effet de bord. Le contrat API doit imposer une vérification stricte de `Host` et de l'origine exacte, une session imprévisible limitée au processus, une protection CSRF pour chaque mutation et aucun CORS générique. Refuser toute origine tierce ou absente pour les mutations; couvrir les tentatives DNS rebinding, les formulaires et requêtes simples inter-origines. Ne pas transmettre un secret dans une query string, un referer ou des logs. Appliquer une CSP stricte, limiter les corps et méthodes, interdire le chargement distant et ne fournir ni commande shell générique ni accès filesystem par chemin arbitraire. Le modèle précis de bootstrap/session et les routes sont définis dans le contrat API avant leur branchement.

En développement, Vite peut servir le frontend; son proxy doit conserver des appels API relatifs et éviter une politique CORS permissive. Toute exception d'origine réservée au mode développeur doit rester explicite et absente du binaire de production. En production, les assets React sont intégrés au binaire et servis depuis la même origine que l'API.

Le service Rust est l'unique autorité pour l'état et les règles métier. React envoie des DTO validés et reçoit des erreurs compréhensibles, sans token Telegram ni chemins confidentiels inutiles. Les commandes de lecture, écriture, enregistrement, lecture, arrêt et intégrations restent des routes explicites. Le serveur doit avoir un cycle de vie déterministe : démarrage visible, ouverture navigateur, fermeture sûre qui arrête les hooks et relâche les entrées, et comportement documenté si l'onglet est fermé ou si une seconde instance démarre.

## Découpage des modules et API de frontière

Découpage indicatif, à garder modeste tant que chaque module reste avec un seul appelant :

- `backend/src/app_state.rs` : état partagé, transitions validées, fermeture et arrêt coordonnés.
- `backend/src/api/` : routes Axum, DTO sérialisables, vérifications d'origine/session/CSRF et traduction d'erreurs. Exemples : `get_snapshot`, `list_macros`, `create_macro`, `rename_macro`, `delete_macro`, `start_recording`, `stop_recording`, `play_macro`, `stop_playback`, `save_settings`, `set_shortcuts`, `request_shutdown`.
- `backend/src/macros/` : proposition de module pour modèle d'étape, validations, hash, lecture/écriture JSON, tri et CRUD des macros utilisateur.
- `backend/src/settings/` : modèle typé et versionné, import CSV et persistance atomique.
- `backend/src/input/` : traits internes pour capture, replay, raccourcis d'action et événements typés; implémentation Windows isolée dans `windows.rs`.
- `backend/src/telegram/` : client, autorisation, mapping de commandes, menus, photos et arrêt du bot.
- `backend/src/system/` : capture d'écran, ouverture `.exe`/`.lnk`, diagnostics et demande d'arrêt système.
- `backend/src/logging.rs` : rotation et filtrage des secrets.
- `frontend/src/` : écrans React Accueil/Atelier, Macros, Enregistrement/Lecture, Paramètres, Raccourcis, Telegram, Aide et Diagnostic; un client HTTP typé appelle les routes de même origine et s'abonne aux événements.

Garder un contrat API étroit. Les actions de replay, arrêt PC, capture et sauvegarde sont des routes distinctes avec paramètres dédiés; ne pas exposer une commande générique « exécuter » ou l'accès direct aux fichiers. Le serveur ne sert que l'interface locale empaquetée et l'API attendue; refuser les méthodes/chemins non prévus et toute navigation vers une origine non autorisée. Définir une CSP et des en-têtes de sécurité adaptés au navigateur.

## Crates et API candidates à vérifier

Vérifier `rustc`, `cargo`, Node et npm présents sur la machine de build; rien dans le dépôt initial ne prouvait leur installation. Épingler les versions retenues dans les lockfiles. La version du navigateur Windows devient un élément de validation utilisateur, pas une dépendance intégrée au binaire.

| Domaine | Candidat | API ou point à valider |
|---|---|---|
| Serveur/API | `axum`, `tokio`, `tower-http` | Serveur HTTP asynchrone local, routage JSON, intégration des assets, middleware d'origine/session, taille maximale des corps, arrêt gracieux et version/MSRV. [Axum](https://docs.rs/axum/latest/axum/) [Tokio TcpListener](https://docs.rs/tokio/latest/tokio/net/struct.TcpListener.html). |
| Types/config | `serde`, `serde_json` | dérivations `Serialize`/`Deserialize`, gestion de champs inconnus, compatibilité UTF-8 et formats de nombre. |
| CSV | crate `csv` | `ReaderBuilder`/`WriterBuilder` avec `;`, en-têtes présents ou absents, BOM, guillemets et ordre. |
| Hash | `sha1` seulement si parité SHA-1 maintenue | produire exactement le SHA-1 attendu par les fichiers historiques ou ne mettre à jour le hash qu'à la première sauvegarde valide; voir compatibilité plus bas. SHA-1 reste une empreinte de compatibilité, pas un mécanisme de sécurité. |
| Async Telegram | `tokio`, `teloxide` | Dispatcher, long polling, handlers message/callback, upload de photo, arrêt contrôlé, erreurs réseau, limite callback Telegram de 64 octets et autorisation par utilisateur/chat. [Dispatcher](https://docs.rs/teloxide/latest/teloxide/dispatching/struct.Dispatcher.html) [polling](https://docs.rs/teloxide/latest/teloxide/update_listeners/struct.Polling.html). |
| Input Windows | crate officiel `windows` | signatures/features exactes pour `SendInput`, `WH_KEYBOARD_LL`, `WH_MOUSE_LL`, `SetWindowsHookExW`, `CallNextHookEx`, `GetMessageW`, `UnhookWindowsHookEx`, `RegisterHotKey`, `UnregisterHotKey`, `SetCursorPos` et structures `INPUT`. Mesurer capture globale, répétitions, DPI et multi-écrans sur Windows. [SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput) [SetWindowsHookExW](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowshookexw) [RegisterHotKey](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-registerhotkey). |
| Simulation alternative | `enigo` seulement comme comparaison de POC | La page `enigo` décrit un support Windows et mouse/keyboard, mais indique aussi que l'API actuelle est encore en alpha; mesurer latence, exactitude des touches, coordonnées et release avant de retenir cette abstraction. [docs.rs/enigo](https://docs.rs/enigo/latest/enigo/). |
| Listener alternatif | `rdev` seulement comme comparaison de POC | `listen` observe les entrées globales; `grab` est une feature `unstable_grab`. Ses propres docs le qualifient de projet personnel, donc ne pas en faire la base sans vérification de maintenance et de fiabilité. [docs.rs/rdev](https://docs.rs/rdev/latest/rdev/). |
| Secret Telegram | API Windows DPAPI via `windows` (`CryptProtectData`, `CryptUnprotectData`) | scope utilisateur, octets protégés, erreur et migration; ne jamais utiliser le scope machine par défaut. [CryptProtectData](https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptprotectdata). |
| Logs | `tracing`/`tracing-subscriber` ou équivalent Rust simple | rotation, niveaux, redaction du token et absence de secrets dans erreurs HTTP; éviter une couche si un logger déjà retenu suffit. |
| Captures | API Windows GDI/WIC via `windows`, ou crate capture maintenu à sélectionner après POC | capture du bureau virtuel, DPI, moniteurs multiples, encodage PNG et conversion à envoyer à Telegram. Ne pas choisir sur la seule parité de nom avec Pillow/MSS. |
| Distribution | proposition d’alternative navigateur + Axum | Le navigateur par défaut et le service loopback appartiennent à cette ancienne proposition; le runtime vérifié est Tauri 2. Toute évolution doit comparer packaging, mise à jour et emplacement des données au shell courant. |

Microsoft précise que `SendInput` peut retourner moins d'événements que demandé, qu'il est soumis à UIPI et qu'il ne réinitialise pas les touches déjà tenues. La couche Rust doit traiter un envoi partiel comme une erreur de replay, tenter les relâchements de sécurité et signaler une cible d'intégrité supérieure comme limite. `SetWindowsHookExW` implique un thread avec pompe de messages et impose de retirer les hooks. Ces propriétés rendent nécessaire un POC Windows isolé; une compilation seule ne valide pas les hooks ni le comportement en jeu.

## Modèle de données et migration JSON/CSV

1. **Ne pas transformer automatiquement les macros au premier démarrage.** Faire une sauvegarde horodatée de `config/data.csv`, `config/macros/*.json` et `config/macro.json` vers un répertoire de sauvegarde utilisateur. Garder la source intacte jusqu'à import validé. Le dépôt ignore déjà ces éléments, ne jamais les inclure dans les commits.
2. **Importer les réglages dans un nouveau modèle JSON versionné** sous `%LOCALAPPDATA%\AUTO-COC\settings.json`, car les macros sont volumineuses et l'application observée est locale à Windows. Si le chemin cible existe, ne pas la remplacer. Mapper `auto_loop`, `coc_path`, `telegram_chat_id`, `last_macro` et les champs inconnus; conserver les valeurs inconnues en extension ou dans le snapshot migré pour éviter une perte silencieuse. Convertir `auto_loop` avec les mêmes booléens acceptés par `utils/config.py:311-316`. Supporter CSV à entête et ancien CSV sans entête, délimiteur point-virgule, UTF-8/BOM et valeurs entre guillemets.
3. **Sortir le secret du CSV.** Chiffrer `telegram_bot_token` avec DPAPI scope `CurrentUser`; conserver seulement `telegram_token_configured` dans le JSON settings. Le frontend reçoit un booléen de présence, jamais le token; une commande dédiée peut le remplacer ou l'effacer. Le flux de migration ne doit ni journaliser ni afficher le token. Si DPAPI échoue, garder le CSV source et afficher une erreur claire, ne pas démarrer Telegram avec une valeur tronquée ou vide.
4. **Préserver les macros utilisateur historiques** en conservant leurs champs `name`, `updated_at`, `sha1`, `steps`, leurs noms Unicode et leur ordre naturel. Ne pas recréer de macros système protégées : `Recharger COC` est une action native et `Valider arrivée` est retirée. Les migrations ne suppriment que les placeholders historiques dont l’empreinte et le manifeste prouvent qu’ils n’ont pas été modifiés; préserver toute macro utilisateur de même nom.
5. **Valider sans réparer silencieusement.** L'ancien `read_macro_file()` retourne une macro vide en cas d'exception (`utils/config.py:149-157`), et `StepType.from_string()` rabat les valeurs inconnues vers `NOP` (`models/macro.py:28-39`). La migration Rust doit afficher les fichiers invalides/événements inconnus, les exclure du replay et préserver les octets originaux. Ne jamais sauvegarder une macro vide par-dessus un JSON illisible.
6. **Hash compatible.** `models/macro.py:106-113` et `utils/config.py:130-136` calculent SHA-1 à partir de JSON UTF-8 trié par clé; `utils/config.py:183-203` s'en sert pour éviter une écriture quand le contenu ne change pas. Les espaces et conventions d'échappement du sérialiseur Python peuvent différer de `serde_json`. Vérifier une implémentation exacte sur des snapshots de macros; sinon, garder le SHA-1 existant à l'import et le recalculer seulement lors de la première sauvegarde, après copie de sauvegarde, en documentant ce changement de métadonnée.
7. **Écrire atomiquement.** Sérialiser dans un fichier temporaire du même dossier, flush/sync, remplacer en gardant une sauvegarde en cas d'échec. Appliquer aux paramètres comme aux macros. Ajouter noms uniques insensibles à la casse sous Windows et bloquer séparateurs/chemins absolus.

## Capture, replay et arrêt sûr

### Séparer capture globale et raccourcis d'application

Les raccourcis globaux F1, Ctrl+Shift+1, Ctrl+Shift+0 et leurs versions personnalisées doivent être enregistrés par le backend Windows, indépendamment des touches du navigateur. Ils ne remplacent pas l'enregistrement de tous les mouvements/clics/touches du bureau. Pour ce dernier, faire un POC Windows avec `WH_KEYBOARD_LL` et `WH_MOUSE_LL` sur un thread Rust dédié et une pompe de messages, ou tester `rdev`/`inputs` en option. Les callbacks de hook ne font que prendre un timestamp monotone, convertir l'événement et l'envoyer à un buffer; aucun accès disque, encodage ou appel UI dans le callback.

La file des événements doit avoir un plafond testé. Si elle déborde, marquer la session comme incomplète, stopper proprement la capture et prévenir l'utilisateur au lieu d'enregistrer silencieusement une macro amputée. Les hooks sont installés seulement après clic sur « Enregistrer », avec une bannière persistante « Capture clavier et souris active », le compte à rebours/grâce actuelle de 3 s, un bouton Stop immédiatement disponible et une fermeture qui les retire. La sélection de raccourci suspend temporairement la capture pour ne pas enregistrer la combinaison de configuration.

### Contrat de temps et modèle d'événement

- Conserver `t` en secondes écoulées depuis l'événement précédent et `data` compatibles avec le JSON existant. Utiliser `std::time::Instant`, timestamps absolus pour le replay, et une attente interruptible; ne pas accumuler des sleeps relatifs.
- Garder la position souris en coordonnées bureau et la version historique des clics, touches et scrolls. Cartographier explicitement les touches virtuelles Windows vers les noms existants (`enter`, `ctrl`, `f1`, lettres, touches spéciales, anciens encodages `<vk>`). Ne pas convertir une touche inconnue en clic ou `nop` sans alerte.
- Les macros réelles contiennent plus de dix mille événements et beaucoup de mouvements. Première parité : ne pas échantillonner ou fusionner ces événements pendant la migration. Si le volume devient un problème, proposer ultérieurement une option de compression qui conserve la durée et les transitions press/release, avec comparaison d'un replay avant/après.
- Distinguer les événements injectés par l'application à l'aide des flags d'injection Windows et/ou du champ `dwExtraInfo`; ne pas réenregistrer les mouvements du Player.

### Gestion du Player et failsafes

- Une seule transition atomique autorise une opération active à la fois. Refuser `play`/`record` si l'état ne le permet pas; les commandes UI, hotkeys et Telegram passent par le même contrôleur Rust.
- Le Player possède le handle clavier/souris sur son thread d'exécution. `stop` envoie l'annulation, attend l'accusé de fin et le thread relâche toutes les touches/boutons suivis dans son `finally`/guard de sortie. Ne pas relâcher les entrées depuis un autre thread pendant que le Player est toujours vivant.
- Un `SendInput` partiel, une erreur de hook, un échec de write, la fermeture de fenêtre ou un panic conduisent à un état `Idle` visible, aux tentatives de release, au retrait des hooks et à un log sans données sensibles. Pas de détachement daemon avec état oublié.
- Ajouter un arrêt d'urgence indépendant, conserver Ctrl+Shift+0 par défaut, et toujours afficher le bouton Stop. Définir le comportement sur perte de focus, veille/réveil, changement d'écran/DPI, saisie physique en cours et application cible élevée par rapport au processus.
- Le trim des 3 dernières secondes est actuel (`gui/app.py:470-484`, `trim_tail()` dans `services/recorder_service.py:458-494`). Le reproduire seulement si les validations confirment le même sens temporel. Un arrêt PC depuis le bureau ou Telegram reste en deux étapes de confirmation; ne pas rendre une action destructive disponible par simple callback de menu.

## Raccourcis personnalisables

Maintenir les trois raccourcis présents comme valeurs de départ : F1 bascule play/stop, Ctrl+Shift+1 démarre, Ctrl+Shift+0 stoppe. Ajouter une configuration typée par action dans `settings.json`, puis une section de paramètres React avec bouton « Enregistrer une combinaison », prévisualisation et action associée. Le composant capture une touche principale plus ses modificateurs quand il a le focus; Escape annule la capture.

La couche native vérifie format, doublons entre actions, touches réservées/intrinsèquement risquées et possibilité réelle d'enregistrement. Si le nouveau raccourci est pris par Windows/une autre application ou si le hotkey registration échoue, conserver l'ancien binding, expliquer l'échec et permettre le reset des valeurs par défaut. Remplacer les enregistrements de façon transactionnelle : désenregistrer/re-enregistrer sans laisser l'application sans stop global. Revalider les raccourcis au démarrage et les rétablir quand l'app se ferme. Le plugin documente un état `Pressed`/`Released`; le valider pour les trois defaults et les combinaisons clavier propres à Windows avant de figer le modèle.

## Telegram, captures, lancement et stockage des paramètres

- Retenir `teloxide` + Tokio comme candidat pour long polling et callback query; reproduire les actions courantes et alias exposés par `TEXT_COMMAND_MAP` et `KNOWN_CALLBACKS`. Canal Telegram vers le contrôleur Rust, même logique de permission que le frontend.
- Remplacer l'association implicite par un appairage voulu depuis l'application : le propriétaire confirme localement, le backend lie un user ID et un chat privé autorisés et ignore les autres. Vérifier `from_user.id` et `chat.id` pour tous les messages, callbacks et confirmations. Le vieux CSV avec chat ID préremplit une association à confirmer, pas une nouvelle autorisation implicite. Documenter les commandes dans la page Aide.
- Filtrer les tokens dans logs, erreurs de client HTTP et diagnostics. Ne jamais afficher le token, son URL Bot API, ou le contenu complet de `settings.json`. La page paramètres indique si le bot est configuré et teste une connexion par un statut non sensible.
- La capture actuelle essaie Pillow puis MSS dans `utils/system.py:206-244`, produit PNG et l'envoie via `services/telegram_service.py:371-391`. Reproduire la capture du bureau virtuel et le statut d'erreur; fournir un aperçu local et une action d'envoi explicite. Tester plusieurs écrans, DPI, verrouillage/écran indisponible et taille maximale Telegram.
- Pour `.exe` et `.lnk`, utiliser `ShellExecuteW` ou une abstraction Windows documentée, pas une chaîne shell reconstruite. Valider le chemin fourni, retourner un état réellement observé et ne pas marquer le jeu « lancé » sur la seule absence d'erreur d'ouverture. Aides actuelles `resolve_exe_from_path`, `kill_process_by_name`, `is_process_running` n'ont pas de callers; ne pas les réintroduire comme actions visibles sans spécification UX. Les entrées `CAPTURE_GIF` et Vision restent hors parité tant que leurs comportements n'existent pas.
- Placer settings, macros et logs sous l'espace utilisateur `%LOCALAPPDATA%\AUTO-COC`, pas dans le dossier d'installation. Fournir dans Aide/Paramètres « Sauvegarder mes données » et « Ouvrir le dossier des macros ». Le CSV et les macros historiques restent importables. La fonction diagnostics n'expose que versions, état des services, chemins masqués si requis, capacité d'écriture et dernier message d'erreur expurgé.

## Séquence incrémentale proposée

1. **Proposition alternative : caractériser les contrats navigateur + Axum.** Établir schémas JSON/CSV, limites de noms, alias de touches et callbacks sur copies synthétiques/sauvegardées. Cette option n’est pas l’architecture actuelle; ne pas modifier le runtime IPC Tauri ni les données réelles au titre de cette proposition.
2. **Proposition alternative uniquement :** créer un client navigateur React + Axum avec contrat HTTP/JSON. Ne pas entreprendre ce portage au titre d’un travail restant; le frontend React et le backend Rust existent déjà dans le shell Tauri avec IPC.
3. **Implémenter les modèles et la migration JSON/CSV avant les entrées natives.** Exposer CRUD macro/settings sur fixtures synthétiques; vérifier compatibilité avec des copies des fichiers historiques, import idempotent, erreurs sans overwrite, hash et atomicité. À cette étape, l'UI affiche les erreurs de migration et fournit sauvegarde/restauration.
4. **Faire le POC Windows de capture globale et replay.** Décider après mesure entre APIs Windows directes et une crate. Démontrer un enregistrement de clic/touche/mouvement, replay chronométré, arrêt rapide, release des entrées, prévention de l'auto-capture, DPI/multi-moniteur et fermeture de hook. Aucun connecteur cloud ne doit être nécessaire pour valider ces opérations.
5. **Intégrer le contrôleur de session et les raccourcis configurables.** Une seule machine d'état sert à tous les déclencheurs. Couvrir conflits et restauration des raccourcis; exposer transitions et progression au frontend par le flux d'événements choisi dans le contrat API.
6. **Ajouter Telegram puis les intégrations système.** Autorisation/appairage, toutes les actions réellement existantes, capture d'écran, lancement jeu et arrêt PC confirmable. Gérer absence de réseau, token incorrect, timeout et arrêt propre au exit.
7. **Construire les écrans React complets et user-oriented.** Relier bibliothèque/recherche/CRUD macros, lecture et enregistrement, commandes d'urgence, personnalisation hotkeys, Telegram, aide, diagnostics, sauvegarde et messages d'erreur. UI moderne inspirée de la hiérarchie visuelle existante et de Perplexity, sans copier ses écrans/assets; animations limitées aux transitions d'interface, jamais à la boucle de replay.
8. **Valider le paquet Windows et effectuer le cutover.** Installer dans un profil utilisateur standard, premier démarrage depuis le CSV historique, relancer après migration, puis fermeture/rollback. Conserver une sauvegarde et un installeur de la dernière version Python jusqu'à validation utilisateur de la nouvelle version. Une fois le binaire Rust adopté, supprimer les sources Python et dépendances uniquement dans une étape finale distincte.

Cette séquence est incrémentale côté travail et évite le risque d'un gros bang non vérifié sur les hooks natifs, mais l'application publiée bascule en une seule version cohérente. Pas de double écriture production ni de backend Rust qui relance Python en sidecar : ces architectures rendraient ambiguës l'autorité des paramètres, l'état Play/Stop et la fermeture.

## Vérifications exigées avant annonce de parité

- **Compatibilité données (proposition de validation)** : importer des copies de macros volumineuses et des anciens placeholders vides avec vérification de provenance, préserver les macros utilisateur de même nom, traiter la macro legacy si présente et les valeurs CSV avec accents/chemins; import idempotent; JSON malformé et champs inconnus signalés sans remplacement; SHA-1 comparé aux fichiers Python historiques.
- **Capture et lecture** : E2E Windows sur cible de test inoffensive; type de touches et mouse buttons; scroll; ordre et durées; répétition; arrêt au milieu d'une touche tenue; entrée utilisateur simultanée; perte de focus; queue saturée; DPI 100/150/200 %, double écran avec origine négative; UIPI / app élevée; veille/reprise; fermeture en pleine capture/replay. Vérifier que toutes les entrées suivies sont libérées et les hooks retirés.
- **Hotkeys** : F1, Ctrl+Shift+1 et Ctrl+Shift+0, puis changement personnalisé; conflit système/logiciel, collisions, Escape et fermeture de la modal, sauvegarde atomique, redémarrage, impossibilité d'enregistrer et récupération du Stop.
- **API/UI et navigateur hostile** : chaque transition Idle/Recording/Playing cohérente depuis boutons, hotkeys et Telegram; état d'abonnement/connexion récupérable après reconnexion; tests navigateur des formulaires inter-origines, origines tierces/absentes, Host falsifié, DNS rebinding, session manquante, jeton périmé et requêtes trop grandes. Aucun chemin d'accès arbitraire; erreurs d'authentification sans détail sensible.
- **Telegram/système** : token absent/invalide, réseau coupé et arrêt du poller; utilisateur/chat non autorisé, groupe rejeté, confirmations shutdown; aucune trace de token dans logs ou retour de commande; capture/PNG; chemin `.lnk`/`.exe`, exécutable manquant; extinction uniquement après la confirmation finale, testée sans arrêter réellement la machine en CI.
- **Distribution** : build release, lancement sans Python sur compte Windows standard, assets servis depuis le binaire, ouverture navigateur, données hors `Program Files`, arrêt du processus et désinstallation qui n'efface pas par défaut macros/settings, sauvegarde/export, signature si publiée et rollback documenté.

Ne pas confondre compilation Rust, tests simulés, build d'installateur et validation réelle des hooks/entrée système. La parité d'enregistrement/replay exige des essais sur Windows et une cible de test contrôlée.

## Risques concrets et réponses

| Risque | Conséquence | Réponse prévue |
|---|---|---|
| Conversion de plus de 30 000 événements souris ou changement de temps/données | Replays inutilisables, latence ou perte de fidélité | JSON initial préservé, test hash, compte d'événements et durée, aucun échantillonnage au cutover. |
| Hook global mal retiré ou callback lent | clavier bloqué/retardé, app qui fuit des hooks | thread propriétaire et pompe de messages, callback minimal, handle RAII, timeout/erreur visible, E2E fermeture et force stop. |
| `SendInput` bloqué par UIPI/élévation ou retour partiel | macro incomplète ou touche coincée | détecter le retour partiel, relâchements suivis, montrer la limite de privilège; ne pas promettre un contournement de sécurité Windows. |
| Migration silencieuse de `data.csv` | perte de paramètres ou exposition du token | source conservée, sauvegarde, import versionné et idempotent, DPAPI utilisateur, aucun secret dans UI/logs/dépôt. |
| Ancien parseur qui masque un JSON invalide en macro vide | overwrite destructif possible | parser strict côté Rust, fichier source préservé, action de restauration et statut par fichier. |
| Autorisation Telegram implicite ou callbacks forgés | un tiers contrôle macro, capture ou extinction | appairage local explicite, validation simultanée user/chat, allowlist de commandes, confirmer les opérations dangereuses. |
| Requête d'un site hostile vers l'API loopback ou DNS rebinding | macro lancée, réglage changé ou action système déclenchée à l'insu de l'utilisateur | bind strict sur 127.0.0.1, contrôle Host + Origin, session aléatoire, anti-CSRF par mutation, pas de CORS générique, CSP et test depuis une origine tierce; aucune route shell/filesystem générique. |
| Coexistence des runtimes avec deux stores | état divergents et doublons de capture | référence Python temporaire pendant développement, cutover paquet unique, aucune double écriture. |
| Navigateur non pris en charge ou fermé pendant une action | page inaccessible ou état actif peu visible | indiquer les navigateurs testés, garder stop global/backend et fermer proprement à la commande explicite; vérifier le cycle multi-instance et après fermeture d'onglet. |

## Décisions confirmées et validations restantes

- Vérifié : React/Vite dans la fenêtre Tauri 2, backend Rust et commandes IPC; aucun serveur Axum ni navigateur externe.
- Proposition alternative : client navigateur avec Axum loopback. Ce n’est pas une décision courante et elle ne doit pas être confondue avec l’architecture Tauri.
- Confirmé dans le périmètre demandé : conserver les raccourcis historiques et rendre leur personnalisation accessible dans les paramètres; ajouter une aide orientée utilisateur. Le comportement réel des aides vision, `gif`, menu contextuel et purge Telegram reste à traiter comme absent ou différé, selon l'audit source.
- La politique d'appairage Telegram, les règles de groupes et le maintien de l'arrêt PC dans le bot doivent être implémentés avec une autorisation explicite et des confirmations. Le contenu Telegram installé localement n'étant pas auditable via Git, vérifier sur copie de profil et ne jamais supposer que le Chat ID historique prouve une autorisation sûre.
- Le type d'installation, la signature et la liste des navigateurs Windows testés restent des validations de distribution, pas des questions qui bloquent le développement.
- Restent à prouver sur la machine Windows cible : précision réelle des hooks/replay, fonctionnement face à une application d'intégrité supérieure, comportements DPI/multi-écran et absence d'effet de bord depuis une page tierce. Une compilation ou un test Rust simulé ne prouve pas ces propriétés.
