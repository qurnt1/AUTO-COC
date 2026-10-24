# Proposition historique de contrat HTTP local — AUTO-COC

> **Architecture actuelle vérifiée au 14 octobre 2026 :** React/Vite est hébergé par Tauri 2 et les commandes passent via IPC vers le runtime/backend Rust. Il n’y a pas de serveur Axum/HTTP local, de routes `/api/*` ni de navigateur externe. Le bouton Telegram `RELOAD_COC` appelle le rechargement natif Rust : fermeture de la fenêtre CoC reconnue, attente de sa disparition, puis lancement du chemin configuré; sans fenêtre CoC reconnue, lancement direct. Le processus partagé Google Play Games n’est pas tué.
>
> **Statut :** le reste de ce fichier conserve une proposition alternative de contrat HTTP, non implémentée et non canonique. Toute mention d’API ci-dessous décrit cette proposition uniquement.

Cette proposition suppose un client React dans un navigateur et un backend local Rust. Elle ne décrit pas le contrat IPC Tauri réellement utilisé.

## Origine et session

- Le binaire Rust choisit un port libre sur `127.0.0.1`, sert l'interface React compilée et ouvre son URL dans le navigateur par défaut.
- `GET /api/session` renvoie `{ "token": "..." }` seulement si `Host` vise l'hôte/port écouté et si `Origin` est exact. Si Origin est absent, accepter seulement `Sec-Fetch-Site: same-origin` avec un `Referer` analysé dont l'origine (schéma, hôte et port) correspond exactement à l'origine attendue. Pas de CORS. Le frontend conserve ce jeton uniquement en mémoire.
- Toutes les routes `/api/*` après `/api/session` exigent `X-Auto-Coc-Session: <token>`, le `Host` exact et une origine navigateur valide. Pour les mutations, exiger `Origin` exact; accepter un Origin absent seulement pour les requêtes GET sans effet de bord, avec `Sec-Fetch-Site: same-origin` et un `Referer` dont l'origine analysée correspond exactement. Les mutations acceptent uniquement `application/json` avec taille limitée. Le serveur n'émet pas de CORS. Les pages n'incluent aucune ressource distante et reçoivent une CSP avec `frame-ancestors 'none'`, plus `X-Frame-Options: DENY` et `Referrer-Policy: same-origin`, afin de bloquer l'encadrement et le partage de référent vers des sites tiers.
- Les mutations sans paramètres envoient un corps JSON `{}`; ne pas utiliser de formulaire HTML. Les tailles de corps sont plafonnées avant désérialisation.
- Le client React utilise toujours des URL relatives (`/api/...`). En développement, Vite proxy `/api` côté serveur vers un backend configuré sur loopback; il n'existe aucune variable `VITE_API_BASE` dans le bundle client. Un backend debug n'autorise que l'origine Vite exacte et uniquement en debug; le proxy réécrit `Host` vers l'adresse loopback backend. La version de production n'autorise jamais l'origine Vite.

## Forme des réponses

Toute erreur est JSON: `{ "error": { "code": "...", "message": "..." } }`, avec un statut HTTP adapté. Les mutations qui modifient l'état de l'application renvoient `{ "snapshot": Snapshot }`. Les routes de fichier documentent leur type de réponse, par exemple `image/png` pour screenshot; les routes de confirmation documentent leur DTO dédié. Une absence ou un conflit d'état est explicite, jamais silencieux.

`GET /api/snapshot` renvoie un `Snapshot` complet:

```json
{
  "revision": 1,
  "status": { "kind": "idle" },
  "selectedMacro": "Ma macro",
  "macros": [{ "name": "Ma macro", "eventCount": 12, "durationSeconds": 4.2, "readable": true, "issue": null }],
  "settings": {
    "loop": false,
    "cocPath": "",
    "shortcuts": { "toggle": "F1", "play": "Ctrl+Shift+1", "stop": "Ctrl+Shift+0" },
    "telegram": { "tokenConfigured": false, "paired": false, "status": "not_configured", "pairingCode": null, "pairingExpiresAt": null }
  },
  "session": { "elapsedSeconds": 0, "cycles": 0 },
  "migration": { "sourceSelected": false, "available": false, "alreadyImported": false },
  "onboardingComplete": false
}
```

`status.kind` est `idle`, `recording`, ou `playing`; seul l'état actif inclut `macroName` et `elapsedSeconds`. L'enregistrement expose aussi `phase: preparing | capturing` et un compte à rebours mesurable pendant les trois secondes de préparation. La révision augmente à chaque changement visible. `GET /api/events?after=<revision>` est un long-poll: il renvoie le snapshot à la première révision supérieure, sinon attend au plus 20 secondes, puis renvoie le snapshot courant. Le client relance le long-poll après réponse ou reconnexion. Cela évite une boucle de requêtes rapide et fonctionne avec l'en-tête de session.

## Routes MVP

- `GET /api/session` — bootstrap du jeton (origine exacte seulement).
- `GET /api/snapshot` — macros, sélection, état de session, réglages non secrets et onboarding.
- `GET /api/events?after=<u64>` — long-poll des changements; réponse toujours un `Snapshot`.
- `GET /api/migration` — état de l'import et, si une source est sélectionnée, aperçu `{ "available": true, "settingsFound": true, "macros": [{ "name": "...", "eventCount": 10, "readable": true, "issue": null }] }` sans valeurs de réglage ni secrets.
- `POST /api/migration/select-source` — ouvre le sélecteur natif de dossier Windows; l'utilisateur choisit l'ancien dossier de projet qui contient `config/data.csv`, `config/macros/*.json` et/ou `config/macro.json`. Le backend garde ce choix en mémoire puis renvoie un aperçu expurgé. Aucune route n'accepte de chemin fourni par le frontend.
- `POST /api/migration/import` — proposition : exige l'état idle au moment de l'import. Crée une sauvegarde expurgée, copie les réglages et macros valides, préserve la source, refuse toute collision et signale les fichiers invalides sans les réécrire. Aucun traitement spécial de collision ne doit préserver ou recréer une macro système. Les données source peuvent encore contenir le token historique en clair; ne pas le journaliser.
- `GET /api/macros/{name}` — `{ "name": "...", "steps": [{ "t": 0.1, "type": "key_down", "data": { "key": "a" } }] }`, après validation de nom.
- `POST /api/macros` `{ "name": "..." }` — créer et sélectionner une macro vide; refuse toute collision après normalisation (y compris casse Windows) sans jamais écraser.
- `PATCH /api/macros/{name}` `{ "newName": "..." }` — renommer une macro utilisateur; refuser les collisions Windows.
- `DELETE /api/macros/{name}` — supprimer une macro utilisateur avec confirmation côté interface.
- `POST /api/selection` `{ "name": "..." }` — sélectionner, interdit pendant une opération active.
- `POST /api/recording/start` et `POST /api/recording/stop` — démarrer/arrêter la capture. Garder les 3 s de préparation, puis capturer; à l'arrêt, couper les 3 dernières secondes de la séquence et sauvegarder la macro sélectionnée. Si la séquence fait au plus 3 s, le résultat est vide. Si la limite de capture est atteinte, sauvegarder les étapes collectées et renvoyer `recording_event_limit` avec un avertissement que la macro est partielle; ne pas abandonner le buffer ni laisser l'opération bloquée. L'interface explique les délais et cette limite avant d'armer les hooks.
- `POST /api/playback/start` — sans champ loop; utilise l'unique valeur persistée `settings.loop` et la macro sélectionnée. `POST /api/playback/stop` interrompt et attend la libération des entrées.
- `PATCH /api/settings` — champs autorisés `loop`, `cocPath`; aucune clé inconnue n'est persistée comme instruction d'exécution.
- `PUT /api/shortcuts` `{ "toggle": "...", "play": "...", "stop": "..." }` — validation de syntaxe, conflits et enregistrement transactionnel; échec garde l'ancien jeu de raccourcis.
- `POST /api/launch-coc` — lance seulement le chemin CoC configuré, jamais un chemin fourni dans cette requête.
- `POST /api/screenshot` — renvoie une capture PNG (`image/png`) pour aperçu local; aucune capture n'est envoyée à Telegram sans action explicite.
- `PUT /api/telegram/config` `{ "token": "..." }` — enregistre ou remplace un token non vide côté Rust dans DPAPI/Windows Credential Manager. `DELETE /api/telegram/config` avec `{}` efface la configuration. Un changement effectif de token invalide l'appairage et le code en attente. Le token n'est jamais retourné ni journalisé.
- `POST /api/telegram/pairing/start` — crée un code temporaire à saisir dans une conversation privée avec le bot. Seul un `/start <code>` correspondant, à usage unique, lie l'ID du chat et l'ID de l'utilisateur; rejeter les groupes. Limiter à cinq essais incorrects par code, invalider le code au seuil et imposer un bref délai avant un nouvel appairage. Invalider le pairing et les codes en attente quand le token change; `POST /api/telegram/pairing/cancel` invalide le code.
- `POST /api/telegram/test` — vérifie le token et renvoie un statut expurgé.
- Les commandes Telegram passent par le contrôleur Rust. `relancer` et le bouton `RELOAD_COC` exécutent une action native de rechargement, pas la lecture d’une macro; vérifier l’identité autorisée à chaque message et callback.
- `POST /api/system/shutdown/prepare` renvoie `{ "confirmationId": "...", "expiresAt": "..." }`; `/api/system/shutdown/confirm` accepte `{ "confirmationId": "..." }`. Le code est à usage unique, lié à la session et expire après 30 secondes. Deux étapes avec confirmation visible; le frontend n'expose pas une commande d'arrêt en un seul clic. Ne jamais tester l'étape finale sur un poste réel.
- `POST /api/macros-folder/open` — ouvre uniquement le dossier macro connu de l'application.
- `POST /api/application/quit` — stoppe proprement le backend local; refuse ou demande un arrêt explicite de l'opération active avant de fermer les hooks et services.
- `GET /api/help` — contenu d'aide local versionné avec raccourcis et explications des délais d'enregistrement.
- `GET /api/diagnostics` — versions, états, erreurs expurgées et chemins masqués; aucun token, contenu brut du CSV ou contenu complet des macros.
- `POST /api/onboarding/complete` — marque le parcours d'aide comme terminé.

Chaque intégration a un DTO dédié, une action explicite et un état vérifiable. Aucun endpoint `exec`, accès fichiers arbitraire, URL arbitraire ou route d'arrêt système à une seule étape ne sera ajouté.

## Frontend

Le frontend définit `ApiClient` typé pour les routes ci-dessus. Il récupère le jeton au démarrage, place le jeton dans l'en-tête de chaque requête, hydrate le premier snapshot, puis maintient un seul long-poll. Il ne met jamais le token Telegram en état global après son enregistrement; le champ de saisie est effacé. Les mutations mettent à jour l'écran depuis le snapshot retourné; en cas de conflit, le client recharge le snapshot et montre l'erreur backend.

Vite ne fait qu'un proxy de développement côté serveur et ne change ni le client same-origin ni le bundle de production. Les tests UI injectent un serveur API contrôlé, sans hooks réels ni déplacement souris.

## Compatibilité métier

Le stockage suit les macros `{name, updated_at, sha1, steps}` et les événements `{t, type, data}`. Les types valides sont `mouse_move`, `mouse_click`, `scroll`, `key_down`, `key_up`, `nop`. Un événement inconnu ou JSON invalide rend la macro non lisible et visible comme problème, sans réécrire le fichier source. Les macros sont des données utilisateur ordinaires; aucune macro système « Recharger COC » ou « Valider arrivée » n’est active ou protégée. Au démarrage, les anciens placeholders « Recharger COC » et « Valider arrivée » ne sont supprimés que si leur manifeste, leur empreinte et leur contenu vide établissent qu’ils sont intacts. Les raccourcis initiaux restent F1, Ctrl+Shift+1 et Ctrl+Shift+0.

Les données `data` attendues sont `mouse_move: {x,y}`, `mouse_click: {x,y,button,action}` (`button` = left/right/middle, `action` = down/up), `scroll: {x,y,dx,dy}`, `key_down|key_up: {key}`, `nop: {}`. Les coordonnées sont des entiers signés pour les écrans à origine négative; les temps `t` sont finis et non négatifs. Le parseur vérifie le type et les champs avant lecture; il ne remplace jamais une valeur inconnue par `nop`. L’application ne crée aucun fichier de macro système au premier démarrage; elle ne répare ni n’écrase les fichiers utilisateur invalides.

Sémantique globale conservée et rendue explicite: F1 démarre la macro sélectionnée quand l'application est idle, arrête un replay quand il tourne, et ne fait rien pendant un enregistrement; Ctrl+Shift+1 démarre uniquement depuis idle; Ctrl+Shift+0 arrête un replay ou finalise/sauvegarde un enregistrement actif. Le bouton Arrêter l'enregistrement suit la même finalisation et le même trim final.
