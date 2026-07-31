# AUTO-COC v4

Application Windows d’enregistrement et de lecture de macros clavier/souris, avec contrôle Telegram.

## Fonctionnalités

- Console opérateur PyQt6 sombre, épurée et orientée action.
- Enregistrement clavier/souris avec délai de préparation.
- Lecture à timing absolu et mode boucle.
- Arrêt d’urgence local et global.
- Macros utilisateur et routines système présentées dans deux espaces distincts.
- Contrôle Telegram asynchrone.
- Capture d’écran et lancement de CoC via un lanceur Windows configurable (`.exe` ou `.lnk`).
- Détection CoC par processus, chemin d’exécutable et titre de fenêtre.
- Safeguard optionnel : arrête automatiquement la lecture si CoC disparaît plusieurs contrôles de suite.
- Diagnostics et journal d’activité intégrés.
- Sauvegarde atomique des paramètres et macros.

## Installation

Prérequis : Python 3.10+ et Windows 10/11.

```bash
pip install -r requirements.txt
python main.pyw
```

## Utilisation

La console est organisée autour de trois informations immédiates : la macro active, l’état de CoC et l’état de lecture. Les routines intégrées comme `Recharger COC` et `Valider arrivée` apparaissent dans une zone dédiée et restent non éditables. Les actions de création, renommage et suppression ne concernent que les macros personnelles.

Dans `Paramètres` :

- configure le chemin du lanceur CoC ;
- renseigne les noms de processus séparés par `|` si nécessaire ;
- renseigne les fragments de titres de fenêtre séparés par `|` ;
- active `Safeguard CoC` pour arrêter une macro si CoC n’est plus détecté.

Le bouton `Ouvrir CoC` demande le lancement, puis l’interface attend une confirmation de présence dans le délai configuré. Aucun lancement n’est considéré comme validé uniquement parce que le processus du lanceur a démarré. Les raccourcis du menu Démarrer dont Windows masque l’extension `.lnk` sont aussi résolus automatiquement.

## Selftests

Les tests headless vérifient les modèles, la persistance, le timing du lecteur, le contraste du thème et le profil de détection CoC.

```bash
python main.pyw --selftest
```

## Raccourcis

| Raccourci | Action |
|---|---|
| `F1` | Lancer ou arrêter |
| `Ctrl+Shift+1` | Lancer la macro sélectionnée |
| `Ctrl+Shift+0` | Arrêter immédiatement |
| `Ctrl+N` | Créer une macro |
| `Ctrl+F` | Rechercher une macro |
| `Ctrl+Entrée` | Lancer la macro sélectionnée |
| `Échap` | Arrêter ou fermer l’action courante |

## Organisation

```text
main.pyw                  # Bootstrap QApplication et selftests
gui/
  app.py                  # QMainWindow et navigation
  controller.py           # État, actions et signaux Qt
  components.py           # Widgets réutilisables
  dialogs.py              # Dialogues PyQt6
  theme.py                # Palette et feuille de style QSS
  pages/                  # Console, bibliothèque, Telegram, diagnostics
models/                   # Modèles de macros
services/                 # Enregistrement, lecture, Telegram, système
  coc/                    # Profil, lancement, détection et safeguard CoC
utils/                    # Persistance, logging et utilitaires
config/macros/            # Macros utilisateur + routines système distribuées
```

## Données et sécurité

Les macros et paramètres restent locaux dans `config/`. Le token Telegram ne doit pas être partagé ni versionné. Le champ de configuration masque le token dans l’interface et conserve la valeur actuelle lorsqu’il est laissé vide.

## Licence

MIT.
