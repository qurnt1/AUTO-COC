# AUTO-COC Design System

## Direction

### Thesis

Une console opérateur calme et lisible qui montre ce qui va se passer, ce qui est détecté et comment arrêter l’automatisation.

### Own-world

Surfaces graphite chaudes, accent citron doux pour l’action principale, ambre pour l’attention et corail uniquement pour l’arrêt ou l’enregistrement. Les routines intégrées sont visuellement séparées des macros créées par l’utilisateur.

### Story

L’utilisateur sélectionne une macro, voit immédiatement si elle est prête, puis lance ou arrête l’exécution. Les états distants Telegram, CoC et le journal d’activité restent visibles sans ouvrir une fenêtre secondaire.

### First viewport

Une barre latérale étroite porte la navigation. Le centre montre la macro active, l’état de CoC et les commandes. La bibliothèque distingue clairement « Mes macros » et « Routines intégrées ». L’action primaire et l’arrêt restent au-dessus de la ligne de flottaison.

### Form

Console opérateur, mode Operate, densité standard à élevée. Les panneaux structurent les responsabilités, mais chaque panneau contient une action ou une information opérationnelle réelle.

## Tokens

- Background: `#121417`
- Surface: `#1A1E22`
- Surface raised: `#22282E`
- Border: `#343C43`
- Text: `#F5F1E8`
- Muted text: `#B5B8B0`
- Accent: `#C7F36B`
- Warning: `#F5B967`
- Danger: `#FF7A68`
- Info: `#B19CFF`
- Radius: 12px
- Spacing: 8px base rhythm

## Motion

Transitions are short and purposeful: 160–220ms for page/state changes. Recording uses a restrained pulse on the status dot. No decorative animation is used on dense macro data.

## Accessibility

Every action has a text label or accessible name. Focus rings use the accent color. Status color is always paired with text. Disabled controls remain readable.
