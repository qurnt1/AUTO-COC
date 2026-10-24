import { useState } from "react";
import type { MacroSummary } from "../api";
import { Dialog } from "./Dialog";

export type MacroDialogState =
  | { kind: "create" }
  | { kind: "rename"; macro: MacroSummary }
  | { kind: "delete"; macro: MacroSummary };

export function MacroDialog({ dialog, busy, onClose, onSubmit }: {
  dialog: MacroDialogState;
  busy: boolean;
  onClose: () => void;
  onSubmit: (name: string) => void;
}) {
  const [name, setName] = useState(dialog.kind === "rename" ? dialog.macro.name : "");
  const destructive = dialog.kind === "delete";
  const label = dialog.kind === "create" ? "Nom de la macro" : "Nouveau nom";
  const title = dialog.kind === "create" ? "Créer une macro" : dialog.kind === "rename" ? "Renommer la macro" : "Supprimer cette macro ?";
  const description = destructive
    ? `« ${dialog.macro.name} » sera supprimée de votre bibliothèque. Cette action ne peut pas être annulée.`
    : "Choisissez un nom simple que vous retrouverez facilement.";

  return (
    <Dialog title={title} description={description} onClose={onClose}>
      <form className="dialog-form" onSubmit={(event) => { event.preventDefault(); onSubmit(name.trim()); }}>
        {!destructive && <label className="field-label" htmlFor="macro-name">{label}</label>}
        {!destructive && <input id="macro-name" autoFocus maxLength={80} value={name} onChange={(event) => setName(event.target.value)} placeholder="Ex. Récolte des ressources" />}
        <div className="dialog-actions">
          <button className="button quiet" type="button" onClick={onClose}>Annuler</button>
          <button className={`button ${destructive ? "danger" : "primary"}`} type="submit" disabled={busy || (!destructive && !name.trim())}>
            {busy ? "Patientez…" : destructive ? "Supprimer" : dialog.kind === "create" ? "Créer la macro" : "Enregistrer le nom"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
