import { useEffect, useRef, type ReactNode } from "react";
import { Icon } from "./Icon";

export function Dialog({ title, description, onClose, children }: {
  title: string;
  description?: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    returnFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog.showModal();
    const initialFocus = dialog.querySelector<HTMLElement>("[autofocus]") ?? dialog.querySelector<HTMLElement>('button:not(:disabled), input:not(:disabled), [tabindex]:not([tabindex="-1"])');
    initialFocus?.focus();
    return () => {
      if (dialog.open) dialog.close();
      window.requestAnimationFrame(() => {
        const target = returnFocus.current;
        if (target?.isConnected && !target.hasAttribute("disabled")) target.focus();
        else document.querySelector<HTMLElement>("#main-content h1")?.focus();
      });
    };
  }, []);

  return (
    <dialog className="dialog" ref={ref} aria-labelledby="dialog-title" onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <div className="dialog-topline">
        <span className="dialog-marker" aria-hidden="true" />
        <button className="icon-button subtle" type="button" aria-label="Fermer" onClick={onClose}><Icon name="close" /></button>
      </div>
      <h2 id="dialog-title">{title}</h2>
      {description && <p className="dialog-description">{description}</p>}
      {children}
    </dialog>
  );
}
