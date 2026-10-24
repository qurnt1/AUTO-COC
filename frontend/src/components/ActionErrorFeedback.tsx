import { Icon } from "./Icon";

export function ActionErrorFeedback({ message, onDismiss }: {
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div className="action-feedback error" role="alert">
      <Icon name="activity" size={16} />
      <span>{message}</span>
      <button type="button" className="icon-button subtle" aria-label="Fermer le message" onClick={onDismiss}>
        <Icon name="close" size={15} />
      </button>
    </div>
  );
}
