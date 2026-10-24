import type { MacroSummary } from "../api";
import { Icon } from "./Icon";
import { filterMacros, handleMacroNavigationKey } from "./macroLibraryUtils";

export function MacroLibrary({ macros, selectedMacro, query, loading, disabled, onQueryChange, onCreate, onSelect }: {
  macros: readonly MacroSummary[];
  selectedMacro: string | null;
  query: string;
  loading: boolean;
  disabled: boolean;
  onQueryChange: (query: string) => void;
  onCreate: () => void;
  onSelect: (name: string) => void;
}) {
  const filtered = filterMacros(macros, query);
  const selectionVisible = filtered.some((macro) => macro.name === selectedMacro);

  return (
    <section className="sidebar-library" aria-labelledby="sidebar-library-title">
      <div className="sidebar-library-heading">
        <div className="sidebar-library-label">
          <h2 id="sidebar-library-title">Bibliothèque</h2>
          <span className="library-count">{query ? `${filtered.length} / ${macros.length}` : loading ? "—" : macros.length}</span>
        </div>
        <button className="icon-button sidebar-add-button" type="button" aria-label="Créer une macro" title="Créer une macro" onClick={onCreate} disabled={disabled}>
          <Icon name="plus" size={16} />
        </button>
      </div>
      <div className="search-box sidebar-search">
        <Icon name="search" size={15} />
        <input aria-label="Rechercher une macro" value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder="Rechercher…" disabled={loading || macros.length === 0} />
        {query && <button type="button" className="clear-search" aria-label="Effacer la recherche" onClick={() => onQueryChange("")}><Icon name="close" size={13} /></button>}
      </div>
      {filtered.length > 0 ? (
        <div className="sidebar-macros" role="listbox" aria-label="Macros">
          {filtered.map((macro, index) => {
            const selected = macro.name === selectedMacro;
            return <button
              key={macro.name}
              id={`macro-${encodeURIComponent(macro.name)}`}
              type="button"
              className={`sidebar-macro ${selected ? "selected" : ""}`}
              role="option"
              aria-selected={selected}
              tabIndex={selected || (!selectionVisible && index === 0) ? 0 : -1}
              onClick={() => onSelect(macro.name)}
              onKeyDown={(event) => {
                const destination = handleMacroNavigationKey(event, index, filtered.length);
                if (destination === null || destination === index) return;
                const target = filtered[destination];
                onSelect(target.name);
                window.requestAnimationFrame(() => document.getElementById(`macro-${encodeURIComponent(target.name)}`)?.focus());
              }}
              disabled={disabled}
            >
              <span className="sidebar-macro-name">{macro.name}</span>
              {selected && <Icon className="sidebar-macro-check" name="check" size={16} />}
            </button>;
          })}
        </div>
      ) : (
        <div className="sidebar-empty" role={loading ? "status" : undefined}>
          <span>{loading ? "Chargement des macros…" : query ? "Aucune macro trouvée." : "Vos macros apparaîtront ici."}</span>
        </div>
      )}
    </section>
  );
}
