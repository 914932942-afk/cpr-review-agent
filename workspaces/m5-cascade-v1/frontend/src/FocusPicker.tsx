export default function FocusPicker({
  labels,
  names,
  selected,
  disabled,
  onChange,
}: {
  labels: number[];
  names: Record<string, string>;
  selected: number[];
  disabled: boolean;
  onChange: (ids: number[]) => void;
}) {
  return (
    <div className="focus-picker" role="group" aria-label="Review focus">
      <div className="focus-picker-heading">
        <strong>What would you like to review?</strong>
        <span>{selected.length} selected</span>
      </div>
      <p className="muted">Choose the categories to display. Findings apply to the whole clip.</p>
      <div className="focus-options">
        {labels.map((id) => (
          <button
            key={id}
            type="button"
            aria-pressed={selected.includes(id)}
            disabled={disabled}
            onClick={() =>
              onChange(
                selected.includes(id)
                  ? selected.filter((i) => i !== id)
                  : [...selected, id].sort((a, b) => a - b),
              )
            }
          >
            <span className="choice-mark" aria-hidden="true">
              {selected.includes(id) ? '✓' : ''}
            </span>
            {names[id] || 'Category'}
          </button>
        ))}
      </div>
      <div className="focus-utilities">
        <button type="button" disabled={disabled} onClick={() => onChange(labels)}>
          Select all
        </button>
        <button type="button" disabled={disabled || !selected.length} onClick={() => onChange([])}>
          Clear
        </button>
      </div>
    </div>
  );
}
