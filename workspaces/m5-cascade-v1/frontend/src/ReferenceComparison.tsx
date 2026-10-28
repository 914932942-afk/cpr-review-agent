import { type RefObject, useEffect, useState } from 'react';
import Icon from './Icon';

export type ReferenceClip = {
  id: string;
  record_id: string;
  label_id: number;
  dataset_label: string;
  kind: 'normal_label' | 'error_label';
  view: string;
  frames: number;
  fps: number;
  duration_seconds: number;
};
export type ReferenceLookup = {
  status: string;
  items: ReferenceClip[];
  default_entry_id?: string | null;
  requested_label_ids?: number[];
  unavailable_label_ids?: number[];
  unavailable_entry_ids?: string[];
  notice?: string;
  dataset?: string;
  dataset_url?: string;
  repository_url?: string;
  dataset_revision?: string;
  license_note?: string;
};

export function ReferenceAnswer({
  data,
  onOpen,
  names,
}: {
  data: ReferenceLookup | null;
  onOpen: () => void;
  names: Record<string, string>;
}) {
  const missing = data?.unavailable_label_ids || [];
  return (
    <section className="assistant-answer" aria-label="Reference search result">
      <div className="answer-author">
        <Icon name="book" size={16} />
        <strong>Review assistant</strong>
      </div>
      <h3>Compare with a research reference</h3>
      {!data ? (
        <p>Checking reference sources…</p>
      ) : data.status !== 'available' ? (
        <p role="status">
          No verified reference is available for this request. Your saved findings are unchanged.
        </p>
      ) : (
        <>
          <p>
            {data.items.some((item) => item.kind === 'normal_label') &&
            data.items.some((item) => item.kind === 'error_label')
              ? 'Choose a dataset-labelled “Correct” reference or an error example to compare with your practice.'
              : data.items.some((item) => item.kind === 'normal_label')
                ? 'A dataset-labelled “Correct” clip is available as a general reference.'
                : 'A dataset-labelled error example is available for comparison.'}
          </p>
          <button className="secondary" onClick={onOpen}>
            <Icon name="play" size={15} /> Open side-by-side
          </button>
          {!!missing.length && (
            <p className="reference-missing">
              A category-specific reference is unavailable for{' '}
              {missing.map((i) => names[i] || 'the selected category').join(', ')}. General
              references are listed separately.
            </p>
          )}
          {!!data.unavailable_entry_ids?.length && (
            <p className="reference-missing">
              Some saved references could not be verified and have been omitted.
            </p>
          )}
        </>
      )}
      <p className="answer-limit">
        These are public research samples with dataset labels, not certified demonstrations. This
        lookup does not assess your video or change its findings.
      </p>
    </section>
  );
}

export default function ReferenceComparison({
  data,
  runId,
  selectedId,
  onSelect,
  onClose,
  onPlayBoth,
  videoRef,
}: {
  data: ReferenceLookup;
  runId: string;
  selectedId: string;
  onSelect: (id: string) => void;
  onClose: () => void;
  onPlayBoth: () => void;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  const [unavailable, setUnavailable] = useState(false);
  const selected = data.items.find((e) => e.id === selectedId);
  useEffect(() => setUnavailable(false), [runId, selectedId]);
  if (!selected) return null;
  return (
    <section className="reference-pane" aria-label="Reference video comparison">
      <div className="comparison-pane-heading">
        <span>
          <Icon name="book" size={14} /> Dataset reference
        </span>
        <button className="icon-button" aria-label="Close comparison" onClick={onClose}>
          <Icon name="close" size={16} />
        </button>
      </div>
      <label className="sr-only" htmlFor="reference-choice">
        Reference clip
      </label>
      <select id="reference-choice" value={selectedId} onChange={(e) => onSelect(e.target.value)}>
        {data.items.map((item) => (
          <option key={item.id} value={item.id}>
            {item.dataset_label} · {item.kind === 'normal_label' ? 'normal label' : 'error example'}
          </option>
        ))}
      </select>
      <div className="video-frame reference-frame">
        <video
          ref={videoRef}
          key={`${runId}:${selectedId}`}
          controls
          muted
          playsInline
          preload="metadata"
          aria-label="Dataset reference video"
          src={`/api/runs/${encodeURIComponent(runId)}/reference-clips/${encodeURIComponent(selectedId)}/media`}
          onError={() => setUnavailable(true)}
        />
      </div>
      {unavailable ? (
        <p className="reference-missing" role="status">
          This reference cannot be played or its source is no longer valid. Close the comparison and
          search again.
        </p>
      ) : (
        <>
          <p className="reference-label">
            Dataset-labelled “{selected.dataset_label}”
            {selected.kind === 'normal_label'
              ? '. Not a certified demonstration.'
              : '. An error example.'}
          </p>
          <button className="quiet play-comparison" onClick={onPlayBoth}>
            <Icon name="play" size={14} /> Play both from start
          </button>
          <p className="comparison-timing">
            Both clips start at zero. Their movements are not time-aligned.
          </p>
        </>
      )}
      <details className="reference-source">
        <summary>
          Source and use information <Icon name="chevron" size={12} />
        </summary>
        <p>
          {data.dataset} · {selected.record_id} · {selected.view} ·{' '}
          {selected.duration_seconds.toFixed(1)} s
        </p>
        <p>
          Label taken from the published annotation. The complete clip was reconstructed from
          existing source frames.
        </p>
        <a href={data.dataset_url} target="_blank" rel="noreferrer">
          Dataset source ↗
        </a>
        <span> · </span>
        <a href={data.repository_url} target="_blank" rel="noreferrer">
          Research project ↗
        </a>
        <p>{data.license_note}</p>
      </details>
    </section>
  );
}
