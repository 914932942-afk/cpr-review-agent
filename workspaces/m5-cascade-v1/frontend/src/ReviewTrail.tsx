import { useEffect, useState } from 'react';

type Entry = {
  run_id: string;
  intent: string | null;
  status: string;
  message: string;
  title: string | null;
  target_label_ids: number[];
};
type Page = {
  items: Entry[];
  latest_recap: { run_id: string; scope_label_ids: number[]; created_at: number } | null;
};

export default function ReviewTrail({
  sessionId,
  runId,
  refreshKey,
  scope,
  disabled,
  onExplain,
  onSummary,
  onOpen,
}: {
  sessionId: string;
  runId: string | null;
  refreshKey: string;
  scope: number[];
  disabled: boolean;
  onExplain: () => void;
  onSummary: () => void;
  onOpen: (id: string) => void;
}) {
  const [page, setPage] = useState<Page | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let disposed = false;
    setPage(null);
    setError(false);
    fetch(`/api/sessions/${encodeURIComponent(sessionId)}/history`)
      .then(async (r) => {
        if (!r.ok) throw new Error('History unavailable');
        return (await r.json()) as Page;
      })
      .then((value) => {
        if (!disposed) setPage(value);
      })
      .catch(() => {
        if (!disposed) setError(true);
      });
    return () => {
      disposed = true;
    };
  }, [sessionId, refreshKey]);
  return (
    <section className="review-trail" aria-label="Continuous review">
      <h4>Continue your review</h4>
      <p className="small-note">
        Current displayed scope: {scope.map((i) => `#${i}`).join(', ') || 'No findings yet'}.
        Follow-ups reuse saved vision results. Each completed request is saved automatically.
      </p>
      <div className="button-row">
        <button disabled={disabled || !scope.length} onClick={onExplain}>
          Explain displayed scope
        </button>
        <button disabled={disabled || !scope.length} onClick={onSummary}>
          Summarise displayed scope
        </button>
        {page?.latest_recap && (
          <button
            disabled={disabled || page.latest_recap.run_id === runId}
            onClick={() => onOpen(page.latest_recap!.run_id)}
          >
            Open latest saved summary
          </button>
        )}
      </div>
      {page?.latest_recap && (
        <p className="small-note">
          Latest saved summary: {page.latest_recap.scope_label_ids.map((i) => `#${i}`).join(', ')} ·{' '}
          {new Date(page.latest_recap.created_at * 1000).toLocaleString()}.
        </p>
      )}
      {error && (
        <p className="small-note" role="status">
          Review trail is unavailable. Refresh history to retry.
        </p>
      )}
      {!!page?.items.length && (
        <details className="review-trail-details" open>
          <summary>Recent review trail · this video</summary>
          <ol>
            {page.items
              .slice(0, 5)
              .reverse()
              .map((entry) => (
                <li key={entry.run_id} className={entry.run_id === runId ? 'current' : ''}>
                  <div>
                    <strong>
                      {entry.message ||
                        {
                          explain: 'Explain saved findings',
                          summarise: 'Summarise saved findings',
                          analyse: 'Review video',
                          manage_memory: 'Update explicit preference',
                        }[entry.intent || ''] ||
                        'Review request'}
                    </strong>
                    <small>
                      {entry.target_label_ids.length
                        ? entry.target_label_ids.map((i) => `#${i}`).join(', ')
                        : 'Scope recorded in result'}{' '}
                      · {entry.status.replaceAll('_', ' ')}
                    </small>
                    <small>{entry.title || 'No completed reply yet'}</small>
                  </div>
                  <button
                    disabled={disabled || entry.run_id === runId}
                    onClick={() => onOpen(entry.run_id)}
                  >
                    {entry.run_id === runId ? 'Current' : 'Reopen'}
                  </button>
                </li>
              ))}
          </ol>
          <p className="small-note">
            Recent five requests. The full session history above includes older, failed and
            cancelled requests. Reopening a record also restores its scope.
          </p>
        </details>
      )}
    </section>
  );
}
