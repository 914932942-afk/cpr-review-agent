import { useEffect, useRef, useState } from 'react';
import Icon from './Icon';

export type Workspace = {
  session_id: string;
  asset_id: string | null;
  clip_id: string | null;
  name: string;
  bundle_compatible?: boolean;
  source_report_id?: string | null;
};
type SavedWorkspace = Workspace & {
  last_active_at: number;
  latest_run_id: string | null;
  latest_status: string | null;
};
export type HistoryEntry = {
  turn_id: string;
  run_id: string;
  created_at: number;
  message: string;
  intent: string | null;
  target_label_ids: number[];
  status: string;
  error_code: string | null;
  report_id: string | null;
  title: string | null;
  summary: string | null;
};
export type HistoryPage = {
  items: HistoryEntry[];
  has_more: boolean;
  next_before_turn_id: string | null;
  latest_recap?: { run_id: string; scope_label_ids: number[]; created_at: number } | null;
};
const titles: Record<string, string> = {
  show_example: 'Reference comparison',
  analyse: 'Video review',
  explain: 'Finding explained',
  summarise: 'Saved recap',
  verify: 'Finding review',
  manage_memory: 'Focus preferences',
};
export const entryTitle = (entry: HistoryEntry) => titles[entry.intent || ''] || 'Review request';
const statusLabel = (state: string | null) =>
  state === 'completed'
    ? 'Saved'
    : state === 'waiting_user'
      ? 'Needs your input'
      : state === 'cancelled'
        ? 'Cancelled'
        : state === 'failed' || state === 'rejected'
          ? 'Not completed'
          : state
            ? 'In progress'
            : 'Ready to review';
const date = (value: number) =>
  new Date(value * 1000).toLocaleDateString('en', { month: 'short', day: 'numeric' });
async function get<T>(url: string): Promise<T> {
  const r = await fetch('/api' + url);
  if (!r.ok) throw new Error('Could not load saved reviews. Please try again.');
  return (await r.json()) as T;
}

export default function HistoryPanel({
  workspace,
  runId,
  refreshKey,
  disabled,
  names,
  onOpen,
}: {
  workspace: Workspace | null;
  runId: string | null;
  refreshKey: string;
  disabled: boolean;
  names: Record<string, string>;
  onOpen: (ws: Workspace, id: string | null) => void;
}) {
  const [saved, setSaved] = useState<SavedWorkspace[]>([]),
    [page, setPage] = useState<HistoryPage | null>(null);
  const [error, setError] = useState(''),
    [loading, setLoading] = useState(false),
    [revision, setRevision] = useState(0),
    [showAll, setShowAll] = useState(false);
  const sid = workspace?.session_id;
  const currentSid = useRef(sid);
  currentSid.current = sid;
  useEffect(() => {
    let disposed = false;
    get<SavedWorkspace[]>('/workspaces')
      .then((value) => {
        if (!disposed) {
          setSaved(value);
          setError('');
        }
      })
      .catch((e) => {
        if (!disposed) setError(e.message);
      });
    return () => {
      disposed = true;
    };
  }, [sid, refreshKey, revision]);
  useEffect(() => {
    let disposed = false;
    setPage(null);
    setLoading(false);
    if (sid)
      get<HistoryPage>(`/sessions/${encodeURIComponent(sid)}/history`)
        .then((value) => {
          if (!disposed) setPage(value);
        })
        .catch((e) => {
          if (!disposed) setError(e.message);
        });
    return () => {
      disposed = true;
    };
  }, [sid, refreshKey, revision]);
  async function older() {
    if (!sid || !page?.next_before_turn_id) return;
    setLoading(true);
    try {
      const value = await get<HistoryPage>(
        `/sessions/${encodeURIComponent(sid)}/history?before_turn_id=${encodeURIComponent(page.next_before_turn_id)}`,
      );
      if (currentSid.current === sid) setPage({ ...value, items: [...page.items, ...value.items] });
    } catch (e) {
      if (currentSid.current === sid) setError((e as Error).message);
    } finally {
      if (currentSid.current === sid) setLoading(false);
    }
  }
  return (
    <section className="session-library" aria-label="Saved practice sessions">
      <div className="sidebar-section-heading">
        <h2>Your practices</h2>
        <button
          className="icon-button"
          aria-label="Refresh saved practices"
          disabled={disabled || loading}
          onClick={() => setRevision((v) => v + 1)}
        >
          <Icon name="refresh" size={15} />
        </button>
      </div>
      <p className="sidebar-caption">Pick up where you left off.</p>
      {error && (
        <p className="inline-error" role="status">
          {error}
        </p>
      )}
      {!saved.length && (
        <div className="library-empty">
          <Icon name="clock" />
          <p>Your saved practices will appear here.</p>
        </div>
      )}
      <div className="session-list">
        {saved.slice(0, showAll ? 50 : 7).map((item) => (
          <button
            key={item.session_id}
            data-session-id={item.session_id}
            className={`session-item ${sid === item.session_id ? 'active' : ''}`}
            disabled={disabled || loading}
            onClick={() => onOpen(item, item.latest_run_id)}
            aria-current={sid === item.session_id ? 'page' : undefined}
          >
            <span className="session-glyph">
              <Icon name="play" size={15} />
            </span>
            <span className="session-copy">
              <strong>{item.name}</strong>
              <small>
                {date(item.last_active_at)} <i>·</i> {statusLabel(item.latest_status)}
              </small>
            </span>
          </button>
        ))}
      </div>
      {saved.length > 7 && (
        <button className="sidebar-text-button" onClick={() => setShowAll((v) => !v)}>
          {showAll ? 'Show fewer practices' : `View all practices (${saved.length})`}
        </button>
      )}
      {workspace?.bundle_compatible === false && (
        <p className="inline-error">
          This is an older review. Read its saved records, or add the video again to start a new
          review.
        </p>
      )}
      {sid && (
        <details className="session-history" key={sid}>
          <summary>
            <Icon name="clock" size={15} /> Review history <Icon name="chevron" size={14} />
          </summary>
          <ol>
            {page?.items.map((entry) => (
              <li key={entry.turn_id} data-run-id={entry.run_id}>
                <button
                  className={runId === entry.run_id ? 'viewing' : ''}
                  disabled={disabled || loading || runId === entry.run_id}
                  onClick={() => workspace && onOpen(workspace, entry.run_id)}
                >
                  <strong>{entryTitle(entry)}</strong>
                  <span>
                    {entry.target_label_ids.length
                      ? entry.target_label_ids
                          .slice(0, 2)
                          .map((i) => names[i] || 'Selected category')
                          .join(' · ')
                      : statusLabel(entry.status)}
                  </span>
                  <small>
                    {date(entry.created_at)} · {statusLabel(entry.status)}
                    {runId === entry.run_id ? ' · Viewing' : ''}
                  </small>
                </button>
              </li>
            ))}
          </ol>
          {page?.items.length === 0 && (
            <p className="sidebar-caption">No reviews for this practice yet.</p>
          )}
          {page?.has_more && (
            <button
              className="sidebar-text-button"
              disabled={disabled || loading}
              onClick={() => void older()}
            >
              {loading ? 'Loading…' : 'Load older reviews'}
            </button>
          )}
        </details>
      )}
    </section>
  );
}
