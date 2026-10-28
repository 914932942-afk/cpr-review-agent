import { useEffect, useRef, useState } from 'react';
import Icon from './Icon';
import './practice-continuity.css';

type PreviousPractice = {
  report_id: string | null;
  kind?: string;
  session_id: string;
  video_name: string;
  created_at: number;
  label_ids: number[];
  same_video_file: boolean;
};
export type ContinuityState = {
  session_id: string;
  revision: number;
  locked: boolean;
  selection: { status: string; label_ids: number[]; source: PreviousPractice | null };
  candidates: PreviousPractice[];
  notebook_candidates?: (PreviousPractice & { notebook_revision: number })[];
};
type Recap = PreviousPractice & {
  recap: {
    summary: string;
    entries: { label_id: number; name: string; meaning: string; recorded_result: string }[];
  };
};
const when = (value: number) =>
  new Date(value * 1000).toLocaleString('en', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

export default function PracticeContinuityPanel({
  data,
  names,
  disabled,
  onUse,
  onUseNotebook,
}: {
  data: ContinuityState;
  names: Record<string, string>;
  disabled: boolean;
  onUse: (id: string | null, revision: number) => void;
  onUseNotebook: (sid: string, sourceRevision: number, revision: number) => void;
}) {
  const saved = data.selection.source;
  const choices = [
    ...new Map(
      [...(saved?.report_id ? [saved] : []), ...data.candidates].map((p) => [p.report_id, p]),
    ).values(),
  ];
  const [selected, setSelected] = useState(saved?.report_id || choices[0]?.report_id || '');
  const [selectedNotebook, setSelectedNotebook] = useState('');
  const notebookChoices = data.notebook_candidates || [];
  const notebook =
    notebookChoices.find((n) => n.session_id === selectedNotebook) || notebookChoices[0];
  const [preview, setPreview] = useState<Recap | null>(null);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState('');
  const dialog = useRef<HTMLDialogElement>(null);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  const item = choices.find((p) => p.report_id === selected) || choices[0];
  if (
    (!choices.length && !notebookChoices.length && data.selection.status === 'not_selected') ||
    (data.locked && data.selection.status === 'not_selected')
  )
    return null;
  async function view() {
    if (!item?.report_id) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setPreview(null);
    setError('');
    setLoading(true);
    dialog.current?.showModal();
    try {
      const r = await fetch(
        `/api/sessions/${encodeURIComponent(data.session_id)}/continuity/recaps/${encodeURIComponent(item.report_id)}`,
        { signal: controller.signal },
      );
      if (!r.ok)
        throw new Error(
          'The earlier recap could not be verified. Your current practice is unchanged.',
        );
      const result = (await r.json()) as Recap;
      if (!controller.signal.aborted) setPreview(result);
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }
  const focus = (labels: number[]) => labels.map((id) => names[id] || 'Review category').join(', ');
  return (
    <>
      <section className="surface practice-continuity" aria-label="Practice continuity">
        <div className="continuity-heading">
          <Icon name="bookmark" size={19} />
          <div>
            <span className="card-step">BETWEEN PRACTICES</span>
            <h2>{saved ? 'Your starting focus' : 'Continue from an earlier practice'}</h2>
          </div>
        </div>
        <p>
          {data.locked
            ? 'Your starting focus is recorded. Earlier notes and findings stay attached to their own practice.'
            : 'Choose a saved next-practice goal or an earlier recap as your starting focus.'}
        </p>
        {data.selection.status === 'unavailable' && (
          <p role="status" className="reference-missing">
            The saved starting focus could not be verified. Choose Focus for this video or start
            fresh.
          </p>
        )}
        {saved && (
          <div className="continuity-saved" role="status">
            <Icon name="check" size={15} />
            <span>
              Starting focus saved: <strong>{focus(data.selection.label_ids)}</strong>
              <small>
                From {saved.kind === 'notebook' ? 'your checklist for ' : ''}
                {saved.video_name} · {when(saved.created_at)}. You can adjust Focus for this review.
              </small>
            </span>
          </div>
        )}
        {!!choices.length && (
          <>
            <label htmlFor="earlier-practice">Earlier recap</label>
            <select
              id="earlier-practice"
              value={item?.report_id || ''}
              disabled={disabled || loading}
              onChange={(e) => setSelected(e.target.value)}
            >
              {choices.map((p) => (
                <option key={p.report_id} value={p.report_id || ''}>
                  {p.video_name} · {when(p.created_at)}
                </option>
              ))}
            </select>
            {item && (
              <p className="continuity-scope">
                Review scope: {focus(item.label_ids)}
                {item.same_video_file && (
                  <span>Same video file as this practice; a separate record.</span>
                )}
              </p>
            )}
            <div className="continuity-actions">
              {!data.locked && (
                <button
                  className="secondary"
                  disabled={disabled || !item || saved?.report_id === item.report_id}
                  onClick={() => item && onUse(item.report_id, data.revision)}
                >
                  <Icon name="arrow" size={15} /> Use this focus
                </button>
              )}
              <button className="quiet" disabled={loading || !item} onClick={() => void view()}>
                <Icon name="file" size={15} /> View earlier recap
              </button>
              {!data.locked && data.selection.status !== 'not_selected' && (
                <button
                  className="quiet"
                  disabled={disabled}
                  onClick={() => onUse(null, data.revision)}
                >
                  Start fresh
                </button>
              )}
            </div>
          </>
        )}
        {!data.locked && !!notebookChoices.length && (
          <div className="notebook-goals">
            <label htmlFor="saved-next-goals">Saved next-practice goals</label>
            <select
              id="saved-next-goals"
              value={notebook?.session_id || ''}
              disabled={disabled}
              onChange={(e) => setSelectedNotebook(e.target.value)}
            >
              {notebookChoices.map((n) => (
                <option key={n.session_id} value={n.session_id || ''}>
                  {n.video_name} · {when(n.created_at)}
                </option>
              ))}
            </select>
            {notebook && (
              <p className="continuity-scope">Your chosen goals: {focus(notebook.label_ids)}</p>
            )}
            <button
              className="secondary"
              disabled={disabled || !notebook}
              onClick={() =>
                notebook &&
                onUseNotebook(notebook.session_id, notebook.notebook_revision, data.revision)
              }
            >
              <Icon name="arrow" size={15} /> Use these goals
            </button>
          </div>
        )}
        {!choices.length && !data.locked && data.selection.status !== 'not_selected' && (
          <button
            className="secondary"
            disabled={disabled}
            onClick={() => onUse(null, data.revision)}
          >
            Start fresh
          </button>
        )}
        <p className="continuity-limit">
          Only your chosen categories carry over. Earlier findings belong to the earlier video,
          rather than this one. This is not a progress score.
        </p>
      </section>
      <dialog
        ref={dialog}
        className="previous-recap-dialog"
        aria-label="Previous practice recap"
        onClick={(e) => {
          if (e.target === e.currentTarget) dialog.current?.close();
        }}
      >
        <div className="previous-recap-header">
          <div>
            <span className="card-step">EARLIER VIDEO ONLY</span>
            <h2>Previous practice recap</h2>
          </div>
          <button
            className="icon-button"
            aria-label="Close previous recap"
            onClick={() => dialog.current?.close()}
          >
            <Icon name="close" size={19} />
          </button>
        </div>
        {loading && <p role="status">Checking the earlier saved source…</p>}
        {error && <p role="status">{error}</p>}
        {preview && (
          <>
            <p className="previous-video-name">
              <strong>{preview.video_name}</strong> · {when(preview.created_at)}
            </p>
            <p className="previous-recap-notice">
              These findings belong to the earlier video. They do not describe your current practice
              or demonstrate improvement.
            </p>
            <p>{preview.recap.summary}</p>
            {preview.recap.entries.map((e) => (
              <article className="previous-recap-entry" key={e.label_id}>
                <h3>{names[e.label_id] || e.name}</h3>
                <p>{e.meaning}</p>
                <span>{e.recorded_result}</span>
              </article>
            ))}
            <a
              className="secondary"
              href={`/api/reports/${encodeURIComponent(preview.report_id || '')}/export`}
              target="_blank"
              rel="noreferrer"
            >
              Open earlier report <Icon name="arrow" size={14} />
            </a>
          </>
        )}
      </dialog>
    </>
  );
}
