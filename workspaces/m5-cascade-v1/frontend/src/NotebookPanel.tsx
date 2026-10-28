import { useEffect, useState } from 'react';
import Icon from './Icon';
import './review-workspace.css';

type Item = { label_id: number; status: 'to_review' | 'reviewed' | 'next_focus' };
type Notebook = {
  session_id: string;
  revision: number;
  note: string;
  items: Item[];
  updated_at: number | null;
};
type Draft = { base_revision: number; note: string; items: Item[] };
const statusNames = {
  to_review: 'Confirm later',
  reviewed: 'Reviewed',
  next_focus: 'Next practice',
};
const draftKey = (sid: string) => `cpr.notebookDraft.${sid}`;
function loadDraft(sid: string): Draft | null {
  try {
    return JSON.parse(localStorage.getItem(draftKey(sid)) || 'null');
  } catch {
    return null;
  }
}
function storeDraft(sid: string, draft: Draft | null) {
  try {
    if (draft) localStorage.setItem(draftKey(sid), JSON.stringify(draft));
    else localStorage.removeItem(draftKey(sid));
  } catch {
    /* Saved server notes remain available if browser storage is full. */
  }
}
export default function NotebookPanel({
  sessionId,
  names,
  onSaved,
}: {
  sessionId: string;
  names: Record<string, string>;
  onSaved: () => void;
}) {
  const [saved, setSaved] = useState<Notebook | null>(null);
  const [note, setNote] = useState(''),
    [items, setItems] = useState<Item[]>([]);
  const [error, setError] = useState(''),
    [loading, setLoading] = useState(true),
    [saving, setSaving] = useState(false);
  const [conflict, setConflict] = useState(false),
    [refresh, setRefresh] = useState(0);
  const [category, setCategory] = useState('');
  const changed =
    !!saved && (note !== saved.note || JSON.stringify(items) !== JSON.stringify(saved.items));
  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setError('');
    fetch(`/api/sessions/${encodeURIComponent(sessionId)}/notebook`, { signal: ctrl.signal })
      .then(async (r) => {
        if (!r.ok) throw Error('Your saved notes could not be loaded.');
        return r.json() as Promise<Notebook>;
      })
      .then((v) => {
        if (ctrl.signal.aborted) return;
        setSaved(v);
        const d = loadDraft(sessionId);
        const valid =
          d &&
          typeof d.note === 'string' &&
          d.note.length <= 2000 &&
          Array.isArray(d.items) &&
          d.items.length <= 11 &&
          new Set(d.items.map((i) => i.label_id)).size === d.items.length &&
          d.items.every((i) => names[i.label_id] && i.status in statusNames);
        setNote(valid ? d.note : v.note);
        setItems(valid ? d.items : v.items);
        setConflict(!!valid && d.base_revision !== v.revision);
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [sessionId, refresh]);
  useEffect(() => {
    if (!saved || loading || conflict) return;
    storeDraft(sessionId, changed ? { base_revision: saved.revision, note, items } : null);
  }, [sessionId, saved, note, items, loading, conflict, changed]);
  function reloadSaved() {
    if (!window.confirm('Discard the local draft and reload the saved notes?')) return;
    storeDraft(sessionId, null);
    // Block draft persistence in this same render before starting the reload effect.
    setLoading(true);
    setConflict(false);
    setRefresh((v) => v + 1);
  }
  async function save() {
    if (!saved || conflict || saving) return;
    setSaving(true);
    setError('');
    const sent = { revision: saved.revision, note, items };
    try {
      const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/notebook`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID() },
        body: JSON.stringify(sent),
      });
      if (r.status === 409) {
        setConflict(true);
        throw Error(
          'These notes changed in another window. Your local draft is kept; reload the saved version before continuing.',
        );
      }
      if (!r.ok) throw Error('Notes could not be saved. Your draft is kept in this browser.');
      const value = (await r.json()) as Notebook;
      setSaved(value);
      storeDraft(sessionId, null);
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  const available = Object.keys(names).filter(
    (id) => !items.some((i) => i.label_id === Number(id)),
  );
  const choice = available.includes(category) ? category : available[0] || '';
  return (
    <section className="surface review-notebook" aria-label="Personal review notes">
      <div className="notebook-heading">
        <Icon name="bookmark" size={20} />
        <div>
          <span className="card-step">YOUR OWN REFLECTION</span>
          <h2>Notes for next time</h2>
        </div>
      </div>
      <p>
        Keep a personal checklist. Marking an item “Reviewed” records that you looked at it; it does
        not confirm corrected technique.
      </p>
      {loading ? (
        <p role="status">Loading your notes…</p>
      ) : (
        <>
          {error && (
            <p role="status" className="workspace-message">
              {error}
            </p>
          )}
          {conflict && (
            <p className="workspace-message">
              A local draft and a newer saved version are available. Copy any draft text you need,
              then{' '}
              <button className="quiet" onClick={reloadSaved}>
                Reload saved notes
              </button>
              .
            </p>
          )}
          {saved && (
            <>
              <label htmlFor="personal-review-note">
                Personal note <span className="workspace-footnote">({note.length}/2000)</span>
              </label>
              <textarea
                id="personal-review-note"
                value={note}
                maxLength={2000}
                disabled={saving}
                readOnly={conflict}
                placeholder="What would you like to revisit?"
                onChange={(e) => setNote(e.target.value)}
              />
              <ul className="notebook-checklist">
                {items.map((item) => (
                  <li key={item.label_id}>
                    <span>{names[item.label_id]}</span>
                    <select
                      aria-label={`Status for ${names[item.label_id]}`}
                      disabled={saving || conflict}
                      value={item.status}
                      onChange={(e) =>
                        setItems((old) =>
                          old.map((i) =>
                            i.label_id === item.label_id
                              ? { ...i, status: e.target.value as Item['status'] }
                              : i,
                          ),
                        )
                      }
                    >
                      {Object.entries(statusNames).map(([key, label]) => (
                        <option key={key} value={key}>
                          {label}
                        </option>
                      ))}
                    </select>
                    <button
                      className="icon-button"
                      aria-label={`Remove ${names[item.label_id]} from checklist`}
                      disabled={saving || conflict}
                      onClick={() =>
                        setItems((old) => old.filter((i) => i.label_id !== item.label_id))
                      }
                    >
                      <Icon name="close" size={15} />
                    </button>
                  </li>
                ))}
              </ul>
              {!!available.length && (
                <div className="notebook-add">
                  <select
                    aria-label="Checklist category"
                    value={choice}
                    disabled={saving || conflict}
                    onChange={(e) => setCategory(e.target.value)}
                  >
                    {available.map((id) => (
                      <option key={id} value={id}>
                        {names[id]}
                      </option>
                    ))}
                  </select>
                  <button
                    className="secondary"
                    disabled={saving || conflict || !choice}
                    onClick={() =>
                      setItems((old) => [...old, { label_id: Number(choice), status: 'to_review' }])
                    }
                  >
                    <Icon name="plus" size={14} /> Add item
                  </button>
                </div>
              )}
              <div className="notebook-save">
                <span role="status">
                  {saving
                    ? 'Saving…'
                    : changed
                      ? 'Unsaved draft · kept in this browser'
                      : saved.revision
                        ? 'Saved for this practice'
                        : 'No saved notes yet'}
                </span>
                <button
                  className="secondary"
                  disabled={!changed || saving || conflict}
                  onClick={() => void save()}
                >
                  <Icon name="check" size={15} /> Save notes
                </button>
              </div>
              <p className="workspace-footnote">
                Save “Next practice” items to make them available when you add another video. You
                decide whether to use them. Personal note text is not sent to the assistant or used
                as training data.
              </p>
            </>
          )}
        </>
      )}
    </section>
  );
}
