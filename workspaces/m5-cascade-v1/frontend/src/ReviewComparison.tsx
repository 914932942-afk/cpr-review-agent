import { useEffect, useRef, useState } from 'react';
import Icon from './Icon';
import './review-workspace.css';

type RecordInfo = {
  report_id: string;
  session_id: string;
  asset_id: string;
  video_name: string;
  created_at: number;
  scope_label_ids: number[];
  source_available?: boolean;
};
type Comparison = {
  status: string;
  reason?: string;
  left: RecordInfo;
  right: RecordInfo;
  common_label_ids?: number[];
  left_only_label_ids?: number[];
  right_only_label_ids?: number[];
  same_video_file?: boolean;
  rows: { label_id: number; name: string; left_state: string; right_state: string }[];
};
const date = (n: number) =>
  new Date(n * 1000).toLocaleString('en', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
const state = (s: string) =>
  s === 'model_flagged' ? 'Flagged for review' : s === 'not_flagged' ? 'Not flagged' : 'Uncertain';
const reasons: Record<string, string> = {
  SAME_PRACTICE: 'Choose a report from a different practice.',
  DIFFERENT_MODEL_VERSION: 'These reports use different model versions and cannot be compared.',
  SOURCE_UNAVAILABLE: 'One saved source is unavailable. No comparison has been made.',
  NO_SHARED_SCOPE: 'These reports have no reviewed categories in common.',
};

export default function ReviewComparison({ reportId }: { reportId: string }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [options, setOptions] = useState<RecordInfo[]>([]);
  const [selected, setSelected] = useState('');
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Comparison | null>(null);
  function close() {
    dialog.current?.querySelectorAll('video').forEach((v) => v.pause());
    dialog.current?.close();
    setOpen(false);
  }
  useEffect(() => {
    if (!open) return;
    const ctrl = new AbortController();
    setLoading(true);
    setError('');
    setResult(null);
    setOptions([]);
    setSelected('');
    fetch(`/api/reports/${encodeURIComponent(reportId)}/comparison-options`, {
      signal: ctrl.signal,
    })
      .then(async (r) => {
        if (!r.ok) throw Error('Saved reports could not be loaded.');
        return r.json();
      })
      .then((v) => {
        if (!ctrl.signal.aborted) {
          setOptions(v.items);
          if (v.status !== 'available') setError(reasons.SOURCE_UNAVAILABLE);
        }
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [reportId, open]);
  useEffect(() => {
    if (!open || !selected) return;
    const ctrl = new AbortController();
    setLoading(true);
    setResult(null);
    setError('');
    const params = new URLSearchParams({ left_report_id: selected, right_report_id: reportId });
    fetch(`/api/review-comparison?${params}`, { signal: ctrl.signal })
      .then(async (r) => {
        if (!r.ok) throw Error('This comparison could not be verified.');
        return r.json();
      })
      .then((v) => {
        if (!ctrl.signal.aborted) setResult(v);
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [reportId, selected, open]);
  return (
    <>
      <button
        className="quiet compare-records-button"
        onClick={() => {
          setOpen(true);
          dialog.current?.showModal();
        }}
      >
        <Icon name="file" size={16} /> Compare practice records
      </button>
      <dialog
        ref={dialog}
        className="record-comparison-dialog"
        aria-label="Practice record comparison"
        onCancel={() => setOpen(false)}
        onClose={() => {
          dialog.current?.querySelectorAll('video').forEach((v) => v.pause());
          setOpen(false);
        }}
        onClick={(e) => {
          if (e.target === e.currentTarget) close();
        }}
      >
        <div className="review-dialog-heading">
          <div>
            <span className="card-step">TWO SAVED PRACTICES</span>
            <h2>Compare the records</h2>
          </div>
          <button className="icon-button" aria-label="Close record comparison" onClick={close}>
            <Icon name="close" size={20} />
          </button>
        </div>
        <p>
          View both videos and the categories checked in both reports. Each video remains a separate
          practice.
        </p>
        <label htmlFor="comparison-record">Compare with</label>
        <select
          id="comparison-record"
          value={selected}
          disabled={loading && !options.length}
          onChange={(e) => setSelected(e.target.value)}
        >
          <option value="">Choose a saved report</option>
          {options.map((o) => (
            <option key={o.report_id} value={o.report_id} disabled={!o.source_available}>
              {o.video_name} · {date(o.created_at)} · {o.scope_label_ids.length} categories
              {o.source_available ? '' : ' · Source unavailable'}
            </option>
          ))}
        </select>
        {loading && <p role="status">Verifying the saved records…</p>}
        {error && (
          <p role="status" className="workspace-message">
            {error}
          </p>
        )}
        {!loading && !error && !options.length && (
          <p>No other verified practice reports are available yet.</p>
        )}
        {result && result.status !== 'available' && (
          <p role="status" className="workspace-message">
            {reasons[result.reason || ''] || 'These reports cannot be compared.'}
          </p>
        )}
        {result?.status === 'available' && (
          <>
            {result.same_video_file && (
              <p className="workspace-message">
                Both practices use the same video file. They are separate records of the same
                recording.
              </p>
            )}
            <div className="record-videos">
              {[result.left, result.right].map((r, i) => (
                <article key={r.report_id}>
                  <span className="card-step">{i ? 'CURRENT RECORD' : 'OTHER RECORD'}</span>
                  <h3>{r.video_name}</h3>
                  <p>{date(r.created_at)}</p>
                  <video
                    controls
                    preload="metadata"
                    playsInline
                    src={`/api/assets/${encodeURIComponent(r.asset_id)}/media`}
                    aria-label={i ? 'Current practice video' : 'Other practice video'}
                  />
                </article>
              ))}
            </div>
            <h3>
              {result.rows.length} shared {result.rows.length === 1 ? 'category' : 'categories'}
            </h3>
            <div className="record-table-wrap">
              <table className="record-table">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Other record</th>
                    <th>Current record</th>
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((r) => (
                    <tr key={r.label_id}>
                      <th scope="row">{r.name}</th>
                      <td className={r.left_state === 'model_flagged' ? 'record-flag' : ''}>
                        {state(r.left_state)}
                      </td>
                      <td className={r.right_state === 'model_flagged' ? 'record-flag' : ''}>
                        {state(r.right_state)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {!!(
              (result.left_only_label_ids?.length || 0) + (result.right_only_label_ids?.length || 0)
            ) && (
              <p className="workspace-footnote">
                Excluded from the comparison: {result.left_only_label_ids?.length || 0} categories
                checked only in the other report; {result.right_only_label_ids?.length || 0} only in
                the current report.
              </p>
            )}
            <p className="workspace-footnote">
              These are model records, not an improvement score or a competence assessment. “Not
              flagged” does not establish correct CPR technique. Video playback is independent.
            </p>
          </>
        )}
      </dialog>
    </>
  );
}
