import { useEffect, useState } from 'react';
import Icon from './Icon';

type Explanation = {
  mode: string;
  intent: string;
  summary: string;
  opening: string;
  scope_label_ids: number[];
  outside_scope_label_ids: number[];
  entries: {
    label_id: number;
    name: string;
    meaning: string;
    state: string;
    recorded_result: string;
    uncalibrated_score: number;
    threshold: number;
    decision_note: string;
  }[];
  limitations: string[];
  next_steps: string[];
  source_execution_id: string;
  organisation_receipt: { code: string; called: boolean };
};
type Result = { status: string; review: Explanation | null; error_code: string | null };

export default function GroundedReview({
  reportId,
  onReplay,
}: {
  reportId: string | null;
  onReplay?: () => void;
}) {
  const [result, setResult] = useState<Result | null>(null);
  useEffect(() => {
    let disposed = false;
    setResult(null);
    if (!reportId) return;
    fetch(`/api/reports/${encodeURIComponent(reportId)}/explanation`)
      .then(async (r) => {
        if (!r.ok) throw new Error('Explanation unavailable');
        return (await r.json()) as Result;
      })
      .then((value) => {
        if (!disposed) setResult(value);
      })
      .catch(() => {
        if (!disposed)
          setResult({ status: 'unavailable', review: null, error_code: 'EXPLANATION_UNAVAILABLE' });
      });
    return () => {
      disposed = true;
    };
  }, [reportId]);
  if (!reportId) return null;
  const review = result?.status === 'available' ? result.review : null;
  return (
    <section className="assistant-answer" aria-label="Evidence-constrained explanation">
      <div className="answer-author">
        <span className="assistant-avatar">
          <Icon name="spark" size={14} />
        </span>
        <strong>Review assistant</strong>
      </div>
      {!result && <p className="muted">Checking the saved source…</p>}
      {result && !review && (
        <p role="status">
          {result.status === 'not_available'
            ? 'This older review has no saved explanation. Ask about the finding to create one from its available evidence.'
            : "I can't verify the evidence for this explanation right now. Open another saved review or add the video again."}
        </p>
      )}
      {review && (
        <>
          <h3>
            {review.intent === 'summarise'
              ? 'Your review recap'
              : review.intent === 'analyse'
                ? 'Your video review'
                : review.scope_label_ids.length > 1
                  ? 'Your selected findings'
                  : 'Let’s look at the saved finding.'}
          </h3>
          <p>
            The model flagged{' '}
            <strong>{review.entries.filter((e) => e.state === 'model_flagged').length}</strong> of{' '}
            <strong>{review.scope_label_ids.length}</strong>{' '}
            {review.scope_label_ids.length === 1 ? 'category' : 'categories'} in this review.
          </p>
          {(review.intent === 'explain' && review.entries.length <= 3
            ? review.entries
            : review.entries.filter((e) => e.state === 'model_flagged').slice(0, 3)
          ).map((entry) => (
            <div className="answer-finding" key={entry.label_id}>
              <strong>{entry.name}</strong>
              <p>
                {entry.meaning.replace(
                  /^A dataset category (referring to|for) /,
                  'This category describes ',
                )}
              </p>
              <span
                className={`text-state ${entry.state === 'model_flagged' ? 'text-state-flagged' : ''}`}
              >
                {entry.state === 'model_flagged'
                  ? 'Flagged in the saved model output; worth reviewing in your video.'
                  : 'Not flagged in the saved output. This does not confirm correct technique.'}
              </span>
            </div>
          ))}
          {!!review.next_steps.length && (
            <div className="answer-next-step">
              <strong>Next step</strong>
              <p>{review.next_steps[0]}</p>
              {onReplay && (
                <button className="quiet" onClick={onReplay}>
                  <Icon name="play" size={14} /> Replay video
                </button>
              )}
            </div>
          )}
          <p className="answer-limit">
            {review.outside_scope_label_ids.length
              ? 'This reply covers your selected focus, rather than every assessed category. '
              : ''}
            These are whole-clip suggestions, not confirmed mistakes or a clinical assessment.
          </p>
          <div className="answer-source">
            <Icon name="check" size={13} /> Source checked <span>·</span>{' '}
            {review.mode === 'model_organised_evidence'
              ? 'AI-assisted explanation'
              : 'From saved results'}
          </div>
        </>
      )}
    </section>
  );
}
