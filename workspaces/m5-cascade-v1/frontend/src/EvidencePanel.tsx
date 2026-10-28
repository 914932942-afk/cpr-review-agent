import { useEffect, useState } from 'react';
import Icon from './Icon';
export type Claim = { label_id: number; name: string; state: string; uncalibrated_score: number };
type Evidence = {
  findings: (Claim & { threshold: number; decision_note: string })[];
  assessed_label_ids: number[];
};
type Result = { status: string; review: Evidence | null; error_code: string | null };
export default function EvidencePanel({
  reportId,
  claims,
  assessment,
  disabled = false,
  onExplain,
}: {
  reportId: string | null;
  claims: Claim[];
  assessment?: { states: Record<string, string>; calibrated_probabilities?: Record<string, number> } | null;
  disabled?: boolean;
  onExplain?: (id: number) => void;
}) {
  const [result, setResult] = useState<Result | null>(null);
  useEffect(() => {
    let disposed = false;
    setResult(null);
    if (reportId)
      fetch(`/api/reports/${encodeURIComponent(reportId)}/evidence`)
        .then(async (r) => {
          if (!r.ok) throw new Error();
          return (await r.json()) as Result;
        })
        .then((value) => {
          if (!disposed) setResult(value);
        })
        .catch(() => {
          if (!disposed)
            setResult({ status: 'unavailable', review: null, error_code: 'EVIDENCE_UNAVAILABLE' });
        });
    return () => {
      disposed = true;
    };
  }, [reportId]);
  const evidence = result?.status === 'available' ? result.review : null;
  const reviewState = (claim: Claim) => assessment?.states[claim.label_id] || claim.state;
  const uncertain = claims.filter((c) => reviewState(c) === 'uncertain');
  const flagged = claims.filter((c) => reviewState(c) === 'model_flagged');
  const others = claims.filter((c) => !['model_flagged', 'uncertain'].includes(reviewState(c)));
  function finding(claim: Claim) {
    const record = evidence?.findings.find((f) => f.label_id === claim.label_id);
    return (
      <article
        className={`finding-card ${reviewState(claim) === 'uncertain' ? 'needs-review' : reviewState(claim) === 'model_flagged' ? 'is-flagged' : ''}`}
        key={claim.label_id}
      >
        <div className="finding-card-heading">
          <span className="finding-symbol">
            <Icon name={claim.state === 'model_flagged' ? 'info' : 'file'} size={19} />
          </span>
          <div>
            <h3>{claim.name}</h3>
            <span className="finding-status">
              {reviewState(claim) === 'uncertain' ? 'Needs another look' : claim.state === 'model_flagged' ? 'Flagged by the model' : 'Not flagged'}
            </span>
          </div>
          {onExplain && (
            <button
              className="finding-ask"
              disabled={disabled || !evidence}
              onClick={() => onExplain(claim.label_id)}
              aria-label={`Explain ${claim.name}`}
            >
              Explain <Icon name="arrow" size={15} />
            </button>
          )}
        </div>
        {reviewState(claim) === 'uncertain' && <p className="reliability-message">The reliability check could not support a confident conclusion. Raw model output: {claim.state === 'model_flagged' ? 'flagged' : 'not flagged'}. Replay the clip and compare the evidence.</p>}
        {record && (
          <details className="finding-evidence">
            <summary>
              How this was assessed <Icon name="chevron" size={13} />
            </summary>
            <p>{record.decision_note}</p>
            <dl>
              <div>
                <dt>Raw branch score</dt>
                <dd>{record.uncalibrated_score.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Branch threshold</dt>
                <dd>{record.threshold.toFixed(4)}</dd>
              </div>
              {assessment?.calibrated_probabilities?.[claim.label_id] !== undefined && <div><dt>Calibrated category estimate</dt><dd>{(assessment.calibrated_probabilities[claim.label_id] * 100).toFixed(1)}%</dd></div>}
            </dl>
            <p>
              These are uncalibrated model readouts, not probabilities of correct CPR. They do not
              fully explain the final combined decision or locate an error in time.
            </p>
          </details>
        )}
      </article>
    );
  }
  return (
    <section className="findings-list" aria-label="Recorded findings and evidence">
      {!result && <p className="muted">Checking the saved evidence…</p>}
      {result && !evidence && (
        <div className="evidence-unavailable" role="status">
          <Icon name="info" size={17} />
          <p>
            Historical findings only. Their source could not be verified; explanations are
            unavailable.
          </p>
        </div>
      )}
      {!!uncertain.length && <div className="reliability-group" role="region" aria-label="Categories needing another look"><h3>Needs another look <span>{uncertain.length}</span></h3><p>These checks remain uncertain. Their original model findings are preserved below.</p>{uncertain.map(finding)}</div>}
      {flagged.length ? (
        flagged.map(finding)
      ) : (
        <div className="no-flags">
          <Icon name="info" />
          <p>
            {uncertain.length ? 'Review the uncertain categories above before drawing a conclusion.' : 'No selected category was flagged.'}
            <small>
              This does not establish correct CPR. You can still review the recorded categories
              below.
            </small>
          </p>
        </div>
      )}
      {!!others.length && (
        <details className="other-findings">
          <summary>
            Other reviewed categories <span>{others.length}</span>
            <Icon name="chevron" size={15} />
          </summary>
          <p className="muted">Not flagged does not mean confirmed correct.</p>
          {others.map(finding)}
        </details>
      )}
      {evidence && (
        <p className="evidence-note">
          <Icon name="check" size={14} /> Saved source checked <span>·</span> Whole clip
        </p>
      )}
    </section>
  );
}
