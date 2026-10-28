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
  disabled = false,
  onExplain,
}: {
  reportId: string | null;
  claims: Claim[];
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
  const flagged = claims.filter((c) => c.state === 'model_flagged'),
    others = claims.filter((c) => c.state !== 'model_flagged');
  function finding(claim: Claim) {
    const record = evidence?.findings.find((f) => f.label_id === claim.label_id);
    return (
      <article
        className={`finding-card ${claim.state === 'model_flagged' ? 'is-flagged' : ''}`}
        key={claim.label_id}
      >
        <div className="finding-card-heading">
          <span className="finding-symbol">
            <Icon name={claim.state === 'model_flagged' ? 'info' : 'file'} size={19} />
          </span>
          <div>
            <h3>{claim.name}</h3>
            <span className="finding-status">
              {claim.state === 'model_flagged' ? 'Flagged by the model' : 'Not flagged'}
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
      {flagged.length ? (
        flagged.map(finding)
      ) : (
        <div className="no-flags">
          <Icon name="info" />
          <p>
            No selected category was flagged.
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
