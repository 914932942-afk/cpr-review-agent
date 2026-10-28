import Icon from './Icon';
export type BoundContext = {
  session_id: string;
  video_name: string | null;
  source_status: 'available' | 'unavailable' | 'none';
  source_error_code: string | null;
  active_label_ids: number[];
  source: { report_id: string; execution_id: string } | null;
};
export default function ContextPanel({
  context,
  labelNames,
  error,
  activity,
}: {
  context: BoundContext | null;
  labelNames: Record<string, string>;
  error: string;
  activity?: 'reviewing' | 'waiting';
}) {
  return (
    <div
      className={`review-context ${error || context?.source_status === 'unavailable' ? 'context-warning' : ''}`}
      aria-label="Current review context"
    >
      <Icon name={error || context?.source_status === 'unavailable' ? 'info' : 'file'} size={17} />
      <div>
        {error ? (
          <>
            <strong>Reopen a saved review</strong>
            <p>We couldn't confirm the current source.</p>
          </>
        ) : context ? (
          <>
            <strong>{context.video_name || 'Your practice video'}</strong>
            <p>
              {context.source_status === 'available'
                ? `Discussing ${context.active_label_ids.length === 1 ? labelNames[context.active_label_ids[0]] : `${context.active_label_ids.length} review categories`}`
                : context.source_status === 'unavailable'
                  ? 'Saved evidence is unavailable. Choose another review.'
                  : activity === 'reviewing'
                    ? 'Review in progress'
                    : activity === 'waiting'
                      ? 'Waiting for your confirmation'
                      : context.active_label_ids.length
                        ? 'Starting focus selected · ready for its first review'
                        : 'Ready for its first review'}
            </p>
          </>
        ) : (
          <p>Preparing this review…</p>
        )}
      </div>
    </div>
  );
}
