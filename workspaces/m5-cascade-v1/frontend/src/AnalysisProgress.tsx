import Icon from './Icon';
type StageTool = { request: { tool_id: string }; status: string; duration_ms: number | null };
const names: Record<string, string> = {
  'pose.extract': 'Reading your video',
  'skeleton.predict': 'Reviewing movement',
  'rgb.predict': 'Looking at visual details',
  'fusion.predict': 'Preparing your findings',
  'video.analyze': 'Reviewing your practice',
};
export default function AnalysisProgress({
  status,
  tools,
  toolIds,
  reused,
}: {
  status: string;
  tools: StageTool[];
  toolIds: string[];
  reused: boolean;
}) {
  const active = tools.find((t) => t.status === 'running' || t.status === 'queued');
  const stopped = ['failed', 'rejected', 'cancelled'].includes(status);
  return (
    <section
      className={`analysis-progress ${stopped ? 'stopped' : ''}`}
      aria-label="Analysis progress"
      role="status"
      aria-live="polite"
    >
      <span className={stopped ? 'progress-symbol' : 'progress-symbol spinning'}>
        <Icon name={stopped ? 'info' : 'refresh'} />
      </span>
      <div>
        <strong>
          {status === 'cancelled'
            ? 'Review cancelled'
            : stopped
              ? 'This review could not be completed'
              : reused
                ? 'Your saved analysis is ready'
                : names[active?.request.tool_id || ''] || 'Preparing your video review'}
        </strong>
        <p>
          {stopped
            ? 'Your saved reviews are still available.'
            : reused
              ? 'Your video did not need to be analysed again.'
              : 'This may take a few minutes. Your progress is saved.'}
        </p>
        {!stopped && !reused && (
          <div className="progress-segments" aria-label="Video review stages">
            {toolIds.map((id) => (
              <span
                key={id}
                className={tools.find((t) => t.request.tool_id === id)?.status || 'waiting'}
                title={names[id] || 'Review stage'}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
