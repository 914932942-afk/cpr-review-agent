import { type ReactNode, useEffect, useRef, useState } from 'react';
import { entryTitle, type HistoryPage } from './HistoryPanel';
import Icon from './Icon';

export default function ReviewConversation({
  sessionId,
  runId,
  status,
  currentReply,
  disabled,
  onOpen,
  names,
  pendingMessage,
}: {
  sessionId: string | null;
  runId: string | null;
  status: string | undefined;
  currentReply: ReactNode;
  disabled: boolean;
  onOpen: (id: string) => void;
  names: Record<string, string>;
  pendingMessage: string;
}) {
  const [history, setHistory] = useState<HistoryPage | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const scroll = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let disposed = false;
    setHistory(null);
    setUnavailable(false);
    if (sessionId)
      fetch(`/api/sessions/${encodeURIComponent(sessionId)}/history`)
        .then(async (r) => {
          if (!r.ok) throw new Error();
          return (await r.json()) as HistoryPage;
        })
        .then((value) => {
          if (!disposed) setHistory(value);
        })
        .catch(() => {
          if (!disposed) setUnavailable(true);
        });
    return () => {
      disposed = true;
    };
  }, [sessionId, runId, status]);
  useEffect(() => {
    const el = scroll.current;
    if (el) el.scrollTop = 0;
  }, [runId, status, history]);
  const current = history?.items.find((i) => i.run_id === runId);
  const older = current
    ? history?.items
        .filter((i) => i.created_at <= current.created_at && i.run_id !== runId)
        .slice(0, 2)
        .reverse() || []
    : [];
  const scopeDescription = current?.target_label_ids.length
    ? current.target_label_ids.length === Object.keys(names).length
      ? `all ${current.target_label_ids.length} categories`
      : current.target_label_ids.length > 2
        ? `${current.target_label_ids.length} selected categories`
        : current.target_label_ids.map((id) => names[id] || 'Selected category').join(', ')
    : '';
  const question =
    current?.message ||
    (current
      ? `${entryTitle(current)}${scopeDescription ? ': ' + scopeDescription : ''}`
      : pendingMessage);
  return (
    <>
      <div className="conversation-toolbar">
        <span>
          <span className="small-dot" /> This practice
        </span>
        {history?.latest_recap && history.latest_recap.run_id !== runId && (
          <button disabled={disabled} onClick={() => onOpen(history.latest_recap!.run_id)}>
            <Icon name="bookmark" size={13} /> Latest recap
          </button>
        )}
      </div>
      <div
        ref={scroll}
        className="conversation-scroll"
        aria-label="Review conversation"
        aria-live="polite"
      >
        {!!older.length && (
          <details className="earlier-conversation" key={runId}>
            <summary>
              <Icon name="clock" size={13} /> Earlier in this practice{' '}
              <Icon name="chevron" size={12} />
            </summary>
            {older.map((entry) => (
              <div className="past-exchange" key={entry.run_id}>
                <p className="user-message">{entry.message || entryTitle(entry)}</p>
                <button
                  className="saved-reply"
                  disabled={disabled}
                  onClick={() => onOpen(entry.run_id)}
                >
                  <Icon name="file" size={15} />
                  <span>
                    {entry.status === 'completed'
                      ? entryTitle(entry)
                      : entry.status === 'cancelled'
                        ? 'Request cancelled'
                        : 'Review record'}
                    <small>Open saved conversation</small>
                  </span>
                  <Icon name="chevron" size={13} />
                </button>
              </div>
            ))}
          </details>
        )}
        {runId && question && <p className="user-message current-question">{question}</p>}
        {currentReply || (
          <div className="assistant-welcome">
            <span className="welcome-mark">
              <Icon name="spark" size={25} />
            </span>
            <h3>A clearer view of your practice.</h3>
            <p>
              {sessionId
                ? 'Start a review, then ask about a finding or make a recap for your next practice.'
                : 'Add a practice video to get started. I can help you understand the findings and keep a useful recap.'}
            </p>
            <div className="welcome-example">
              <Icon name="chat" size={15} />
              <span>“What should I look at in this finding?”</span>
            </div>
          </div>
        )}
        {unavailable && (
          <p className="muted" role="status">
            Conversation history is temporarily unavailable. Your current review is still shown.
          </p>
        )}
      </div>
    </>
  );
}
