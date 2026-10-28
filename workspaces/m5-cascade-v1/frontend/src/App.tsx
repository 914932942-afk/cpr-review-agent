import { useEffect, useRef, useState } from 'react';
import AnalysisProgress from './AnalysisProgress';
import ContextPanel, { type BoundContext } from './ContextPanel';
import EvidencePanel from './EvidencePanel';
import GroundedReview from './GroundedReview';
import HistoryPanel, { type Workspace } from './HistoryPanel';
import ReviewConversation from './ReviewConversation';
import FocusPicker from './FocusPicker';
import Icon from './Icon';
import ReferenceComparison, { ReferenceAnswer, type ReferenceLookup } from './ReferenceComparison';
import PracticeContinuityPanel, { type ContinuityState } from './PracticeContinuity';
import ReviewComparison from './ReviewComparison';
import NotebookPanel from './NotebookPanel';

type Goal = {
  intent: string;
  target_label_ids?: number[];
  analysis_mode?: string;
  allow_partial?: boolean;
  analysis_budget_seconds?: number;
  output_options?: { example_scope: 'general' | 'category' };
  memory_commands?: { operation: string; label_ids?: number[] }[];
};
type Response = {
  title: string;
  message: string;
  workflow_reached_end: boolean;
  analysis_status: string;
  target_states: Record<string, string>;
  classification_claims: {
    label_id: number;
    name: string;
    state: string;
    uncalibrated_score: number;
  }[];
  reused_result?: boolean;
  module_assessment?: {
    calibration_status: string;
    calibration_profile: string;
    states: Record<string, string>;
    calibrated_probabilities?: Record<string, number>;
  } | null;
  prediction?: {
    clip_id: string;
    asset_id: string;
    decoded_frames: number;
    reported_fps: number;
    video_sha256: string;
    bundle_sha256: string;
    analysis_seconds: number;
    evidence_scope: string;
    analysis_route?: 'full-v15' | 'body-qualified-v1';
    supported_label_ids?: number[];
  } | null;
  scheduling?: { partial_scope_authorised?: boolean };
  visual_processing?: { mode: "lightweight" | "full_visual"; sampled_frames: number; requested_frames: number; scope_unchanged: boolean } | null;
  memory_receipts: unknown[];
  limitations: string[];
};
type Run = {
  id: string;
  session_id: string;
  status: string;
  revision: number;
  trace_id: string;
  error_code: string | null;
  report_id: string | null;
  response: Response | null;
  reference_requested?: boolean;
  resume_payload: unknown;
  cancel_requested: boolean;
  review_context?: {
    source_report_id: string | null;
    snapshot_id: string;
    target_label_ids: number[];
  };
  wait: null | {
    kind: string;
    context_code?: string;
    wait_id: string;
    interrupt_id?: string;
    question?: string;
    context_snapshot_id?: string;
    source_confirmation_required?: boolean;
    scope_confirmation_required?: boolean;
    bound_video_name?: string;
    bound_source_report_id?: string | null;
  };
};
type Event = {
  seq: number;
  node_id: string | null;
  kind: string;
  payload: Record<string, unknown>;
  created_at: number;
};
type Tool = {
  id: string;
  status: string;
  duration_ms: number | null;
  request: { tool_id: string };
  output: null | { is_stub: boolean };
};
type GatewayStatus = {
  mode: 'disabled' | 'deepseek';
  network_enabled: boolean;
  key_configured: boolean;
  connection_verified: boolean;
  model?: string;
};
type Capabilities = {
  tools?: string[];
  reference_tools?: string[];
  upload_preflight?: boolean;
  review_workspace_tools?: boolean;
  practice_continuity?: boolean;
  context_binding?: string;
  vision_pipeline?: string;
  module_status?: Record<string, string>;
  module_slots?: Record<string, { implementation_id: string; kind: string }>;
  llm: GatewayStatus;
  mode: string;
  nodes: Record<string, string>;
  candidate_labels: number[];
  label_names: Record<string, string>;
};
const terminal = new Set(['completed', 'partially_completed', 'rejected', 'failed', 'cancelled']);

function load<T>(key: string): T | null {
  try {
    return JSON.parse(localStorage.getItem(key) || 'null');
  } catch {
    return null;
  }
}
class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
function friendlyError(code: string | undefined, status: number) {
  const inputChecks: Record<string, string> = {
    VIDEO_TOO_SHORT: 'This video is too short. Choose a complete clip with at least 8 frames.',
    VIDEO_TOO_LONG:
      'This video exceeds 900 frames or 45 seconds. Choose a shorter complete practice clip.',
    VIDEO_NOT_DECODABLE:
      'This video could not be decoded, or its frame rate is unsupported. Export a playable MP4, MOV or WebM.',
    VIDEO_DIMENSIONS_UNSUPPORTED:
      'The resolution is unsupported. Use a fixed resolution no larger than 4K, with each side at least 128 pixels.',
    VIDEO_DIMENSIONS_CHANGED:
      'The frame dimensions change during this video. Export it at a fixed resolution.',
    VIDEO_DECODE_INCOMPLETE:
      'The full video could not be decoded. Export a complete copy and try again.',
    UPLOAD_CHECK_BUSY: 'Another upload is being checked. Please try again shortly.',
    UPLOAD_CHECK_TIMEOUT: 'Video checking timed out. Try a shorter clip.',
    UPLOAD_CHECK_UNAVAILABLE:
      'Video checking is temporarily unavailable. Your current practice is unchanged.',
  };
  if (code && inputChecks[code]) return inputChecks[code];
  if (status === 415)
    return 'This file could not be opened as a video. Choose an MP4, MOV or WebM file.';
  if (status === 413) return 'This video is too large. Choose a clip smaller than 50 MiB.';
  if (status === 409)
    return 'This request is no longer available, or another review is in progress. Reopen the practice and try again.';
  if (status === 404)
    return 'This saved item is no longer available. Open another practice or add your video again.';
  if (code?.startsWith('CONTEXT_'))
    return 'The source for this request could not be confirmed. Reopen the saved review and try again.';
  return 'We could not complete this request. Please check your connection and try again.';
}
async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch('/api' + path, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(method === 'POST' ? { 'Idempotency-Key': crypto.randomUUID() } : {}),
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  const value = await response.json();
  if (!response.ok)
    throw new ApiError(friendlyError(value.error?.error_code, response.status), response.status);
  return value as T;
}

export default function App() {
  const [workspace, setWorkspace] = useState<Workspace | null>(() => load('cpr.workspace'));
  const [runId, setRunId] = useState<string | null>(() => load('cpr.runId'));
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<Event[]>([]),
    [tools, setTools] = useState<Tool[]>([]);
  const [caps, setCaps] = useState<Capabilities | null>(null),
    [health, setHealth] = useState(false);
  const [preferences, setPreferences] = useState<{ label_ids: number[] }[]>([]);
  const [targets, setTargets] = useState<number[]>([4, 5]);
  const [reviewBudget, setReviewBudget] = useState(900);
  const [allowPartial, setAllowPartial] = useState(false);
  const [intent, setIntent] = useState('analyse');
  const [text, setText] = useState('');
  const [error, setError] = useState(''),
    [pending, setPending] = useState(false);
  const [boundContext, setBoundContext] = useState<BoundContext | null>(null);
  const [contextError, setContextError] = useState('');
  const [sourceConfirmed, setSourceConfirmed] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const practiceVideo = useRef<HTMLVideoElement>(null);
  const referenceVideo = useRef<HTMLVideoElement>(null);
  const [referenceResult, setReferenceResult] = useState<{
    run_id: string;
    session_id: string;
    source_report_id: string | null;
    data: ReferenceLookup;
  } | null>(null);
  const [selectedReference, setSelectedReference] = useState<string | null>(null);
  const [continuity, setContinuity] = useState<ContinuityState | null>(null);
  const [continuityError, setContinuityError] = useState('');
  const [continuityRefresh, setContinuityRefresh] = useState(0);
  const [startingFocusRevision, setStartingFocusRevision] = useState(0);
  const continuityHydrated = useRef('');
  const currentSession = useRef(workspace?.session_id);
  currentSession.current = workspace?.session_id;
  const clarificationScopeGuard = useRef(false);
  const restoredScope = useRef('');
  const supportDialog = useRef<HTMLDialogElement>(null);
  const sidebarElement = useRef<HTMLElement>(null);
  const assistantElement = useRef<HTMLElement>(null);
  const [mobileNav, setMobileNav] = useState(() => window.matchMedia('(max-width: 760px)').matches);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [focusOpen, setFocusOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [fileCheck, setFileCheck] = useState<{
    asset_id: string;
    duration_seconds: number;
    decoded_frames: number;
    dimensions_hw: number[];
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [lastMessage, setLastMessage] = useState('');
  const [savedReport, setSavedReport] = useState<{
    id: string;
    session_id: string;
    response: Response;
  } | null>(null);
  const real = caps?.mode === 'experimental';
  const llmEnabled = caps?.llm?.mode === 'deepseek';
  const busy = !!runId && (!run || !terminal.has(run.status));
  const sessionCompatible = workspace?.bundle_compatible !== false;
  const bound = caps?.context_binding === 'bound-v1';
  const activeRun = run?.session_id === workspace?.session_id ? run : null;
  const viewedReportId = activeRun?.response?.prediction
    ? activeRun.report_id
    : activeRun?.review_context?.source_report_id || workspace?.source_report_id;
  const contextReady =
    !bound ||
    (boundContext?.session_id === workspace?.session_id &&
      !contextError &&
      (!viewedReportId || boundContext?.source?.report_id === viewedReportId));
  const canReview = !!caps && health && !!workspace?.asset_id && sessionCompatible && contextReady;
  useEffect(() => {
    let disposed = false;
    setFileCheck(null);
    if (workspace?.asset_id) {
      const assetId = workspace.asset_id;
      api<{
        inspection?: {
          status: string;
          duration_seconds: number;
          decoded_frames: number;
          dimensions_hw: number[];
        };
      }>(`/assets/${encodeURIComponent(assetId)}`)
        .then((asset) => {
          if (!disposed && asset.inspection?.status === 'passed')
            setFileCheck({ asset_id: assetId, ...asset.inspection });
        })
        .catch(() => {});
    }
    return () => {
      disposed = true;
    };
  }, [workspace?.asset_id]);
  const scopeLocked = pending || (busy && run?.status !== 'waiting_user');
  const seen = new Set(events.filter((e) => e.kind === 'node.enter').map((e) => e.node_id));
  const current = [...events].reverse().find((e) => e.kind === 'node.enter')?.node_id;

  useEffect(() => {
    if (!caps || !activeRun?.response?.classification_claims.length || !activeRun.report_id) return;
    const key = `${activeRun.session_id}:${activeRun.id}`;
    if (restoredScope.current === key) return;
    restoredScope.current = key;
    setTargets(activeRun.response.classification_claims.map((c) => c.label_id));
  }, [activeRun?.report_id, caps, activeRun?.session_id]);

  useEffect(() => {
    const sid = workspace?.session_id;
    if (!bound || !sid) {
      setBoundContext(null);
      return;
    }
    let disposed = false;
    setBoundContext(null);
    setContextError('');
    const query = viewedReportId ? `?source_report_id=${encodeURIComponent(viewedReportId)}` : '';
    api<BoundContext>(`/sessions/${encodeURIComponent(sid)}/context${query}`)
      .then((value) => {
        if (!disposed) setBoundContext(value);
      })
      .catch((e) => {
        if (!disposed) setContextError(e.message);
      });
    return () => {
      disposed = true;
    };
  }, [bound, workspace?.session_id, viewedReportId, run?.status, startingFocusRevision]);

  useEffect(() => {
    const sid = workspace?.session_id;
    let disposed = false;
    setContinuity(null);
    setContinuityError('');
    if (sid && caps?.practice_continuity) {
      api<ContinuityState>(`/sessions/${encodeURIComponent(sid)}/continuity`)
        .then((value) => {
          if (disposed) return;
          setContinuity(value);
          if (!value.locked && continuityHydrated.current !== sid)
            setTargets(
              value.selection.status === 'available'
                ? value.selection.label_ids
                : caps.candidate_labels,
            );
          continuityHydrated.current = sid;
        })
        .catch(() => {
          if (!disposed)
            setContinuityError(
              'Earlier practice details are unavailable. You can still review this video.',
            );
        });
    }
    return () => {
      disposed = true;
    };
  }, [workspace?.session_id, caps, continuityRefresh, run?.status]);

  async function usePreviousFocus(
    reportId: string | null,
    revision: number,
    notebook?: { sessionId: string; revision: number },
  ) {
    const sid = workspace?.session_id;
    if (!sid) return;
    await api(`/sessions/${encodeURIComponent(sid)}/continuity`, 'POST', {
      source_report_id: reportId,
      revision,
      ...(notebook
        ? {
            source_notebook_session_id: notebook.sessionId,
            source_notebook_revision: notebook.revision,
          }
        : {}),
    });
    const value = await api<ContinuityState>(`/sessions/${encodeURIComponent(sid)}/continuity`);
    if (currentSession.current !== sid) return;
    setContinuity(value);
    setTargets(
      value.selection.status === 'available'
        ? value.selection.label_ids
        : caps?.candidate_labels || [],
    );
    setStartingFocusRevision((v) => v + 1);
  }

  useEffect(() => {
    if (!viewedReportId) return;
    setWorkspace((ws) =>
      ws && ws.source_report_id !== viewedReportId
        ? { ...ws, source_report_id: viewedReportId }
        : ws,
    );
  }, [viewedReportId]);

  useEffect(() => {
    setSourceConfirmed(false);
    if (run?.wait?.scope_confirmation_required) {
      clarificationScopeGuard.current = true;
      setTargets([]);
    }
  }, [run?.wait?.wait_id]);

  useEffect(() => {
    api<Capabilities>('/capabilities')
      .then((value) => {
        setCaps(value);
        if (value.mode === 'experimental' && !clarificationScopeGuard.current)
          setTargets(value.candidate_labels);
      })
      .catch((e) => setError(e.message));
    api<{ status: string }>('/health')
      .then((x) => setHealth(x.status === 'ok'))
      .catch(() => setHealth(false));
    api<{ label_ids: number[] }[]>('/preferences')
      .then(setPreferences)
      .catch(() => {});
  }, []);
  useEffect(() => {
    localStorage.setItem('cpr.workspace', JSON.stringify(workspace));
  }, [workspace]);
  useEffect(() => {
    localStorage.setItem('cpr.runId', JSON.stringify(runId));
    if (!runId) {
      setRun(null);
      setEvents([]);
      setTools([]);
      return;
    }
    let disposed = false,
      timer: number;
    async function poll() {
      try {
        const [next, log, executions] = await Promise.all([
          api<Run>(`/runs/${runId}`),
          api<{ events: Event[] }>(`/runs/${runId}/events`),
          api<Tool[]>(`/runs/${runId}/tools`),
        ]);
        if (disposed) return;
        setRun(next);
        setEvents(log.events);
        setTools(executions);
        if (terminal.has(next.status)) {
          api<{ label_ids: number[] }[]>('/preferences')
            .then((x) => {
              if (!disposed) setPreferences(x);
            })
            .catch(() => {});
          return;
        }
      } catch (e) {
        if (!disposed) {
          setError((e as Error).message);
          if (e instanceof ApiError && e.status === 404) {
            setRunId(null);
            return;
          }
        }
      }
      if (!disposed) timer = window.setTimeout(poll, 400);
    }
    void poll();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
    };
  }, [runId]);

  async function perform(action: () => Promise<void>) {
    setPending(true);
    setError('');
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  async function demo() {
    const value = await api<{
      session: { id: string };
      asset: { id: string; name: string };
      clip: { id: string };
    }>('/demo/session', 'POST');
    const ws = {
      session_id: value.session.id,
      asset_id: value.asset.id,
      clip_id: value.clip.id,
      name: value.asset.name,
    };
    setWorkspace(ws);
    setRunId(null);
    setRun(null);
    setEvents([]);
    setTools([]);
    return ws;
  }
  async function send(goal?: Goal, message = '', reference?: { scope: string; source: string }) {
    if (real && !workspace?.asset_id)
      throw new Error('Upload a training video or open a saved review first.');
    const ws: Workspace = workspace || (await demo());
    if (ws.bundle_compatible === false)
      throw new Error('Create a new session for the current model bundle.');
    const value = await api<{ run_id: string }>(`/sessions/${ws.session_id}/turns`, 'POST', {
      message,
      ...(goal ? { goal } : {}),
      ...(bound
        ? {
            context: {
              source_report_id: viewedReportId || boundContext?.source?.report_id || null,
              reference: reference || {
                scope: 'explicit',
                source: viewedReportId ? 'viewed_report' : 'current_clip',
              },
            },
          }
        : {}),
    });
    setRun(null);
    setEvents([]);
    setTools([]);
    setRunId(value.run_id);
  }
  async function upload(file: File) {
    if (file.size > 50 * 1024 * 1024) throw new Error('Choose a video smaller than 50 MiB.');
    const form = new FormData();
    form.append('file', file);
    const response = await fetch('/api/assets', { method: 'POST', body: form });
    const asset = await response.json();
    if (!response.ok) throw new Error(friendlyError(asset.error?.error_code, response.status));
    const clip = await api<{ id: string }>('/clips', 'POST', { asset_id: asset.id });
    const session = await api<{ id: string }>('/sessions', 'POST', { clip_id: clip.id });
    setTargets(caps?.candidate_labels || []);
    setWorkspace({
      session_id: session.id,
      clip_id: clip.id,
      asset_id: asset.id,
      name: asset.name,
    });
    setRunId(null);
    setRun(null);
    setEvents([]);
    setTools([]);
  }
  async function continueForm() {
    if (!run?.wait?.interrupt_id) throw new Error('Waiting for checkpoint registration');
    await api(`/runs/${run.id}/resume`, 'POST', {
      kind: 'user_answer',
      wait_id: run.wait.wait_id,
      interrupt_id: run.wait.interrupt_id,
      revision: run.revision,
      goal: { intent, target_label_ids: targets, allow_partial: allowPartial, analysis_budget_seconds: reviewBudget },
      ...(run.wait.context_snapshot_id
        ? {
            context_confirmation: {
              snapshot_id: run.wait.context_snapshot_id,
              use_bound_source: true,
            },
          }
        : {}),
    });
  }
  const nodeList = Object.entries(caps?.nodes || {});
  const displayedReportId =
    (activeRun?.response?.prediction ? activeRun.report_id : viewedReportId) || null;
  useEffect(() => {
    let disposed = false;
    setSavedReport(null);
    if (displayedReportId)
      api<{ id: string; session_id: string; response: Response }>(
        `/reports/${encodeURIComponent(displayedReportId)}`,
      )
        .then((value) => {
          if (!disposed) setSavedReport(value);
        })
        .catch(() => {
          if (!disposed) setSavedReport(null);
        });
    return () => {
      disposed = true;
    };
  }, [displayedReportId, workspace?.session_id]);
  useEffect(() => {
    if (activeRun?.wait?.wait_id) setFocusOpen(false);
  }, [activeRun?.wait?.wait_id]);
  useEffect(() => {
    const query = window.matchMedia('(max-width: 760px)');
    const change = () => {
      setMobileNav(query.matches);
      if (!query.matches) setSidebarOpen(false);
    };
    query.addEventListener('change', change);
    return () => query.removeEventListener('change', change);
  }, []);
  useEffect(() => {
    if (!sidebarOpen || !mobileNav) return;
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const focusables = () =>
      Array.from(
        sidebarElement.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), summary, a[href]',
        ) || [],
      ).filter((el) => el.getClientRects().length > 0);
    focusables()[0]?.focus();
    const close = (event: KeyboardEvent) => {
      if (supportDialog.current?.open) return;
      if (event.key === 'Escape') setSidebarOpen(false);
      if (event.key === 'Tab') {
        const elements = focusables(),
          first = elements[0],
          last = elements.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener('keydown', close);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener('keydown', close);
      previous?.focus();
    };
  }, [sidebarOpen, mobileNav]);
  const referenceRunId =
    activeRun?.reference_requested &&
    ['completed', 'partially_completed'].includes(activeRun.status)
      ? activeRun.id
      : null;
  useEffect(() => {
    let disposed = false;
    setReferenceResult(null);
    setSelectedReference(null);
    if (referenceRunId && workspace) {
      const sessionId = workspace.session_id;
      const sourceId = viewedReportId || null;
      api<ReferenceLookup>(`/runs/${referenceRunId}/references`)
        .then((data) => {
          if (!disposed) {
            setReferenceResult({
              run_id: referenceRunId,
              session_id: sessionId,
              source_report_id: sourceId,
              data,
            });
            setSelectedReference(
              data.status === 'available' ? data.default_entry_id || null : null,
            );
          }
        })
        .catch(() => {
          if (!disposed)
            setReferenceResult({
              run_id: referenceRunId,
              session_id: sessionId,
              source_report_id: sourceId,
              data: { status: 'unavailable', items: [] },
            });
        });
    }
    return () => {
      disposed = true;
    };
  }, [referenceRunId, workspace?.session_id, viewedReportId]);
  const validReference =
    referenceResult?.session_id === workspace?.session_id &&
    referenceResult?.source_report_id === (viewedReportId || null)
      ? referenceResult
      : null;
  const comparing =
    !!validReference &&
    validReference.data.status === 'available' &&
    !!selectedReference &&
    validReference.data.items.some((e) => e.id === selectedReference);
  function closeComparison() {
    referenceVideo.current?.pause();
    setSelectedReference(null);
  }
  function openComparison() {
    if (validReference?.data.status !== 'available') return;
    setSelectedReference(validReference.data.default_entry_id || null);
    window.setTimeout(
      () => practiceVideo.current?.scrollIntoView({ behavior: 'smooth', block: 'center' }),
      0,
    );
  }
  function playBoth() {
    const players = [practiceVideo.current, referenceVideo.current].filter(
      (p): p is HTMLVideoElement => !!p,
    );
    players.forEach((p) => {
      p.currentTime = 0;
    });
    void Promise.all(players.map((p) => p.play())).catch(() =>
      setError(
        'Use the Play controls on each clip if your browser prevents simultaneous playback.',
      ),
    );
  }
  const sourceMatches =
    savedReport?.id === displayedReportId &&
    savedReport?.session_id === workspace?.session_id &&
    savedReport?.response.prediction?.clip_id === workspace?.clip_id;
  const displayResponse = ['failed', 'rejected'].includes(activeRun?.status || '')
    ? null
    : activeRun?.response?.prediction
      ? activeRun.response
      : sourceMatches
        ? savedReport!.response
        : null;
  const hasFindings = !!displayResponse?.prediction;
  const sourceAvailable = !bound || boundContext?.source_status === 'available';
  const canFollowUp = canReview && hasFindings && sourceAvailable && !pending && !busy;
  const flags =
    displayResponse?.classification_claims.filter((c) => c.state === 'model_flagged') || [];
  const plannedTools = (() => {
    const plan = [...events].reverse().find((e) => e.kind === 'policy.decision' && e.payload.profile_id);
    const audit = plan?.payload.audit as { selected?: { missing_tools?: unknown } } | undefined;
    const ids = audit?.selected?.missing_tools;
    const remaining = Array.isArray(ids) && ids.every((id) => typeof id === 'string') ? ids as string[] : caps?.tools || [];
    const visible = new Set([...tools.map(t => t.request.tool_id), ...remaining]);
    return (caps?.tools || []).filter(id => visible.has(id));
  })();
  const focusDescription =
    targets.length === caps?.candidate_labels.length
      ? 'All categories'
      : targets.length === 1
        ? caps?.label_names[targets[0]] || 'One category'
        : `${targets.length} categories`;
  function openReview(saved: Workspace, id: string | null) {
    setWorkspace(saved);
    setRun(null);
    setEvents([]);
    setTools([]);
    setError('');
    setLastMessage('');
    setRunId(id);
    setSidebarOpen(false);
    setFocusOpen(false);
  }
  function home() {
    setWorkspace(null);
    setRunId(null);
    setRun(null);
    setEvents([]);
    setTools([]);
    setError('');
    setLastMessage('');
    setFocusOpen(false);
    setSidebarOpen(false);
  }
  async function addFile(file: File) {
    setUploading(true);
    try {
      await upload(file);
      setLastMessage('');
      setFocusOpen(false);
      setSidebarOpen(false);
    } finally {
      setUploading(false);
    }
  }
  async function ask(goal?: Goal, message = '') {
    setLastMessage(
      message ||
        (goal?.intent === 'summarise'
          ? 'Make a recap of this review.'
          : goal?.intent === 'show_example'
            ? 'Show research references for my selected focus.'
            : goal?.intent === 'explain'
              ? 'Help me understand the selected findings.'
              : goal?.intent === 'manage_memory'
                ? 'Update my saved focus.'
                : 'Review this practice video.'),
    );
    await send(goal, message);
    if (message) setText('');
    if (window.matchMedia('(max-width: 1050px)').matches)
      assistantElement.current?.scrollIntoView({
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
        block: 'start',
      });
    setFocusOpen(false);
  }
  function focusPicker() {
    return (
      <FocusPicker
        labels={caps?.candidate_labels || []}
        names={caps?.label_names || {}}
        selected={targets}
        disabled={scopeLocked}
        onChange={setTargets}
      />
    );
  }
  function replayVideo() {
    const player = practiceVideo.current;
    if (!player) return;
    player.currentTime = 0;
    player.scrollIntoView({
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
      block: 'center',
    });
    void player
      .play()
      .catch(() => setError('Select the Play button on your video to start playback.'));
  }

  function continuityPanel() {
    if (!continuity || continuity.session_id !== workspace?.session_id) return null;
    return (
      <PracticeContinuityPanel
        key={`continuity:${workspace.session_id}`}
        data={continuity}
        names={caps?.label_names || {}}
        disabled={pending || busy || !canReview || hasFindings}
        onUse={(id, revision) => void perform(() => usePreviousFocus(id, revision))}
        onUseNotebook={(sid, sourceRevision, revision) =>
          void perform(() =>
            usePreviousFocus(null, revision, { sessionId: sid, revision: sourceRevision }),
          )
        }
      />
    );
  }
  const failed = activeRun?.status === 'failed' || activeRun?.status === 'rejected';
  const reply =
    activeRun?.status === 'waiting_user' ? (
      <div className="confirmation-card" role="region" aria-label="Confirm your review">
        <span className="eyebrow">A QUICK CHECK</span>
        <h3>What would you like to review?</h3>
        <p>
          {activeRun.wait?.scope_confirmation_required
            ? 'More than one finding could match. Choose the categories you mean below.'
            : activeRun.wait?.context_code === 'CONTEXT_SOURCE_AMBIGUOUS'
              ? 'This conversation belongs to the video shown above. Confirm it, or open another practice from your history.'
              : 'Choose a task and confirm the practice you want to use.'}
        </p>
        <label className="field-label" htmlFor="intent">
          I would like to
        </label>
        <select id="intent" value={intent} onChange={(e) => setIntent(e.target.value)}>
          <option value="analyse">Review the video</option>
          <option value="explain">Understand saved findings</option>
          <option value="summarise">Create a recap</option>
          {!!caps?.reference_tools?.length && (
            <option value="show_example">Find research references</option>
          )}
        </select>
        {focusPicker()}
        {activeRun.wait?.source_confirmation_required && (
          <label className="source-confirmation">
            <input
              type="checkbox"
              checked={sourceConfirmed}
              onChange={(e) => setSourceConfirmed(e.target.checked)}
            />
            <span>
              Use <strong>{activeRun.wait.bound_video_name || 'the current practice'}</strong>
              {activeRun.wait.bound_source_report_id
                ? ' and its saved review.'
                : ' for this review.'}
            </span>
          </label>
        )}
        <button
          className="primary full-width"
          disabled={
            pending ||
            !!activeRun.resume_payload ||
            !activeRun.wait?.interrupt_id ||
            (!!activeRun.wait?.source_confirmation_required && !sourceConfirmed) ||
            (!!activeRun.wait?.scope_confirmation_required &&
              intent !== 'summarise' &&
              !targets.length) ||
            (intent === 'analyse' && !targets.length)
          }
          onClick={() => void perform(continueForm)}
        >
          Continue review <Icon name="arrow" size={16} />
        </button>
        <button
          className="quiet full-width"
          disabled={pending}
          onClick={() =>
            void perform(async () => {
              await api(`/runs/${activeRun.id}/cancel`, 'POST');
            })
          }
        >
          Cancel request
        </button>
      </div>
    ) : busy ? (
      <div className="assistant-thinking" role="status">
        <span className="typing-dots">
          <i />
          <i />
          <i />
        </span>
        <p>
          {tools.length
            ? 'Your video is being reviewed. I’ll help you explore the findings when it’s ready.'
            : 'Preparing your review…'}
        </p>
      </div>
    ) : activeRun?.reference_requested &&
      ['completed', 'partially_completed'].includes(activeRun.status) ? (
      <ReferenceAnswer
        data={validReference?.data || null}
        names={caps?.label_names || {}}
        onOpen={openComparison}
      />
    ) : activeRun?.response?.prediction ? (
      <GroundedReview
        key={activeRun.report_id}
        reportId={activeRun.report_id}
        onReplay={workspace?.asset_id ? replayVideo : undefined}
      />
    ) : activeRun?.status === 'cancelled' ? (
      <div className="assistant-note">
        <Icon name="info" />
        <p>This request was cancelled. Your saved reviews are still available.</p>
      </div>
    ) : failed ? (
      <div className="assistant-note">
        <Icon name="info" />
        <p>
          This request couldn't be completed. You can reopen a saved review or add another video.
        </p>
      </div>
    ) : activeRun?.response?.memory_receipts?.length ? (
      <div className="assistant-answer">
        <div className="answer-author">
          <Icon name="bookmark" size={16} />
          <strong>Focus updated</strong>
        </div>
        <p>
          Your explicit preference has been updated. Changing a saved focus does not change the
          findings in an existing report.
        </p>
      </div>
    ) : activeRun?.response ? (
      <div className="assistant-note">
        <Icon name="info" />
        <p>There are no saved findings for this request yet. Start a video review to continue.</p>
      </div>
    ) : null;

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button
          className="sidebar-scrim"
          aria-label="Close sessions"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        ref={sidebarElement}
        className={`app-sidebar ${sidebarOpen ? 'is-open' : ''}`}
        aria-label="Practice navigation"
        inert={mobileNav && !sidebarOpen}
      >
        <button
          className="icon-button mobile-sidebar-close"
          aria-label="Close sessions"
          onClick={() => setSidebarOpen(false)}
        >
          <Icon name="close" />
        </button>
        <button
          className="brand"
          onClick={home}
          disabled={pending || busy}
          aria-label="CPR Review home"
        >
          <span className="brand-mark">
            <svg viewBox="0 0 32 32" aria-hidden="true">
              <path
                d="M4 17h6l3-9 5 17 3-8h7"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <span>
            CPR<span className="brand-subtitle">practice review</span>
          </span>
        </button>
        <button
          className="primary new-review"
          disabled={pending || busy || !caps || !health}
          onClick={() => fileInput.current?.click()}
        >
          <Icon name="plus" size={18} /> New practice
        </button>
        <HistoryPanel
          workspace={workspace}
          runId={runId}
          refreshKey={`${runId || ''}:${activeRun?.status || ''}`}
          disabled={pending || busy}
          names={caps?.label_names || {}}
          onOpen={openReview}
        />
        <div className="sidebar-bottom">
          <div className="practice-space">
            <span className="space-avatar">P</span>
            <div>
              <strong>Your practice space</strong>
              <small>One review at a time.</small>
            </div>
          </div>
          <button
            className="sidebar-text-button technical-trigger"
            onClick={() => supportDialog.current?.showModal()}
          >
            <Icon name="settings" size={15} /> Technical details
          </button>
        </div>
      </aside>
      <div className="main-shell" inert={mobileNav && sidebarOpen}>
        <header className="workspace-header">
          <div>
            <button
              className="icon-button mobile-menu"
              aria-label="Open sessions"
              aria-expanded={sidebarOpen}
              onClick={() => setSidebarOpen((v) => !v)}
            >
              <Icon name="menu" />
            </button>
            <span className="header-section">My practice</span>
            <Icon name="chevron" size={13} />
            <span>{workspace ? 'Review' : 'Get started'}</span>
          </div>
          <span className={`service-status ${health ? 'online' : ''}`}>
            <i />
            {health ? 'Connected' : 'Connecting…'}
          </span>
        </header>
        <main className="product-main">
          <div className="page-heading">
            <div>
              <span className="eyebrow">CPR TRAINING · VIDEO REVIEW</span>
              <h1>{workspace ? 'Review your practice.' : 'Start your practice review.'}</h1>
              <p>
                {workspace
                  ? 'Look back at your video, understand the findings, and keep a useful recap.'
                  : 'Add a short CPR training video. We’ll help you review it, one finding at a time.'}
              </p>
            </div>
            {hasFindings && displayedReportId && (
              <a
                className="button secondary export-review"
                href={`/api/reports/${displayedReportId}/export`}
                target="_blank"
                rel="noreferrer"
              >
                <Icon name="download" size={17} /> Export review
              </a>
            )}
          </div>
          <div className="journey" aria-label="Your review steps">
            <span className={workspace ? 'done' : 'active'}>
              <b>{workspace ? <Icon name="check" size={12} /> : '1'}</b>Add a video
            </span>
            <i />
            <span className={hasFindings ? 'done' : workspace ? 'active' : ''}>
              <b>{hasFindings ? <Icon name="check" size={12} /> : '2'}</b>Explore the findings
            </span>
            <i />
            <span
              className={
                activeRun?.response?.title === 'Review summary'
                  ? 'done'
                  : hasFindings
                    ? 'active'
                    : ''
              }
            >
              <b>
                {activeRun?.response?.title === 'Review summary' ? (
                  <Icon name="check" size={12} />
                ) : (
                  '3'
                )}
              </b>
              Keep a recap
            </span>
          </div>
          {error && (
            <div className="alert" role="alert">
              <Icon name="info" size={19} />
              <span>{error}</span>
              <button
                className="icon-button"
                onClick={() => setError('')}
                aria-label="Dismiss error"
              >
                <Icon name="close" size={16} />
              </button>
            </div>
          )}
          {!real && caps && (
            <p className="training-notice">
              Preview workspace: CPR video analysis is not connected in this environment.
            </p>
          )}
          <nav className="mobile-section-nav" aria-label="Review sections">
            <a href="#practice-review">
              <Icon name="play" size={14} /> Video &amp; findings
            </a>
            <a href="#review-assistant">
              <Icon name="chat" size={14} /> Assistant
            </a>
          </nav>
          <div className="review-layout">
            <div className="practice-column" id="practice-review">
              <section className="surface video-card" aria-label="Practice video">
                <div className="card-heading">
                  <div>
                    <span className="card-step">YOUR VIDEO</span>
                    <h2>{workspace ? workspace.name : 'Start with your practice'}</h2>
                  </div>
                  {workspace && (
                    <button
                      className="quiet replace-video"
                      disabled={pending || busy}
                      onClick={() => fileInput.current?.click()}
                    >
                      <Icon name="upload" size={15} /> Change video
                    </button>
                  )}
                </div>
                {workspace?.asset_id ? (
                  <div className={`video-comparison-grid ${comparing ? 'is-comparing' : ''}`}>
                    <div className="practice-video-pane">
                      {comparing && (
                        <div className="comparison-pane-heading">
                          <span>
                            <Icon name="play" size={14} /> Your practice
                          </span>
                        </div>
                      )}
                      <div className="video-frame">
                        <video
                          ref={practiceVideo}
                          key={workspace.asset_id}
                          controls
                          preload="metadata"
                          src={`/api/assets/${workspace.asset_id}/media`}
                          aria-label="Practice video player"
                        />
                      </div>
                      {comparing && (
                        <p className="reference-label">
                          Your saved findings stay attached to this video.
                        </p>
                      )}
                    </div>
                    {comparing && validReference && selectedReference && (
                      <ReferenceComparison
                        data={validReference.data}
                        runId={validReference.run_id}
                        selectedId={selectedReference}
                        onSelect={(id) => {
                          referenceVideo.current?.pause();
                          setSelectedReference(id);
                        }}
                        onClose={closeComparison}
                        onPlayBoth={playBoth}
                        videoRef={referenceVideo}
                      />
                    )}
                  </div>
                ) : (
                  <div
                    className={`upload-zone ${dragging ? 'is-dragging' : ''}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      if (!pending && !busy && caps && health) setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragging(false);
                      if (pending || busy || !caps || !health) return;
                      const files = e.dataTransfer.files;
                      if (files.length !== 1) {
                        setError('Please add one video at a time.');
                        return;
                      }
                      void perform(() => addFile(files[0]));
                    }}
                  >
                    <span className="upload-symbol">
                      <Icon name="upload" size={29} />
                    </span>
                    <h3>{uploading ? 'Adding your practice…' : 'Add your practice video'}</h3>
                    <p>
                      Drop a practice video here,
                      <br />
                      or choose one from your device.
                    </p>
                    <button
                      className="primary"
                      disabled={pending || busy || !caps || !health}
                      onClick={() => fileInput.current?.click()}
                    >
                      <Icon name="plus" size={17} />
                      {uploading
                        ? caps?.upload_preflight
                          ? 'Uploading & checking…'
                          : 'Adding video…'
                        : 'Choose video'}
                    </button>
                    <span className="upload-formats">MP4, MOV or WebM · up to 50 MiB</span>
                  </div>
                )}
                <input
                  ref={fileInput}
                  type="file"
                  accept=".mp4,.mov,.webm"
                  hidden
                  aria-label="Upload practice video"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void perform(() => addFile(file));
                    e.target.value = '';
                  }}
                />
                <div className="video-caption">
                  <span>
                    <Icon name="play" size={14} />
                    {displayResponse?.prediction
                      ? `${(displayResponse.prediction.decoded_frames / displayResponse.prediction.reported_fps).toFixed(0)} s · Whole clip`
                      : 'One complete practice clip'}
                  </span>
                  <span>Single camera</span>
                </div>
                {fileCheck && fileCheck.asset_id === workspace?.asset_id && (
                  <p className="file-check-status" role="status">
                    File check passed · {fileCheck.decoded_frames} frames ·{' '}
                    {fileCheck.duration_seconds.toFixed(1)} s · {fileCheck.dimensions_hw[1]} ×{' '}
                    {fileCheck.dimensions_hw[0]}. File compatibility only.
                  </p>
                )}
                <div className="video-controls">
                  <button
                    className={`focus-toggle ${focusOpen ? 'selected' : ''}`}
                    aria-expanded={focusOpen}
                    disabled={scopeLocked || !caps}
                    onClick={() => setFocusOpen((v) => !v)}
                  >
                    <Icon name="settings" size={16} />
                    <span>Focus: {focusDescription}</span>
                    <Icon name="chevron" size={14} />
                  </button>
                  <button
                    className="primary start-review"
                    disabled={pending || busy || !canReview || !targets.length}
                    onClick={() =>
                      void perform(() =>
                        ask({
                          allow_partial: allowPartial,
                          analysis_budget_seconds: reviewBudget,
                          intent: hasFindings && sourceAvailable ? 'explain' : 'analyse',
                          target_label_ids: targets,
                          analysis_mode:
                            hasFindings && sourceAvailable ? 'existing_only' : 'adaptive',
                        }),
                      )
                    }
                  >
                    {busy
                      ? 'Review in progress'
                      : hasFindings
                        ? 'Show selected focus'
                        : 'Start review'}
                    <Icon name="arrow" size={17} />
                  </button>
                </div>
                <details className="review-options">
                  <summary>Review options</summary>
                  <div className="review-options-body">
                    <label htmlFor="review-time-limit">Time limit</label>
                    <select id="review-time-limit" value={reviewBudget} disabled={pending || busy} onChange={e => setReviewBudget(Number(e.target.value))}>
                      <option value={900}>15 minutes · complete check preferred</option>
                      <option value={300}>5 minutes</option>
                      <option value={180}>3 minutes</option>
                    </select>
                    <label className="partial-consent"><input type="checkbox" checked={allowPartial} disabled={pending || busy} onChange={e => setAllowPartial(e.target.checked)}/><span>Allow a partial review if the complete check cannot fit.</span></label>
                    <p>Unchecked categories will be listed clearly. You can expand a saved partial review later. Time estimates are conservative; completion is not guaranteed.</p>
                  </div>
                </details>
                {focusOpen && (
                  <div className="focus-drawer">
                    {focusPicker()}
                    <div className="saved-focus">
                      <span>
                        <Icon name="bookmark" size={15} />
                        {preferences.length
                          ? 'Saved focus: ' +
                            preferences[0].label_ids
                              .map((i) => caps?.label_names[i] || 'Category')
                              .join(', ')
                          : 'Save a focus to use in a later practice.'}
                      </span>
                      <div>
                        {!!preferences.length && (
                          <>
                            <button
                              className="quiet"
                              disabled={scopeLocked}
                              onClick={() =>
                                setTargets(
                                  preferences[0].label_ids.filter((i) =>
                                    caps?.candidate_labels.includes(i),
                                  ),
                                )
                              }
                            >
                              Use saved focus
                            </button>
                            <button
                              className="quiet"
                              disabled={pending || busy || !canReview}
                              onClick={() =>
                                void perform(() =>
                                  ask({
                                    intent: 'manage_memory',
                                    memory_commands: [{ operation: 'forget_preference' }],
                                  }),
                                )
                              }
                            >
                              Forget focus
                            </button>
                          </>
                        )}
                        <button
                          className="quiet"
                          disabled={pending || busy || !canReview || !targets.length}
                          onClick={() =>
                            void perform(() =>
                              ask({
                                intent: 'manage_memory',
                                memory_commands: [
                                  { operation: 'remember_focus', label_ids: targets },
                                ],
                              }),
                            )
                          }
                        >
                          Save focus
                        </button>
                      </div>
                    </div>
                  </div>
                )}
                <details className="video-requirements">
                  <summary>
                    <Icon name="info" size={13} /> Tips for a useful review
                  </summary>
                  <p>
                    Use one complete, fixed-camera CPR training clip: 8–900 frames, up to 45
                    seconds, no larger than 3840 × 2160 pixels. Keep the person and practice area
                    visible. The review covers the whole clip; it does not mark exact error times.
                  </p>
                </details>
              </section>
              {!hasFindings && continuityPanel()}
              {continuityError && (
                <div className="continuity-unavailable" role="status">
                  {continuityError}{' '}
                  <button className="quiet" onClick={() => setContinuityRefresh((v) => v + 1)}>
                    Try again
                  </button>
                </div>
              )}
              {busy && tools.length > 0 && (
                <AnalysisProgress
                  status={activeRun?.status || 'queued'}
                  tools={tools}
                  toolIds={plannedTools}
                  reused={false}
                />
              )}
              {failed ? (
                <section className="surface review-failure" role="status">
                  <span className="failure-symbol">
                    <Icon name="info" size={25} />
                  </span>
                  <h2>We couldn't complete this review.</h2>
                  <p>
                    {activeRun?.error_code === 'M4_BUDGET_INSUFFICIENT'
                      ? 'The selected checks could not fit the time limit. Increase the limit, or allow a partial review in Review options. Your video is still available.'
                      : activeRun?.error_code?.startsWith('CONTEXT_')
                      ? 'The saved source is no longer available for this request. Reopen another saved review or add the video again.'
                      : 'The video or its analysis could not be processed. Try another complete practice clip, or reopen a saved review.'}
                  </p>
                  {activeRun?.error_code === 'M4_BUDGET_INSUFFICIENT' && <button className="primary" disabled={pending || busy || !canReview} onClick={() => void perform(() => ask({intent:'analyse', target_label_ids:activeRun?.review_context?.target_label_ids || targets, analysis_mode:'adaptive', allow_partial:false, analysis_budget_seconds:900}))}>Retry with more time</button>}
                  <button
                    className="secondary"
                    onClick={() => fileInput.current?.click()}
                    disabled={pending || busy}
                  >
                    <Icon name="upload" size={15} /> Choose another video
                  </button>
                </section>
              ) : hasFindings && displayResponse ? (
                <section className="surface findings-card" aria-label="Practice findings">
                  <div className="card-heading">
                    <div>
                      <span className="card-step">YOUR REVIEW</span>
                      <h2>What to look at</h2>
                    </div>
                    <span className="pill neutral">Saved review</span>
                  </div>
                  <div className="review-overview">
                    <span className="review-count">{flags.length.toString().padStart(2, '0')}</span>
                    <div>
                      <strong>
                        {flags.length === 1
                          ? 'category flagged for review'
                          : 'categories flagged for review'}
                      </strong>
                      <p>
                        Across {displayResponse.classification_claims.length} displayed{' '}
                        {displayResponse.classification_claims.length === 1
                          ? 'category'
                          : 'categories'}
                        . Explore a finding with your assistant.
                      </p>
                    </div>
                  </div>
                  {displayResponse.visual_processing && <p className="visual-evidence-status" role="status">Visual evidence checked · {displayResponse.visual_processing.sampled_frames} sampled frames. Findings apply to the whole clip.</p>}
                  {displayResponse.scheduling?.partial_scope_authorised && <div className="partial-result-note" role="status"><strong>Partial review · {Object.values(displayResponse.target_states).filter(s => s !== 'not_evaluated').length} of {Object.keys(displayResponse.target_states).length} requested categories checked</strong><p>You allowed a partial review because the complete route could not fit the planning budget. Not checked: {Object.entries(displayResponse.target_states).filter(([,s]) => s === 'not_evaluated').map(([k]) => caps?.label_names[Number(k)] || k).join(', ')}. These are not negative findings.</p></div>}
                  {displayResponse.prediction?.analysis_route === 'body-qualified-v1'  && <div className="scope-route-note" role="region" aria-label="Scope of this analysis"><strong>Focused body review</strong><p>Checked: single hand, bent arms, standing and slow-frequency categories. RGB-based checks and other categories were not run. This does not establish correct CPR.</p><button className="quiet" disabled={!canFollowUp} onClick={() => void perform(() => ask({intent: 'analyse', target_label_ids: caps?.candidate_labels || [], analysis_mode: 'adaptive'}))}>Expand to all categories <Icon name="arrow" size={14}/></button></div>}
                  {Object.values(displayResponse.target_states).includes('not_evaluated') && <p className="scope-not-checked" role="status">Some selected categories have not been analysed. Expand the review to check them; they are not negative findings.</p>}
                  <EvidencePanel
                    key={`evidence:${displayedReportId}`}
                    reportId={displayedReportId}
                    claims={displayResponse.classification_claims}
                    assessment={displayResponse.module_assessment}
                    disabled={!canFollowUp}
                    onExplain={(id) => {
                      setTargets([id]);
                      void perform(() =>
                        ask({
                          intent: 'explain',
                          target_label_ids: [id],
                          analysis_mode: 'existing_only',
                        }),
                      );
                    }}
                  />
                  <div className="recap-invitation">
                    <span className="recap-icon">
                      <Icon name="bookmark" />
                    </span>
                    <div>
                      <strong>Save a recap for your next practice.</strong>
                      <p>Make a recap of the categories shown here.</p>
                    </div>
                    <button
                      className="secondary"
                      disabled={!canFollowUp}
                      onClick={() =>
                        void perform(() =>
                          ask({
                            intent: 'summarise',
                            target_label_ids: displayResponse.classification_claims.map(
                              (c) => c.label_id,
                            ),
                            analysis_mode: 'existing_only',
                          }),
                        )
                      }
                    >
                      Make a recap <Icon name="arrow" size={15} />
                    </button>
                  </div>
                  {caps?.review_workspace_tools && displayedReportId && (
                    <ReviewComparison
                      key={`record-comparison:${displayedReportId}`}
                      reportId={displayedReportId}
                    />
                  )}
                </section>
              ) : (
                !failed &&
                !busy && (
                  <section className="review-explainer">
                    <div className="mini-step">
                      <span>
                        <Icon name="play" size={19} />
                      </span>
                      <h3>See your practice</h3>
                      <p>A short video, captured from one camera.</p>
                    </div>
                    <div className="mini-step">
                      <span>
                        <Icon name="chat" size={19} />
                      </span>
                      <h3>Understand the findings</h3>
                      <p>Ask questions with the saved evidence in view.</p>
                    </div>
                    <div className="mini-step">
                      <span>
                        <Icon name="bookmark" size={19} />
                      </span>
                      <h3>Come back prepared</h3>
                      <p>Keep a recap and pick up where you left off.</p>
                    </div>
                  </section>
                )
              )}
              {hasFindings && continuityPanel()}
              {caps?.review_workspace_tools && workspace && (
                <NotebookPanel
                  key={`notebook:${workspace.session_id}`}
                  sessionId={workspace.session_id}
                  names={caps.label_names}
                  onSaved={() => setContinuityRefresh((v) => v + 1)}
                />
              )}
              <p className="training-note">
                <Icon name="info" size={14} /> For training reflection only. Model findings are
                suggestions to review, not proof of correct CPR.
              </p>
            </div>
            <aside
              ref={assistantElement}
              id="review-assistant"
              className="surface assistant-panel"
              aria-label="Practice review assistant"
            >
              <div className="assistant-heading">
                <span className="assistant-icon">
                  <Icon name="spark" size={21} />
                </span>
                <div>
                  <h2>Your review assistant</h2>
                  <p>
                    {llmEnabled
                      ? 'Ask, reflect, and keep a recap.'
                      : 'A guided way through your review.'}
                  </p>
                </div>
                <span
                  className="assistant-online"
                  title={llmEnabled ? 'AI-assisted review available' : 'Guided review available'}
                />
              </div>
              {bound && workspace && (
                <ContextPanel
                  context={boundContext}
                  labelNames={caps?.label_names || {}}
                  error={contextError}
                  activity={
                    activeRun?.status === 'waiting_user'
                      ? 'waiting'
                      : busy
                        ? 'reviewing'
                        : undefined
                  }
                />
              )}
              <ReviewConversation
                key={workspace?.session_id || 'empty'}
                sessionId={workspace?.session_id || null}
                runId={runId}
                status={activeRun?.status}
                currentReply={reply}
                disabled={pending || busy}
                onOpen={(id) => workspace && openReview(workspace, id)}
                names={caps?.label_names || {}}
                pendingMessage={lastMessage}
              />
              <div className="assistant-actions">
                {!!caps?.reference_tools?.length && (
                  <button
                    disabled={pending || busy || !canReview || (hasFindings && !sourceAvailable)}
                    onClick={() =>
                      void perform(() =>
                        ask({
                          intent: 'show_example',
                          target_label_ids: targets,
                          analysis_mode: 'existing_only',
                        }),
                      )
                    }
                  >
                    <Icon name="book" size={14} /> Compare references
                  </button>
                )}

                <button
                  disabled={!canFollowUp || !targets.length}
                  onClick={() =>
                    void perform(() =>
                      ask({
                        intent: 'explain',
                        target_label_ids: targets,
                        analysis_mode: 'existing_only',
                      }),
                    )
                  }
                >
                  <Icon name="chat" size={14} /> Explain my focus
                </button>
                <button
                  disabled={!canFollowUp}
                  onClick={() =>
                    void perform(() =>
                      ask({
                        intent: 'summarise',
                        target_label_ids:
                          displayResponse?.classification_claims.map((c) => c.label_id) || [],
                        analysis_mode: 'existing_only',
                      }),
                    )
                  }
                >
                  <Icon name="bookmark" size={14} /> Make a recap
                </button>
              </div>
              <form
                className="message-composer"
                aria-label="Ask the review assistant"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!pending && !busy && canReview && text.trim())
                    void perform(() => ask(undefined, text));
                }}
              >
                <label className="sr-only" htmlFor="request-text">
                  Your question
                </label>
                <textarea
                  id="request-text"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={
                    workspace ? 'Ask about this practice…' : 'Add a video to start a conversation…'
                  }
                  maxLength={2000}
                  disabled={pending || busy || !canReview}
                  rows={2}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                      e.preventDefault();
                      e.currentTarget.form?.requestSubmit();
                    }
                  }}
                />
                <div className="composer-bottom">
                  <span>{llmEnabled ? 'About this practice' : 'Guided review'}</span>
                  <button
                    className="send-message"
                    type="submit"
                    aria-label="Send message"
                    disabled={pending || busy || !canReview || !text.trim()}
                  >
                    <Icon name="arrow" size={19} />
                  </button>
                </div>
              </form>
              <p className="composer-note">
                {llmEnabled
                  ? 'Your video is not sent to the language model.'
                  : 'Send a request, then confirm your task in a form.'}
              </p>
              {busy && activeRun?.status !== 'waiting_user' && (
                <button
                  className="cancel-request quiet"
                  disabled={pending || !!activeRun?.cancel_requested}
                  onClick={() =>
                    activeRun &&
                    void perform(async () => {
                      await api(`/runs/${activeRun.id}/cancel`, 'POST');
                    })
                  }
                >
                  {activeRun?.cancel_requested ? 'Cancelling…' : 'Cancel request'}
                </button>
              )}
            </aside>
          </div>
          <footer className="product-footer">
            <span>
              CPR Review <i>·</i> Reflect on your practice.
            </span>
            <span>Your reviews stay in this workspace.</span>
          </footer>
        </main>
      </div>
      <dialog ref={supportDialog} className="support-dialog" aria-label="Technical details">
        <div className="dialog-heading">
          <div>
            <span className="eyebrow">READ-ONLY SERVICE INFORMATION</span>
            <h2>Technical details</h2>
          </div>
          <button
            className="icon-button"
            aria-label="Close technical details"
            onClick={() => supportDialog.current?.close()}
          >
            <Icon name="close" />
          </button>
        </div>
        <p className="muted">
          For troubleshooting and project review. These details do not add to the model’s findings.
        </p>
        <dl className="support-facts">
          <div>
            <dt>Language model</dt>
            <dd>
              {caps?.llm.mode === 'deepseek'
                ? `${caps.llm.model || 'DeepSeek'} · ${caps.llm.connection_verified ? 'Connection verified' : 'Not verified in this process'}`
                : 'Disabled · guided forms and saved-evidence templates'}
            </dd>
          </div>
          <div>
            <dt>Trace</dt>
            <dd>{activeRun?.trace_id || 'No active request'}</dd>
          </div>
          <div>
            <dt>Stopping reason</dt>
            <dd>{activeRun?.error_code || 'None'}</dd>
          </div>
        </dl>
        <details>
          <summary>State machine and execution</summary>
          <ol className="technical-nodes">
            {nodeList.map(([id, name]) => (
              <li key={id}>
                <code>{id}</code>
                <span>{name}</span>
                <small>
                  {seen.has(id) ? 'Visited' : 'Not visited'}
                  {busy && current === id ? ' · Current' : ''}
                </small>
              </li>
            ))}
          </ol>
          <pre>
            {JSON.stringify(
              {
                tools,
                context: activeRun?.review_context,
                assessment: activeRun?.response?.module_assessment,
                prediction: activeRun?.response?.prediction,
                recent_events: events.slice(-18),
              },
              null,
              2,
            )}
          </pre>
        </details>
        <details>
          <summary>Algorithm modules</summary>
          {['M1', 'M2', 'M3', 'M4', 'M5'].map((id) => (
            <p key={id}>
              <strong>{id}</strong> ·{' '}
              {caps?.module_status?.[id] ||
                caps?.module_slots?.[id]?.implementation_id ||
                'Not connected'}
              {caps?.module_slots?.[id]?.kind === 'baseline' ? ' · Member improvement pending' : ''}
            </p>
          ))}
        </details>
        <a href="/docs" target="_blank" rel="noreferrer">
          API documentation ↗
        </a>
        {!real && caps && (
          <button
            disabled={pending || busy}
            onClick={() =>
              void perform(async () => {
                await demo();
                supportDialog.current?.close();
              })
            }
          >
            Load workflow fixture
          </button>
        )}
      </dialog>
    </div>
  );
}
