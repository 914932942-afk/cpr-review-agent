"""Make actual analysis coverage and expanded review explicit in the product UI."""

from prepare_m4_workspace import TARGET


def patch(path, old, new):
    s = path.read_text(encoding="utf-8")
    if s.count(old) != 1:
        raise ValueError("Unexpected UI source " + str(path) + " " + old[:50])
    path.write_text(s.replace(old, new), encoding="utf-8")


def main():
    p = TARGET / "frontend/src/App.tsx"
    patch(
        p,
        """    evidence_scope: string;
""",
        """    evidence_scope: string;
    analysis_route?: 'full-v15' | 'body-qualified-v1';
    supported_label_ids?: number[];
""",
    )
    patch(
        p,
        """  const focusDescription =""",
        """  const plannedTools = (() => {
    const plan = [...events].reverse().find((e) => e.kind === 'policy.decision' && e.payload.profile_id);
    const audit = plan?.payload.audit as { selected?: { missing_tools?: unknown } } | undefined;
    const ids = audit?.selected?.missing_tools;
    return Array.isArray(ids) && ids.every((id) => typeof id === 'string') ? ids as string[] : caps?.tools || [];
  })();
  const focusDescription =""",
    )
    patch(
        p, """                  toolIds={caps?.tools || []}""", """                  toolIds={plannedTools}"""
    )
    patch(
        p,
        """                  <EvidencePanel
""",
        """                  {displayResponse.prediction?.analysis_route === 'body-qualified-v1' && <div className="scope-route-note" role="region" aria-label="Scope of this analysis"><strong>Focused body review</strong><p>Checked: single hand, bent arms, standing and slow-frequency categories. RGB-based checks and other categories were not run. This does not establish correct CPR.</p><button className="quiet" disabled={!canFollowUp} onClick={() => void perform(() => ask({intent: 'analyse', target_label_ids: caps?.candidate_labels || [], analysis_mode: 'adaptive'}))}>Expand to all categories <Icon name="arrow" size={14}/></button></div>}
                  {Object.values(displayResponse.target_states).includes('not_evaluated') && <p className="scope-not-checked" role="status">Some selected categories have not been analysed. Expand the review to check them; they are not negative findings.</p>}
                  <EvidencePanel
""",
    )
    with (TARGET / "frontend/src/styles.css").open("a", encoding="utf-8") as f:
        f.write("""
.scope-route-note { background: #edf5f2; border: 1px solid #d4e7dd; padding: 14px 16px; border-radius: 9px; margin-bottom: 16px; }
.scope-route-note > strong { font-size: 13px; color: #27644e; }
.scope-route-note > p, .scope-not-checked { font-size: 12px; line-height: 1.7; color: #5d786a; margin: 7px 0; }
.scope-route-note .quiet { margin-top: 5px; }
""")
    print("M4 actual-scope and expansion controls added")


if __name__ == "__main__":
    main()
