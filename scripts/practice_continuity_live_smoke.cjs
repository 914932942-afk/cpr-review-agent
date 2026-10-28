/* One optional paid goal parse before new-video analysis. Persistent state prevents retries. */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..'), baseURL = 'http://127.0.0.1:8007';
const out = path.join(root, 'test-results/practice-continuity-v1/live'), restore = process.argv.includes('--restore');
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const stateFile = path.join(out, 'run-state.json');
const state = fs.existsSync(stateFile) ? JSON.parse(fs.readFileSync(stateFile, 'utf8')) : {};
const save = () => fs.writeFileSync(stateFile, JSON.stringify(state, null, 2));
async function main() {
  assert(restore || process.argv.includes('--allow-live'), 'Explicit --allow-live required');
  fs.mkdirSync(out, { recursive: true });
  const acceptance = path.join(out, restore ? 'restore.json' : 'acceptance.json');
  assert(!fs.existsSync(acceptance), 'Preserve accepted evidence');
  const source = read('test-results/product-demo-v2/browser/acceptance.json');
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  const errors = [], external = [];
  try {
    const c = await browser.newContext({ baseURL, viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    c.on('page', p => p.on('pageerror', e => errors.push(e.message)));
    c.on('request', r => { if (new URL(r.url()).origin !== baseURL) external.push(r.url()); });
    const get = async u => { const r = await c.request.get('/api' + u); assert(r.ok(), u); return r.json(); };
    const p = await c.newPage(); await p.goto('/');
    if (restore) {
      assert(state.run_id);
      await p.locator(`[data-session-id="${state.workspace.session_id}"]`).click();
    } else {
      assert.equal((await get('/capabilities')).llm.mode, 'deepseek');
      if (!state.workspace) {
        await p.locator('input[type=file]').setInputFiles({ name: 'next-practice-focus.mp4', mimeType: 'video/mp4', buffer: fs.readFileSync(path.join(root, 'data/public_cpr/rgb-acceptance-v1/records/SupC05--r01--ch0/video.mp4')) });
        await p.waitForFunction(() => !!JSON.parse(localStorage.getItem('cpr.workspace'))?.asset_id);
        state.workspace = await p.evaluate(() => JSON.parse(localStorage.getItem('cpr.workspace'))); save();
      } else {
        await p.evaluate(({ ws, id }) => { localStorage.setItem('cpr.workspace', JSON.stringify(ws)); localStorage.setItem('cpr.runId', JSON.stringify(id)); }, { ws: state.workspace, id: state.run_id || null }); await p.reload();
      }
      const panel = p.getByRole('region', { name: 'Practice continuity', exact: true }); await panel.waitFor();
      if ((await get(`/sessions/${state.workspace.session_id}/continuity`)).selection.status !== 'available') {
        await panel.getByLabel('Earlier recap', { exact: true }).selectOption(source.summary_report_id);
        await panel.getByRole('button', { name: 'Use this focus', exact: true }).click();
      }
      await p.getByRole('button', { name: 'Focus: Bent arms', exact: true }).waitFor();
      if (!state.run_id) {
        await p.getByLabel('Your question', { exact: true }).fill('Show me research reference clips for the starting focus I selected for this practice. Do not analyse the video.');
        await p.getByRole('button', { name: 'Send message', exact: true }).click();
        await p.waitForFunction(() => !!JSON.parse(localStorage.getItem('cpr.runId')));
        state.run_id = await p.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId'))); save();
      }
    }
    let run;
    for (let i = 0; i < 150; i++) {
      run = await get('/runs/' + state.run_id);
      if (['completed', 'failed', 'cancelled', 'rejected', 'waiting_user'].includes(run.status)) break;
      await p.waitForTimeout(200);
    }
    assert.equal(run.status, 'completed', JSON.stringify({ status: run.status, error_code: run.error_code }));
    const turn = await get('/turns/' + run.turn_id), events = (await get('/runs/' + run.id + '/events')).events;
    assert.equal(turn.context_snapshot.source, null); assert.equal(turn.context_snapshot.clip_id, state.workspace.clip_id);
    assert.deepEqual(turn.context_snapshot.active_label_ids, [4]);
    assert.equal(turn.llm_resolution.code, 'LLM_GOAL_PARSED'); assert.equal(turn.llm_resolution.called, true);
    assert.equal(turn.resolved_goal.intent, 'show_example'); assert.deepEqual(turn.resolved_goal.target_label_ids, [4]);
    assert.equal(events.filter(e => e.payload?.called === true).length, 1);
    assert.equal(run.response.prediction, null); assert.deepEqual(run.response.classification_claims, []);
    assert.deepEqual(await get('/runs/' + run.id + '/tools'), []);
    assert.equal(await p.getByRole('region', { name: 'Practice findings', exact: true }).count(), 0);
    const context = await get(`/sessions/${state.workspace.session_id}/context`);
    assert.equal(context.source, null); assert.deepEqual(context.active_label_ids, [4]);
    const refs = await get('/runs/' + run.id + '/references'); assert.deepEqual(refs.items.map(i => i.label_id), [0, 4]);
    await p.getByRole('region', { name: 'Reference video comparison', exact: true }).waitFor();
    await p.getByRole('region', { name: 'Practice continuity', exact: true }).getByText(/Starting focus saved:/).waitFor();
    await p.getByRole('button', { name: 'Play both from start', exact: true }).click(); await p.waitForTimeout(650);
    const playback = await p.locator('.video-comparison-grid video').evaluateAll(vs => vs.map(v => { v.pause(); return { time: v.currentTime, width: v.videoWidth, error: v.error?.code || null }; }));
    assert.equal(playback.length, 2); assert(playback.every(v => v.time > 0 && v.width > 0 && !v.error));
    await p.screenshot({ path: path.join(out, restore ? 'restored-starting-focus.png' : 'agent-uses-starting-focus.png'), fullPage: true });
    await p.reload(); await p.getByRole('button', { name: 'Focus: Bent arms', exact: true }).waitFor();
    assert.deepEqual((await get('/runs/' + run.id + '/events')).events, events);
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    const result = { passed: true, workspace: state.workspace, run_id: run.id, previous_recap_id: source.summary_report_id,
      confirmed_starting_focus: [4], new_video_has_no_analysis_source: true, no_prediction_or_findings_in_reference_response: true,
      source_classification_not_copied: true, goal_resolution_code: turn.llm_resolution.code,
      real_provider_requests: restore ? 0 : 1, new_vision_calls: 0, playback,
      fresh_browser_restored_from_history: restore, mocked_ui_or_cv: false, browser_errors: errors, external_browser_requests: external.length };
    fs.writeFileSync(acceptance, JSON.stringify(result, null, 2)); console.log(JSON.stringify(result));
  } catch (e) {
    for (const c of browser.contexts()) await c.pages()[0]?.screenshot({ path: path.join(out, 'failure-' + Date.now() + '.png'), fullPage: true }).catch(() => {});
    throw e;
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
