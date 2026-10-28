/* One paid goal parse, or a zero-call restart replay. Never retries a paid request. */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..'), out = path.join(root, 'test-results/reference-v1/live');
const baseURL = 'http://127.0.0.1:8007', restore = process.argv.includes('--restore');
const statePath = path.join(out, 'run-state.json');
const read = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const state = fs.existsSync(statePath) ? read(statePath) : {};
const save = () => fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
async function main() {
  assert(restore || process.argv.includes('--allow-live'), 'Explicit --allow-live required');
  fs.mkdirSync(out, { recursive: true });
  const acceptance = path.join(out, restore ? 'restore.json' : 'acceptance.json');
  assert(!fs.existsSync(acceptance), 'Preserve accepted evidence');
  const source = read(path.join(root, 'test-results/reference-v1/ui/acceptance.json'));
  const original = read(path.join(root, 'test-results/product-demo-v2/browser/acceptance.json'));
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  const errors = [], external = [];
  try {
    const c = await browser.newContext({ baseURL, viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    c.on('page', p => p.on('pageerror', e => errors.push(e.message)));
    c.on('request', r => { if (new URL(r.url()).origin !== baseURL) external.push(r.url()); });
    const get = async url => { const r = await c.request.get('/api' + url); assert(r.ok(), url); return r.json(); };
    const caps = await get('/capabilities');
    const p = await c.newPage(); await p.goto('/');
    const current = () => p.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
    if (restore) {
      assert(state.run_id, 'A previously accepted reference run is required');
      await p.locator(`[data-session-id="${source.workspace.session_id}"]`).click();
      await p.waitForFunction(id => JSON.parse(localStorage.getItem('cpr.runId')) === id, state.run_id);
    } else {
      assert.equal(caps.llm.mode, 'deepseek');
      await p.evaluate(({ ws, id }) => {
        localStorage.setItem('cpr.workspace', JSON.stringify(ws));
        localStorage.setItem('cpr.runId', JSON.stringify(id));
      }, { ws: source.workspace, id: state.run_id || source.reference_run_id });
      await p.reload();
      await p.getByRole('region', { name: 'Reference video comparison', exact: true }).waitFor();
      if (!state.run_id) {
        await p.getByLabel('Your question', { exact: true }).fill('Show me a research reference clip for that same finding. Do not analyse my video again.');
        await p.getByRole('button', { name: 'Send message', exact: true }).click();
        await p.waitForFunction(id => {
          const next = JSON.parse(localStorage.getItem('cpr.runId')); return next && next !== id;
        }, source.reference_run_id);
        state.run_id = await current(); save();
      }
    }
    let run;
    for (let i = 0; i < 150; i++) {
      run = await get('/runs/' + state.run_id);
      if (['completed', 'partially_completed', 'failed', 'rejected', 'cancelled', 'waiting_user'].includes(run.status)) break;
      await p.waitForTimeout(200);
    }
    assert.equal(run.status, 'completed', JSON.stringify({ status: run.status, error_code: run.error_code }));
    const turn = await get('/turns/' + run.turn_id), events = (await get('/runs/' + run.id + '/events')).events;
    assert.equal(turn.llm_resolution.called, true);
    assert.equal(turn.llm_resolution.code, 'LLM_GOAL_PARSED');
    assert.equal(turn.resolved_goal.intent, 'show_example');
    assert.deepEqual(turn.resolved_goal.target_label_ids, [4]);
    assert.deepEqual(await get('/runs/' + run.id + '/tools'), []);
    assert.equal(events.filter(e => e.kind === 'reference.lookup').length, 1);
    const providerCalls = events.filter(e => e.payload?.called === true).length;
    assert.equal(providerCalls, 1, 'Only N02 may call DeepSeek for this reference request');
    assert.equal(run.response.prediction, null); assert.deepEqual(run.response.classification_claims, []);
    assert.deepEqual((await get('/runs/' + original.analysis_run_id)).response.prediction, original.prediction);
    const references = await get('/runs/' + run.id + '/references');
    assert.equal(references.status, 'available'); assert.deepEqual(references.requested_label_ids, [4]);
    assert.deepEqual(references.items.map(i => i.label_id), [0, 4]);
    await p.getByRole('region', { name: 'Reference video comparison', exact: true }).waitFor();
    await p.getByRole('button', { name: 'Play both from start', exact: true }).click(); await p.waitForTimeout(700);
    const playback = await p.locator('.video-comparison-grid video').evaluateAll(vs => vs.map(v => {
      v.pause(); return { time: v.currentTime, width: v.videoWidth, height: v.videoHeight, error: v.error?.code || null };
    }));
    assert.equal(playback.length, 2); assert(playback.every(v => v.time > 0 && v.width > 0 && !v.error));
    await p.evaluate(() => scrollTo(0, 0));
    await p.screenshot({ path: path.join(out, restore ? 'after-restart.png' : 'live-reference.png'), fullPage: true });
    await p.reload(); await p.getByRole('region', { name: 'Reference video comparison', exact: true }).waitFor();
    assert.deepEqual((await get('/runs/' + run.id + '/events')).events, events);
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    const result = { passed: true, run_id: run.id, report_id: run.report_id,
      source_analysis_run_id: original.analysis_run_id, reference_ids: references.items.map(i => i.id),
      goal_resolution_code: turn.llm_resolution.code, goal_duration_ms: turn.llm_resolution.duration_ms,
      real_provider_requests: restore ? 0 : providerCalls, new_vision_calls: 0, playback,
      source_prediction_unchanged: true, refresh_reused_saved_lookup: true,
      fresh_browser_restored_from_server_history: restore, service_mode: caps.llm.mode,
      browser_errors: errors, external_browser_requests: external.length, mocked_ui_or_cv: false };
    state.provider_requests = providerCalls; save();
    fs.writeFileSync(acceptance, JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } catch (e) {
    for (const c of browser.contexts()) await c.pages()[0]?.screenshot({ path: path.join(out, 'failure-' + Date.now() + '.png'), fullPage: true }).catch(() => {});
    throw e;
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
