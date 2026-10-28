/* Real browser and public videos. One independent CV run; no paid requests or mocked APIs. */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..'), out = path.join(root, 'test-results/practice-continuity-v1/ui');
const baseURL = 'http://127.0.0.1:8007', restore = process.argv.includes('--restore');
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const statePath = path.join(out, 'run-state.json');
const state = fs.existsSync(statePath) ? JSON.parse(fs.readFileSync(statePath, 'utf8')) : {};
const save = () => fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
const source = read('test-results/product-demo-v2/browser/acceptance.json');
async function main() {
  fs.mkdirSync(out, { recursive: true });
  const target = path.join(out, restore ? 'restore.json' : 'acceptance.json');
  assert(!fs.existsSync(target), 'Preserve accepted evidence');
  assert(restore || process.argv.includes('--allow-vision'), 'Explicit --allow-vision required for the one new CV run');
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  const errors = [], external = [];
  try {
    const c = await browser.newContext({ baseURL, viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    c.on('page', p => p.on('pageerror', e => errors.push(e.message)));
    c.on('request', r => { if (new URL(r.url()).origin !== baseURL) external.push(r.url()); });
    const get = async u => { const r = await c.request.get('/api' + u); assert(r.ok(), u); return r.json(); };
    assert.equal((await get('/capabilities')).practice_continuity, true);
    const p = await c.newPage(); await p.goto('/');
    const current = () => p.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
    const panel = () => p.getByRole('region', { name: 'Practice continuity', exact: true });
    const seed = async (ws, id) => {
      await p.evaluate(({ ws, id }) => { localStorage.setItem('cpr.workspace', JSON.stringify(ws)); localStorage.setItem('cpr.runId', JSON.stringify(id)); }, { ws, id });
      await p.reload();
    };
    if (restore) {
      assert(state.analysis_run_id && state.workspace);
      await p.locator(`[data-session-id="${state.workspace.session_id}"]`).click();
      await p.waitForFunction(id => JSON.parse(localStorage.getItem('cpr.runId')) === id, state.analysis_run_id);
    } else {
      if (!state.workspace) {
        await seed(source.workspace, source.summary_run_id);
        await p.getByRole('heading', { name: 'Your review recap', exact: true }).waitFor();
        await p.getByRole('button', { name: /^Focus:/ }).click();
        await p.locator('.focus-drawer').getByRole('button', { name: 'Clear', exact: true }).click();
        await p.locator('.focus-drawer').getByRole('button', { name: 'Bent arms', exact: true }).click();
        await p.locator('input[type=file]').setInputFiles({ name: 'follow-up-practice.mp4', mimeType: 'video/mp4', buffer: fs.readFileSync(path.join(root, 'data/public_cpr/rgb-acceptance-v1/records/SupC05--r00--ch0/video.mp4')) });
        await p.waitForFunction(old => JSON.parse(localStorage.getItem('cpr.workspace'))?.session_id !== old, source.workspace.session_id);
        state.workspace = await p.evaluate(() => JSON.parse(localStorage.getItem('cpr.workspace'))); save();
      } else await seed(state.workspace, state.analysis_run_id || null);
      if (!state.setup_complete) {
        await panel().waitFor();
        if ((await get(`/sessions/${state.workspace.session_id}/continuity`)).selection.status !== 'not_selected') {
          await panel().getByRole('button', { name: 'Start fresh', exact: true }).click();
        }
        await p.getByRole('button', { name: 'Focus: All categories', exact: true }).waitFor();
        assert.equal(await p.getByRole('region', { name: 'Practice findings', exact: true }).count(), 0);
        assert.equal(await p.getByRole('region', { name: 'Reference video comparison', exact: true }).count(), 0);
        const before = await get(`/sessions/${state.workspace.session_id}/context`);
        assert.equal(before.source, null); assert.deepEqual(before.active_label_ids, []);
        await panel().getByLabel('Earlier recap', { exact: true }).selectOption(source.summary_report_id);
        await panel().getByRole('button', { name: 'View earlier recap', exact: true }).click();
        const modal = p.getByRole('dialog', { name: 'Previous practice recap', exact: true });
        await modal.getByText(source.workspace.name, { exact: true }).waitFor();
        await modal.getByText(/These findings belong to the earlier video/).waitFor();
        assert.equal(await current(), null);
        assert.equal((await get(`/sessions/${state.workspace.session_id}/context`)).source, null);
        await p.screenshot({ path: path.join(out, 'previous-recap-1440.png'), fullPage: true });
        await p.keyboard.press('Escape');
        await panel().getByRole('button', { name: 'Use this focus', exact: true }).click();
        await panel().getByText(/Starting focus saved:/).waitFor();
        await p.getByRole('button', { name: 'Focus: Bent arms', exact: true }).waitFor();
        let context = await get(`/sessions/${state.workspace.session_id}/context`);
        assert.equal(context.source, null); assert.equal(context.source_status, 'none'); assert.deepEqual(context.active_label_ids, [4]);
        const link = await get(`/sessions/${state.workspace.session_id}/continuity`);
        assert.equal(link.selection.source.same_video_file, false);
        await p.reload(); await panel().getByText(/Starting focus saved:/).waitFor();
        await p.getByRole('button', { name: 'Focus: Bent arms', exact: true }).waitFor();
        for (const width of [1440, 1024, 768, 390]) {
          await p.setViewportSize({ width, height: 1000 }); await p.waitForTimeout(80);
          assert.equal(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
          await p.screenshot({ path: path.join(out, `starting-focus-${width}.png`), fullPage: true });
        }
        await panel().getByRole('button', { name: 'View earlier recap', exact: true }).click();
        await modal.getByText(source.workspace.name, { exact: true }).waitFor();
        assert.equal(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        await p.screenshot({ path: path.join(out, 'previous-recap-390.png'), fullPage: true }); await p.keyboard.press('Escape');
        await p.setViewportSize({ width: 1440, height: 1000 });
        await panel().getByRole('button', { name: 'Start fresh', exact: true }).click();
        await p.getByRole('button', { name: 'Focus: All categories', exact: true }).waitFor();
        await panel().getByRole('button', { name: 'Use this focus', exact: true }).click();
        await p.getByRole('button', { name: 'Focus: Bent arms', exact: true }).waitFor();
        assert.deepEqual(await get('/preferences'), []);
        state.setup_complete = true; save();
      }
      if (!state.analysis_run_id) {
        await p.getByRole('button', { name: 'Start review', exact: true }).click();
        await p.waitForFunction(() => !!JSON.parse(localStorage.getItem('cpr.runId')));
        state.analysis_run_id = await current(); save();
      }
    }
    const started = Date.now(); let run, previous = '';
    for (let n = 0; n < 1200; n++) {
      run = await get('/runs/' + state.analysis_run_id);
      const tools = await get('/runs/' + run.id + '/tools');
      const marker = run.status + ':' + tools.map(t => t.request.tool_id + '=' + t.status).join(',');
      if (marker !== previous) { console.log(marker); previous = marker; }
      if (['completed', 'partially_completed', 'failed', 'rejected', 'cancelled', 'waiting_user'].includes(run.status)) break;
      await p.waitForTimeout(500);
    }
    assert.equal(run.status, 'completed', JSON.stringify({ status: run.status, code: run.error_code }));
    const tools = await get('/runs/' + run.id + '/tools');
    assert.deepEqual(tools.map(t => t.request.tool_id), ['pose.extract', 'skeleton.predict', 'rgb.predict', 'fusion.predict']);
    assert(tools.every(t => t.status === 'succeeded' && !t.output?.is_stub));
    assert.equal(run.response.prediction.clip_id, state.workspace.clip_id);
    assert.notEqual(run.response.prediction.video_sha256, source.prediction.video_sha256);
    assert.deepEqual(run.response.classification_claims.map(e => e.label_id), [4]);
    const events = (await get('/runs/' + run.id + '/events')).events;
    assert.equal(events.filter(e => e.payload?.called === true).length, 0);
    const turn = await get('/turns/' + run.turn_id);
    assert.equal(turn.context_snapshot.source, null); assert.deepEqual(turn.context_snapshot.active_label_ids, [4]);
    const context = await get(`/sessions/${state.workspace.session_id}/context`);
    assert.equal(context.source.report_id, run.report_id);
    const linked = await get(`/sessions/${state.workspace.session_id}/continuity`);
    assert.equal(linked.locked, true); assert.equal(linked.selection.source.report_id, source.summary_report_id);
    await p.getByRole('region', { name: 'Practice findings', exact: true }).waitFor(); await panel().waitFor();
    await panel().getByRole('button', { name: 'View earlier recap', exact: true }).click();
    await p.getByRole('dialog', { name: 'Previous practice recap', exact: true }).getByText(source.workspace.name, { exact: true }).waitFor();
    assert.equal(await current(), run.id); await p.keyboard.press('Escape');
    assert.deepEqual((await get('/runs/' + source.analysis_run_id)).response.prediction, source.prediction);
    const player = p.getByLabel('Practice video player', { exact: true });
    await player.evaluate(async v => { v.muted = true; await v.play(); }); await p.waitForTimeout(650);
    const playback = await player.evaluate(v => { v.pause(); return { time: v.currentTime, width: v.videoWidth, error: v.error?.code || null }; });
    assert(playback.time > 0 && playback.width > 0 && !playback.error);
    await p.evaluate(() => scrollTo(0, 0)); await p.screenshot({ path: path.join(out, restore ? 'after-restart.png' : 'independent-new-review.png'), fullPage: true });
    await p.reload(); await panel().getByText(/Starting focus saved:/).waitFor();
    assert.equal(await current(), run.id); assert.deepEqual((await get('/runs/' + run.id + '/events')).events, events);
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    const result = { passed: true, workspace: state.workspace, analysis_run_id: run.id, report_id: run.report_id,
      previous_session_id: source.workspace.session_id, previous_recap_id: source.summary_report_id,
      explicit_scope: [4], previous_prediction_unchanged: true, new_video_hash_is_distinct: true,
      carried_scope_is_not_source_evidence: true, current_context_uses_new_report: true,
      old_recap_opened_without_switching_video: true, scope_reset_and_clear_passed: true,
      new_vision_calls: restore ? 0 : 4, real_provider_requests: 0,
      real_tools: tools.map(t => ({ id: t.request.tool_id, status: t.status, duration_ms: t.duration_ms })),
      wait_ms: Date.now() - started, playback, widths: [1440, 1024, 768, 390],
      service_restarted: restore, fresh_browser_restored_from_history: restore,
      browser_errors: errors, external_browser_requests: external.length, mocked_ui_or_cv: false };
    fs.writeFileSync(target, JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ passed: true, run_id: run.id, new_vision_calls: result.new_vision_calls, paid_calls: 0, restore }));
  } catch (e) {
    for (const c of browser.contexts()) await c.pages()[0]?.screenshot({ path: path.join(out, 'failure-' + Date.now() + '.png'), fullPage: true }).catch(() => {});
    throw e;
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
