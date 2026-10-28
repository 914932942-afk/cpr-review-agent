/* Actual UI and server state; uses already analysed public video, no new ML or LLM calls. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const sourceArtifacts = path.join(root, 'test-results/context-binding-v1');
const output = process.env.CPR_SMOKE_OUTPUT_DIR ? path.resolve(root, process.env.CPR_SMOKE_OUTPUT_DIR) : sourceArtifacts;
const acceptance = path.join(output, 'ui-acceptance.json');
const baseURL = process.env.CPR_SMOKE_URL || 'http://127.0.0.1:8004';

async function main() {
  assert(!fs.existsSync(acceptance), 'Preserve prior acceptance results');
  fs.mkdirSync(output, { recursive: true });
  const prepared = JSON.parse(fs.readFileSync(path.join(sourceArtifacts, 'http/prepared.json'), 'utf8'));
  const source = JSON.parse(fs.readFileSync(path.join(root, 'test-results/mainline-product-v1/ui-acceptance.json'), 'utf8'));
  const errors = [], external = [], checks = [], runIds = [];
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  try {
    async function context(width) {
      const c = await browser.newContext({ baseURL, viewport: { width, height: 1000 } });
      c.on('page', p => p.on('pageerror', e => errors.push(e.message)));
      await c.route('**/*', route => {
        const u = new URL(route.request().url());
        if (u.origin !== baseURL) { external.push(u.origin); return route.abort(); }
        return route.continue();
      });
      return c;
    }
    const c = await context(1440), page = await c.newPage();
    async function get(url) { const r = await c.request.get(url); assert(r.ok(), url); return r.json(); }
    const caps = await get('/api/capabilities');
    assert.equal(caps.context_binding, 'bound-v1');
    assert.equal(caps.llm.mode, 'disabled');
    const result = page.getByRole('region', { name: 'Workflow result', exact: true });
    const panel = page.getByLabel('Current review context', { exact: true });
    async function openRun(id, report) {
      await page.goto('/', { waitUntil: 'networkidle' });
      await page.evaluate(({ ws, id, report }) => {
        localStorage.setItem('cpr.workspace', JSON.stringify({ ...ws, source_report_id: report }));
        localStorage.setItem('cpr.runId', JSON.stringify(id));
      }, { ws: source.workspace, id, report });
      await page.reload({ waitUntil: 'domcontentloaded' });
    }
    async function settled(previous) {
      await page.waitForFunction(prev => {
        const id = JSON.parse(localStorage.getItem('cpr.runId')); return id && id !== prev;
      }, previous);
      const id = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
      let run;
      for (let i = 0; i < 100; i++) {
        run = await get(`/api/runs/${id}`);
        if (['completed', 'failed', 'rejected', 'cancelled'].includes(run.status)) break;
        await page.waitForTimeout(150);
      }
      assert.equal(run.status, 'completed', JSON.stringify(run));
      assert.equal(run.response.reused_result, true);
      assert.deepEqual(await get(`/api/runs/${id}/tools`), []);
      await result.getByRole('heading', { name: run.response.title, exact: true }).waitFor();
      runIds.push(id);
      return run;
    }
    await openRun(prepared.broad_run_id, prepared.broad_report_id);
    await panel.getByText(`report ${prepared.broad_report_id.slice(0, 8)}`, { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Continue this review scope', exact: true }).click();
    const inherited = await settled(prepared.broad_run_id);
    assert.deepEqual(inherited.response.review.displayed_label_ids, [2, 4]);
    assert.equal(inherited.review_context.source_report_id, prepared.broad_report_id);
    checks.push('Opening an older report binds follow-up to that report and its confirmed scope');

    // Create an actual ambiguous server request, equivalent to a parsed reference contract.
    const response = await c.request.post(`/api/sessions/${source.workspace.session_id}/turns`, {
      headers: { 'Idempotency-Key': require('node:crypto').randomUUID() },
      data: { goal: { intent: 'explain' }, context: { source_report_id: prepared.broad_report_id, reference: { scope: 'single_finding', source: 'viewed_report' } } },
    });
    assert.equal(response.status(), 202);
    const pending = await response.json();
    await openRun(pending.run_id, prepared.broad_report_id);
    const form = page.getByRole('region', { name: 'Structured goal form', exact: true });
    await form.waitFor();
    await page.waitForFunction(() => document.querySelectorAll('.target-options button[aria-pressed=true]').length === 0);
    assert.equal(await page.locator('.target-options button[aria-pressed=true]').count(), 0);
    const next = form.getByRole('button', { name: 'Continue from checkpoint →', exact: true });
    assert(await next.isDisabled());
    await form.getByLabel('Intent', { exact: true }).selectOption('explain');
    await page.getByRole('button', { name: `4: ${caps.label_names[4]}`, exact: true }).click();
    assert(await next.isDisabled());
    await form.locator('input[type=checkbox]').check();
    await next.click();
    const clarified = await settled(inherited.id);
    assert.deepEqual(clarified.response.review.displayed_label_ids, [4]);
    assert.equal(clarified.review_context.source_report_id, prepared.broad_report_id);
    await page.getByLabel('Saved context binding', { exact: true }).locator('summary').click();
    await page.getByLabel('Saved context binding', { exact: true }).getByText(`Input report: ${prepared.broad_report_id}`, { exact: true }).waitFor();
    await panel.getByText(`report ${clarified.report_id.slice(0, 8)}`, { exact: true }).waitFor();
    checks.push('Ambiguous category requires both a chosen category and explicit source confirmation');
    await page.screenshot({ path: path.join(output, 'context-1440.png'), fullPage: true });

    // Disabled mode must use the form and must not silently parse the request.
    await page.locator('.request-details > summary').click();
    await page.getByLabel('Your request', { exact: true }).fill('Explain this issue from the report I am viewing.');
    await page.getByRole('button', { name: 'Choose task in a form →', exact: true }).click();
    await form.waitFor();
    const before = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
    const paused = await get(`/api/runs/${before}`);
    const turn = await get(`/api/turns/${paused.turn_id}`);
    assert.equal(turn.llm_resolution.called, false);
    assert.equal(turn.llm_resolution.code, 'LLM_DISABLED');
    assert.equal(turn.context_snapshot.source.report_id, clarified.report_id);
    await form.getByLabel('Intent', { exact: true }).selectOption('explain');
    await form.locator('input[type=checkbox]').check();
    await form.getByRole('button', { name: 'Continue from checkpoint →', exact: true }).click();
    const afterForm = await settled(clarified.id);
    checks.push('Text request in disabled mode confirms the already bound video/report without external calls');

    const video = path.join(root, 'data/public_cpr/rgb-acceptance-v1/records/SupC04--r00--ch0/video.mp4');
    await page.locator('input[type=file]').setInputFiles(video);
    await panel.getByText(/No saved analysis yet/).waitFor();
    assert(await page.getByRole('button', { name: 'Continue this review scope', exact: true }).isDisabled());
    const newWs = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.workspace')));
    assert.notEqual(newWs.session_id, source.workspace.session_id);
    assert.equal(newWs.source_report_id || null, null);
    assert.deepEqual((await get(`/api/sessions/${newWs.session_id}/history`)).items, []);
    checks.push('Uploading a new video clears the previous source and scope, even for identical bytes');

    const mobile = await context(390), mp = await mobile.newPage();
    await mp.goto('/', { waitUntil: 'networkidle' });
    const select = mp.getByLabel('Saved session', { exact: true });
    await select.locator(`option[value="${source.workspace.session_id}"]`).waitFor({ state: 'attached' });
    await select.selectOption(source.workspace.session_id);
    const mr = mp.getByRole('region', { name: 'Workflow result', exact: true });
    await mr.getByRole('heading', { name: 'Finding evidence', exact: true }).waitFor();
    await mp.getByRole('button', { name: 'Continue this review scope', exact: true }).click();
    let mobileResult;
    for (let i = 0; i < 100; i++) {
      const id = await mp.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
      const run = await get(`/api/runs/${id}`);
      if (id !== afterForm.id && run.status === 'completed') { mobileResult = run; runIds.push(id); assert.deepEqual(await get(`/api/runs/${id}/tools`), []); break; }
      await mp.waitForTimeout(200);
      assert(i < 99, 'Mobile follow-up did not complete');
    }
    await mp.getByLabel('Saved context binding', { exact: true }).waitFor();
    await mp.getByLabel('Current review context', { exact: true }).getByText(`report ${mobileResult.report_id.slice(0, 8)}`, { exact: true }).waitFor();
    await mr.getByText('Verified source · whole-clip record', { exact: true }).waitFor();
    await mp.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
    assert.equal(await mp.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await mp.screenshot({ path: path.join(output, 'context-390.png'), fullPage: true });
    await mp.reload({ waitUntil: 'networkidle' });
    await mp.getByLabel('Saved context binding', { exact: true }).waitFor();
    await mp.getByLabel('Current review context', { exact: true }).getByText(`report ${mobileResult.report_id.slice(0, 8)}`, { exact: true }).waitFor();
    checks.push('Fresh mobile browser restores the binding and continues the same review without new analysis');
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    assert.equal((await get('/api/health')).llm, 'disabled');
    fs.writeFileSync(acceptance, JSON.stringify({ passed: true, mode: 'actual_browser_saved_real_video', checks,
      tested_run_ids: runIds, new_video_session_id: newWs.session_id,
      new_vision_calls: 0, real_llm_calls: 0, mock_ui_or_cv_results: false,
      widths: [1440, 390], browser_errors: errors, external_browser_requests: external.length }, null, 2));
    console.log(JSON.stringify({ passed: true, checks, new_vision_calls: 0, real_llm_calls: 0 }));
  } catch (error) {
    const last = browser.contexts()[0]?.pages()[0];
    if (last) await last.screenshot({ path: path.join(output, `failure-${Date.now()}.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
