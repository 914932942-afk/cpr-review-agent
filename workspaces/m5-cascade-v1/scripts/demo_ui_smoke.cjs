/* Real demo UI with accepted public-video results; never fakes API or classifier responses. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'test-results/demo-acceptance-v1/ui');
const baseURL = 'http://127.0.0.1:8005';

async function main() {
  fs.mkdirSync(output, { recursive: true });
  assert(!fs.existsSync(path.join(output, 'acceptance.json')), 'Never overwrite prior UI acceptance');
  const record = JSON.parse(fs.readFileSync(path.join(root, 'test-results/demo-acceptance-v1/http/acceptance.json'), 'utf8'));
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  const errors = [], external = [], checks = [], runs = [];
  try {
    async function newContext(width) {
      const context = await browser.newContext({ baseURL, viewport: { width, height: 1000 } });
      context.on('page', p => p.on('pageerror', e => errors.push(e.message)));
      await context.route('**/*', r => {
        if (new URL(r.request().url()).origin !== baseURL) { external.push(r.request().url()); return r.abort(); }
        return r.continue();
      });
      return context;
    }
    const context = await newContext(1440), page = await context.newPage();
    const get = async url => { const r = await context.request.get(url); assert(r.ok(), url); return r.json(); };
    const result = p => p.getByRole('region', { name: 'Workflow result', exact: true });
    const explanation = p => p.getByLabel('Evidence-constrained explanation', { exact: true });
    async function settled(previous) {
      await page.waitForFunction(old => { const id = JSON.parse(localStorage.getItem('cpr.runId')); return id && id !== old; }, previous);
      const id = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
      let run;
      for (let i = 0; i < 100; i++) {
        run = await get(`/api/runs/${id}`);
        if (['completed','failed','rejected','cancelled'].includes(run.status)) break;
        await page.waitForTimeout(150);
      }
      assert.equal(run.status, 'completed');
      assert.deepEqual(run.response.prediction, record.prediction);
      assert.deepEqual(await get(`/api/runs/${id}/tools`), []);
      await explanation(page).getByText(run.grounded_review.summary, { exact: true }).waitFor();
      await page.getByLabel('Current review context', { exact: true }).getByText(`report ${run.report_id.slice(0,8)}`, { exact: true }).waitFor();
      runs.push(id); return run;
    }
    await page.goto('/', { waitUntil: 'networkidle' });
    assert(await page.getByRole('button', { name: 'Review selected categories →', exact: true }).isDisabled());
    await page.locator('input[type=file]').setInputFiles({ name: 'broken.mp4', mimeType: 'video/mp4', buffer: Buffer.from('not an mp4') });
    await page.getByRole('alert').filter({ hasText: 'Container header does not match the selected format' }).waitFor();
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.workspace'))), null);
    await page.screenshot({ path: path.join(output, 'invalid-upload.png') });
    await page.getByRole('button', { name: 'Dismiss error', exact: true }).click();
    checks.push('No-video actions are guarded; invalid upload gives an actionable error without creating a session');

    const selector = page.getByLabel('Saved session', { exact: true });
    await selector.locator(`option[value="${record.workspace.session_id}"]`).waitFor({ state: 'attached' });
    await selector.selectOption(record.workspace.session_id);
    await explanation(page).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    await result(page).getByRole('button', { name: 'Review this category', exact: true }).click();
    const explained = await settled(record.final_summary_run_id);
    await page.getByLabel('Continuous review', { exact: true }).getByRole('button', { name: 'Summarise displayed scope', exact: true }).click();
    const summary = await settled(explained.id);
    const exportResponse = await context.request.get(`/api/reports/${summary.report_id}/export`);
    assert(exportResponse.ok()); assert((await exportResponse.text()).includes('Evidence-constrained review'));
    checks.push('Saved real analysis → category follow-up → summary and printable export; zero new vision executions');

    await page.locator('.request-details > summary').click();
    await page.getByLabel('Your request', { exact: true }).fill('Explain this result.');
    await page.getByRole('button', { name: 'Choose task in a form →', exact: true }).click();
    await page.getByRole('region', { name: 'Structured goal form', exact: true }).waitFor();
    const cancelledId = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await page.getByLabel('Analysis progress', { exact: true }).getByText('Request cancelled', { exact: true }).waitFor();
    assert.equal((await get(`/api/runs/${cancelledId}`)).response, null);
    assert.deepEqual(await get(`/api/runs/${cancelledId}/tools`), []);
    await page.screenshot({ path: path.join(output, 'cancelled-request.png'), fullPage: true });
    checks.push('Text falls back to explicit confirmation while DeepSeek is disabled; cancelling produces no result');

    const failedRun = await get(`/api/runs/${record.failed_decode_run_id}`);
    await selector.selectOption(failedRun.session_id);
    await result(page).getByRole('heading', { name: 'Review unavailable', exact: true }).waitFor();
    assert.equal(await page.locator('.finding-list').count(), 0);
    assert.equal(await explanation(page).count(), 0);
    await page.screenshot({ path: path.join(output, 'decode-failure.png'), fullPage: true });
    checks.push('Decode failure does not retain the previous video findings or explanation in the interface');

    await selector.selectOption(record.workspace.session_id);
    await page.getByLabel('Analysis progress', { exact: true }).getByText('Request cancelled', { exact: true }).waitFor();
    await page.locator('.history-list li').filter({ has: page.getByText('Review summary', { exact: true }) }).first().getByRole('button', { name: 'Open record', exact: true }).click();
    await explanation(page).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    await page.getByLabel('Continuous review', { exact: true }).getByRole('button', { name: 'Summarise displayed scope', exact: true }).click();
    const final = await settled(summary.id);
    await page.reload({ waitUntil: 'networkidle' });
    await explanation(page).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({ path: path.join(output, 'demo-1440.png'), fullPage: true });
    checks.push('History can reopen a successful summary after cancellation/failure; refresh restores the same saved result');

    const mobile = await newContext(390), mp = await mobile.newPage();
    await mp.goto('/', { waitUntil: 'networkidle' });
    const mobileSelector = mp.getByLabel('Saved session', { exact: true });
    await mobileSelector.locator(`option[value="${record.workspace.session_id}"]`).waitFor({ state: 'attached' });
    await mobileSelector.selectOption(record.workspace.session_id);
    await explanation(mp).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    assert.equal(await mp.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId'))), final.id);
    const video = mp.locator('video'); await video.scrollIntoViewIfNeeded();
    await video.evaluate(async e => { e.muted = true; await e.play(); });
    await mp.waitForTimeout(800);
    const playback = await video.evaluate(e => { e.pause(); return { time: e.currentTime, width: e.videoWidth, height: e.videoHeight, error: e.error?.code || null }; });
    assert(playback.time > 0 && playback.width > 0 && !playback.error);
    assert.equal(await mp.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await mp.screenshot({ path: path.join(output, 'demo-390.png'), fullPage: true });
    checks.push('Fresh mobile browser recovers the server-saved session, verified explanation and playable source video');
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    fs.writeFileSync(path.join(output, 'acceptance.json'), JSON.stringify({ passed: true, checks,
      workspace: record.workspace, final_summary_run_id: final.id, final_summary_report_id: final.report_id,
      tested_run_ids: runs, browser_cancelled_run_id: cancelledId, new_vision_calls: 0, real_llm_calls: 0,
      mocked_ui_or_cv: false, widths: [1440,390], mobile_playback: playback, errors, external_requests: external.length }, null, 2));
    console.log(JSON.stringify({ passed: true, checks, new_vision_calls: 0, real_llm_calls: 0 }));
  } catch (e) {
    for (const [i, c] of browser.contexts().entries()) await c.pages()[0]?.screenshot({ path: path.join(output, `failure-${i}-${Date.now()}.png`), fullPage: true }).catch(() => {});
    throw e;
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
