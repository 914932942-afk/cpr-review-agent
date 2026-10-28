/* Explicit live UI check: one text turn, at most one goal parse and one organisation request. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'test-results/deepseek-live-v1/ui');
const baseURL = 'http://127.0.0.1:8006';

async function main() {
  assert(process.argv.includes('--allow-live'), 'Include --allow-live to explicitly allow provider requests');
  assert(!fs.existsSync(path.join(output, 'acceptance.json')), 'Preserve prior acceptance');
  fs.mkdirSync(output, { recursive: true });
  const live = JSON.parse(fs.readFileSync(path.join(root, 'test-results/deepseek-live-v1/http/acceptance.json'), 'utf8'));
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  const errors = [], external = [];
  try {
    const context = await browser.newContext({ baseURL, viewport: { width: 1440, height: 1000 } });
    context.on('page', p => p.on('pageerror', e => errors.push(e.message)));
    await context.route('**/*', r => {
      if (new URL(r.request().url()).origin !== baseURL) { external.push(r.request().url()); return r.abort(); }
      return r.continue();
    });
    const get = async url => { const r = await context.request.get(url); assert(r.ok(), url); return r.json(); };
    assert.equal((await get('/api/capabilities')).llm.mode, 'deepseek');
    const page = await context.newPage();
    await page.goto('/', { waitUntil: 'networkidle' });
    await page.evaluate(record => {
      localStorage.setItem('cpr.workspace', JSON.stringify({ ...record.workspace, source_report_id: record.single_report_id }));
      localStorage.setItem('cpr.runId', JSON.stringify(record.single_run_id));
    }, live);
    await page.reload({ waitUntil: 'networkidle' });
    const explanation = page.getByLabel('Evidence-constrained explanation', { exact: true });
    await explanation.getByText(/DeepSeek organised the categories/).waitFor();
    await page.locator('.request-details > summary').click();
    await page.getByLabel('Your request', { exact: true }).fill('继续解释刚才那个问题的已有结果，不要重新分析视频。');
    await page.getByRole('button', { name: 'Submit request →', exact: true }).click();
    await page.waitForFunction(old => { const id = JSON.parse(localStorage.getItem('cpr.runId')); return id && id !== old; }, live.single_run_id);
    const rid = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
    let run;
    for (let i = 0; i < 150; i++) {
      run = await get(`/api/runs/${rid}`);
      if (['completed','failed','rejected','cancelled','waiting_user'].includes(run.status)) break;
      await page.waitForTimeout(200);
    }
    assert.equal(run.status, 'completed', 'Live text request did not finish normally');
    assert.deepEqual(run.grounded_review.scope_label_ids, [4]);
    assert.equal(run.grounded_review.mode, 'model_organised_evidence');
    assert.equal(run.grounded_review.organisation_receipt.code, 'LLM_PLAN_VALIDATED');
    assert.deepEqual(await get(`/api/runs/${rid}/tools`), []);
    const source = await get(`/api/runs/${live.source_analysis_run_id}`);
    assert.deepEqual(run.response.prediction, source.response.prediction);
    await explanation.getByText(/DeepSeek organised the categories/).waitFor();
    await page.getByLabel('Current review context', { exact: true }).getByText(`report ${run.report_id.slice(0,8)}`, { exact: true }).waitFor();
    await explanation.scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, 'live-1440.png'), fullPage: true });
    await page.reload({ waitUntil: 'networkidle' });
    await explanation.getByText(/DeepSeek organised the categories/).waitFor();
    await page.setViewportSize({ width: 390, height: 950 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await explanation.scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, 'live-390.png'), fullPage: true });
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    const turn = await get(`/api/turns/${run.turn_id}`);
    assert.equal(turn.llm_resolution.code, 'LLM_GOAL_PARSED');
    assert.equal(turn.llm_resolution.called, true);
    const value = { passed: true, run_id: rid, report_id: run.report_id, scope: [4],
      actual_browser_text_to_live_provider: true, real_provider_requests: 2, new_vision_calls: 0,
      refresh_reused_saved_explanation: true, widths: [1440,390], browser_errors: errors,
      external_browser_requests: external.length, goal_duration_ms: turn.llm_resolution.duration_ms,
      organisation_duration_ms: run.grounded_review.organisation_receipt.duration_ms };
    fs.writeFileSync(path.join(output, 'acceptance.json'), JSON.stringify(value, null, 2));
    console.log(JSON.stringify(value));
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
