/* Real browser + saved public-video analysis. No fake UI responses, CV or provider calls. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const output = path.resolve(root, process.env.CPR_SMOKE_OUTPUT_DIR || 'test-results/grounded-review-v1/ui');
const baseURL = process.env.CPR_SMOKE_URL || 'http://127.0.0.1:8004';

async function main() {
  fs.mkdirSync(output, { recursive: true });
  assert(!fs.existsSync(path.join(output, 'ui-acceptance.json')), 'Preserve prior acceptance');
  const original = JSON.parse(fs.readFileSync(path.join(root, 'test-results/mainline-product-v1/ui-acceptance.json'), 'utf8'));
  const prepared = JSON.parse(fs.readFileSync(path.join(root, 'test-results/grounded-review-v1/context-http-v2/prepared.json'), 'utf8'));
  const errors = [], external = [], checks = [], runIds = [];
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  try {
    async function newContext(width) {
      const context = await browser.newContext({ baseURL, viewport: { width, height: 1000 } });
      context.on('page', page => page.on('pageerror', e => errors.push(e.message)));
      await context.route('**/*', route => {
        if (new URL(route.request().url()).origin !== baseURL) {
          external.push(route.request().url()); return route.abort();
        }
        return route.continue();
      });
      return context;
    }
    const context = await newContext(1440), page = await context.newPage();
    async function get(url) {
      const r = await context.request.get(url); assert(r.ok(), url); return r.json();
    }
    const oracle = (await get(`/api/runs/${original.run_id}`)).response.prediction;
    const caps = await get('/api/capabilities');
    assert.equal(caps.llm.mode, 'disabled'); assert.equal(caps.grounded_explanations, true);
    const result = p => p.getByRole('region', { name: 'Workflow result', exact: true });
    const explanation = p => p.getByLabel('Evidence-constrained explanation', { exact: true });
    const trail = p => p.getByLabel('Continuous review', { exact: true });
    async function selected(p, ids) {
      await p.waitForFunction(expected => {
        const current = [...document.querySelectorAll('.target-options button[aria-pressed=true]')].map(e => Number(e.getAttribute('aria-label').split(':')[0]));
        return JSON.stringify(current.sort((a,b) => a-b)) === JSON.stringify(expected);
      }, [...ids].sort((a,b) => a-b));
    }
    async function waitRun(p, previous, scope) {
      await p.waitForFunction(id => {
        const current = JSON.parse(localStorage.getItem('cpr.runId')); return current && current !== id;
      }, previous);
      const id = await p.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
      let run;
      for (let i = 0; i < 150; i++) {
        run = await get(`/api/runs/${id}`);
        if (['completed','failed','rejected','cancelled'].includes(run.status)) break;
        await p.waitForTimeout(100);
      }
      assert.equal(run.status, 'completed', JSON.stringify(run));
      assert.deepEqual(run.response.prediction, oracle);
      assert.deepEqual(run.grounded_review.scope_label_ids, scope);
      assert.equal(run.response.reused_result, true);
      assert.deepEqual(await get(`/api/runs/${id}/tools`), []);
      await explanation(p).getByText(run.grounded_review.summary, { exact: true }).waitFor();
      await trail(p).waitFor();
      await p.getByLabel('Current review context', { exact: true }).getByText(`report ${run.report_id.slice(0,8)}`, { exact: true }).waitFor();
      await selected(p, scope);
      runIds.push(id);
      return run;
    }
    await page.goto('/', { waitUntil: 'networkidle' });
    await page.evaluate(({workspace, run, report}) => {
      localStorage.setItem('cpr.workspace', JSON.stringify({ ...workspace, source_report_id: report }));
      localStorage.setItem('cpr.runId', JSON.stringify(run));
    }, { workspace: original.workspace, run: prepared.broad_run_id, report: prepared.broad_report_id });
    await page.reload({ waitUntil: 'networkidle' });
    await explanation(page).getByRole('heading', { name: 'Understand this result' }).waitFor();
    await explanation(page).getByText('1 model-flagged categories in this 2-category review scope.', { exact: true }).waitFor();
    await selected(page, [2, 4]);
    const details = explanation(page).locator('details').filter({ hasText: '#4 · Bent arms' }).first();
    await details.locator('summary').click();
    await details.getByText(/A dataset category referring to arm bending during compression\./).waitFor();
    await details.getByText(/The saved final output flags this category/).waitFor();
    await details.getByText(/Recorded branch readout/).waitFor();
    assert.equal(await explanation(page).getByText(/Evidence template/).count(), 1);
    checks.push('Verified category meaning, exact saved model flag and separate raw score interpretation');

    await result(page).locator('.finding-list > div').filter({ hasText: '#4 · Bent arms' }).getByRole('button', { name: 'Review this category', exact: true }).click();
    const narrow = await waitRun(page, prepared.broad_run_id, [4]);
    await trail(page).getByRole('button', { name: 'Summarise displayed scope', exact: true }).click();
    const summary = await waitRun(page, narrow.id, [4]);
    assert.equal(summary.grounded_review.intent, 'summarise');
    await explanation(page).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    const exported = await context.request.get(`/api/reports/${summary.report_id}/export`);
    assert(exported.ok()); const html = await exported.text();
    assert(html.includes('Evidence-constrained review') && html.includes('1 model-flagged categories in this 1-category review scope.'));
    assert(!html.includes('<h3>#2'));
    checks.push('Category follow-up → scope summary → export preserves one-category scope and the original prediction');

    await page.getByRole('button', { name: '2: Clenched hands', exact: true }).click();
    await page.getByRole('button', { name: 'Explain selected findings', exact: true }).click();
    const broader = await waitRun(page, summary.id, [2, 4]);
    await trail(page).getByRole('button', { name: 'Open latest saved summary', exact: true }).click();
    await page.waitForFunction(id => JSON.parse(localStorage.getItem('cpr.runId')) === id, summary.id);
    await explanation(page).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    await selected(page, [4]);
    await page.reload({ waitUntil: 'networkidle' });
    await explanation(page).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    await selected(page, [4]);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({ path: path.join(output, 'review-1440.png'), fullPage: true });
    checks.push('Adjust scope → reopen the saved summary → refresh restores that report and its confirmed scope');

    await page.getByRole('button', { name: 'Summarise record', exact: true }).click();
    const full = await waitRun(page, summary.id, caps.candidate_labels);
    assert.deepEqual(full.grounded_review.outside_scope_label_ids, []);
    await explanation(page).getByText(/All assessed categories are included/).waitFor();
    checks.push('Full-record summary is explicitly distinguishable from the displayed-scope summary');

    const mobile = await newContext(390), mp = await mobile.newPage();
    await mp.goto('/', { waitUntil: 'networkidle' });
    const selector = mp.getByLabel('Saved session', { exact: true });
    await selector.locator(`option[value="${original.workspace.session_id}"]`).waitFor({ state: 'attached' });
    await selector.selectOption(original.workspace.session_id);
    await explanation(mp).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    await selected(mp, caps.candidate_labels);
    await result(mp).locator('.finding-list > div').filter({ hasText: '#4 · Bent arms' }).getByRole('button', { name: 'Review this category', exact: true }).click();
    const mNarrow = await waitRun(mp, full.id, [4]);
    await trail(mp).getByRole('button', { name: 'Summarise displayed scope', exact: true }).click();
    const mSummary = await waitRun(mp, mNarrow.id, [4]);
    await mp.reload({ waitUntil: 'networkidle' });
    await explanation(mp).getByRole('heading', { name: 'Saved review summary', exact: true }).waitFor();
    await selected(mp, [4]);
    await mp.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
    assert.equal(await mp.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await explanation(mp).scrollIntoViewIfNeeded();
    await mp.screenshot({ path: path.join(output, 'review-390.png'), fullPage: true });
    checks.push('Fresh mobile browser restores the saved video and continues category review and summary without inference');
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    assert.equal((await get('/api/health')).llm, 'disabled');
    const acceptance = { passed: true, mode: 'actual_browser_saved_public_video', checks,
      workspace: original.workspace, original_execution_id: oracle.execution_id,
      summary_run_id: summary.id, mobile_summary_run_id: mSummary.id, broader_run_id: broader.id,
      tested_run_ids: runIds, new_vision_calls: 0, real_llm_calls: 0, mock_ui_or_cv_results: false,
      responsive_widths: [1440,390], browser_errors: errors, external_requests: external.length };
    fs.writeFileSync(path.join(output, 'ui-acceptance.json'), JSON.stringify(acceptance, null, 2));
    console.log(JSON.stringify({ passed: true, checks, new_vision_calls: 0, real_llm_calls: 0 }));
  } catch (e) {
    for (const [i, context] of browser.contexts().entries()) {
      await context.pages()[0]?.screenshot({ path: path.join(output, `failure-${i}-${Date.now()}.png`), fullPage: true }).catch(() => {});
    }
    throw e;
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
