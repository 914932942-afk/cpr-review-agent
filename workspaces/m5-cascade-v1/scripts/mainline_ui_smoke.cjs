/* Actual browser workflow with one public video; no LLM calls or training. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const root = path.resolve(__dirname, '..');
const output = path.join(root, 'test-results/mainline-product-v1');
const acceptance = path.join(output, 'ui-acceptance.json');
const baseURL = process.env.CPR_SMOKE_URL || 'http://127.0.0.1:8004';
const terminal = ['completed', 'partially_completed', 'failed', 'rejected', 'cancelled'];
const reuseVerifiedVideo = process.argv.includes('--resume-verified-video');

async function main() {
  assert(!fs.existsSync(acceptance), 'Preserve prior acceptance results');
  fs.mkdirSync(output, { recursive: true });
  const errors = [], external = [], posts = [], checks = [];
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true,
  });
  try {
    async function context(width) {
      const ctx = await browser.newContext({ baseURL, viewport: { width, height: 1000 } });
      ctx.on('page', page => page.on('pageerror', e => errors.push(e.message)));
      await ctx.route('**/*', route => {
        const url = new URL(route.request().url());
        if (url.origin !== baseURL) { external.push(url.origin); return route.abort(); }
        if (route.request().method() === 'POST') posts.push(url.pathname);
        return route.continue();
      });
      return ctx;
    }
    const ctx = await context(1440), page = await ctx.newPage();
    const api = async url => {
      const res = await ctx.request.get(url); assert(res.ok(), `${url}: ${res.status()}`); return res.json();
    };
    const caps = await api('/api/capabilities');
    assert.equal(caps.llm.mode, 'disabled');
    assert.deepEqual(Object.fromEntries(Object.entries(caps.module_slots).map(([k, v]) => [k, v.kind])),
      { M2: 'baseline', M3: 'baseline', M4: 'baseline', M5: 'baseline' });
    const currentRunId = () => page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
    async function settled(previous, expected = 'completed', timeout = 20000) {
      await page.waitForFunction(prev => {
        const id = JSON.parse(localStorage.getItem('cpr.runId'));
        return id && id !== prev;
      }, previous, { timeout: 15000 });
      const id = await currentRunId(), until = Date.now() + timeout;
      let run;
      do {
        run = await api(`/api/runs/${id}`);
        if (terminal.includes(run.status)) break;
        await page.waitForTimeout(300);
      } while (Date.now() < until);
      assert.equal(run.status, expected, JSON.stringify({ id, status: run.status, error: run.error_code }));
      if (run.response) {
        await page.getByRole('region', { name: 'Workflow result', exact: true }).getByRole('heading', { name: run.response.title, exact: true }).waitFor();
      } else {
        await page.getByLabel('Analysis progress', { exact: true }).getByText('Request cancelled', { exact: true }).waitFor();
      }
      return run;
    }
    const result = page.getByRole('region', { name: 'Workflow result', exact: true });
    await page.goto('/', { waitUntil: 'networkidle' });
    assert(await page.getByRole('button', { name: 'Review selected categories →', exact: true }).isDisabled());
    assert.equal(await page.getByRole('button', { name: 'Load workflow fixture', exact: true }).count(), 0);
    await page.getByRole('button', { name: `4: ${caps.label_names[4]}`, exact: true }).waitFor();
    checks.push('No-video actions guarded; real mode does not load a synthetic fixture');

    const video = path.join(root, 'data/public_cpr/rgb-acceptance-v1/records/SupC04--r00--ch0/video.mp4');
    assert(fs.existsSync(video));
    let verified;
    if (reuseVerifiedVideo) {
      verified = JSON.parse(fs.readFileSync(path.join(output, 'real-video-progress.json'), 'utf8'));
      await page.evaluate(record => {
        localStorage.setItem('cpr.workspace', JSON.stringify(record.workspace));
        localStorage.setItem('cpr.runId', JSON.stringify(record.run_id));
      }, verified);
      await page.reload({ waitUntil: 'networkidle' });
      await result.getByRole('heading', { name: 'Clip-level review', exact: true }).waitFor();
    } else {
      await page.locator('input[type=file]').setInputFiles(video);
      await page.getByRole('button', { name: 'Review selected categories →', exact: true }).waitFor();
      await page.waitForFunction(() => JSON.parse(localStorage.getItem('cpr.workspace'))?.asset_id);
    }
    await page.getByRole('button', { name: 'Clear selection', exact: true }).click();
    assert(await page.getByRole('button', { name: 'Review selected categories →', exact: true }).isDisabled());
    await page.getByRole('button', { name: `4: ${caps.label_names[4]}`, exact: true }).click();
    let analysis;
    if (reuseVerifiedVideo) analysis = await api(`/api/runs/${verified.run_id}`);
    else {
      await page.getByRole('button', { name: 'Review selected categories →', exact: true }).click();
      analysis = await settled(null, 'completed', 240000);
    }
    const workspace = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.workspace')));
    const tools = await api(`/api/runs/${analysis.id}/tools`);
    assert.equal(tools.length, 4);
    assert(tools.every(t => t.status === 'succeeded'));
    assert.deepEqual(analysis.response.prediction.predicted_label_ids, [4]);
    assert.equal(analysis.response.reused_result, false);
    assert.equal(await page.locator('.analysis-stages .succeeded').count(), 4);
    const oracle = JSON.parse(fs.readFileSync(path.join(root, 'test-results/module-slots-v1/http-final-acceptance.json'), 'utf8')).cases[0];
    const old = await api(`/api/runs/${oracle.run_id}`);
    const delta = Math.max(...analysis.response.prediction.scores.map((s, i) => Math.abs(s.uncalibrated_score - old.response.prediction.scores[i].uncalibrated_score)));
    assert(delta < 1e-8);
    if (!reuseVerifiedVideo) fs.writeFileSync(path.join(output, 'real-video-progress.json'), JSON.stringify({ workspace, run_id: analysis.id, report_id: analysis.report_id, maximum_score_difference: delta }, null, 2));
    console.log(JSON.stringify({ stage: 'real-video-passed', run_id: analysis.id, maximum_score_difference: delta }));
    checks.push('Actual browser upload and four real ML stages; frozen prediction unchanged');
    assert.equal(await page.locator('.engineering-details').getAttribute('open'), null);
    await result.getByText('Verified source · whole-clip record', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'workspace-1440.png'), fullPage: true });

    await page.getByRole('button', { name: 'Remember focus', exact: true }).click();
    const memory = await settled(analysis.id);
    assert.deepEqual((await api('/api/preferences'))[0].label_ids, [4]);
    assert.deepEqual(await api(`/api/runs/${memory.id}/tools`), []);
    await page.getByRole('button', { name: 'Clear selection', exact: true }).click();
    await page.getByRole('button', { name: 'Use saved focus', exact: true }).click();
    assert.deepEqual(await page.locator('.target-options button[aria-pressed=true]').allTextContents(), [`04${caps.label_names[4]}`]);
    checks.push('Explicit focus saved and reapplied without inference');

    await page.getByRole('button', { name: 'Use existing results only', exact: true }).click();
    const reused = await settled(memory.id);
    assert.equal(reused.response.reused_result, true);
    assert.equal(reused.response.prediction.execution_id, analysis.response.prediction.execution_id);
    assert.deepEqual(await api(`/api/runs/${reused.id}/tools`), []);
    await result.getByRole('button', { name: 'Review this category', exact: true }).click();
    const explain = await settled(reused.id);
    assert.deepEqual(explain.response.review.displayed_label_ids, [4]);
    assert.deepEqual(await api(`/api/runs/${explain.id}/tools`), []);
    checks.push('Category follow-up keeps the same clip, scope and evidence; zero new ML calls');

    await page.locator('.request-details > summary').click();
    await page.getByLabel('Your request', { exact: true }).fill('Explain the movement category in this review.');
    await page.getByRole('button', { name: 'Choose task in a form →', exact: true }).click();
    const form = page.getByRole('region', { name: 'Structured goal form', exact: true });
    await form.waitFor();
    await form.getByLabel('Intent', { exact: true }).selectOption('explain');
    await page.getByRole('button', { name: 'Clear selection', exact: true }).click();
    await page.getByRole('button', { name: `4: ${caps.label_names[4]}`, exact: true }).click();
    const confirmation = form.locator('input[type=checkbox]');
    if (await confirmation.count()) await confirmation.check();
    await form.getByRole('button', { name: 'Continue from checkpoint →', exact: true }).click();
    const formRun = await settled(explain.id);
    assert.equal(formRun.response.reused_result, true);
    assert.deepEqual(await api(`/api/runs/${formRun.id}/tools`), []);
    checks.push('Disabled LLM falls back to editable goal confirmation and resumes with saved evidence');

    await page.getByRole('button', { name: 'Summarise record', exact: true }).click();
    const summary = await settled(formRun.id);
    assert.deepEqual(await api(`/api/runs/${summary.id}/tools`), []);
    assert.equal(summary.response.review.source_execution_id, analysis.response.prediction.execution_id);
    const href = await result.getByRole('link', { name: 'Open printable review report ↗', exact: true }).getAttribute('href');
    const report = await ctx.newPage();
    await report.goto(href, { waitUntil: 'networkidle' });
    await report.getByRole('heading', { name: 'Review summary', exact: true }).waitFor();
    assert.match(await report.locator('.source-note').textContent(), /Source files and bindings verified/);
    assert.equal(await report.locator('tbody tr').count(), 11);
    assert.match(await report.locator('body').textContent(), /not medical probabilities/);
    await report.screenshot({ path: path.join(output, 'report-1440.png'), fullPage: true });
    fs.writeFileSync(path.join(output, 'report-preview.html'), await report.content());
    await report.close();
    await page.reload({ waitUntil: 'networkidle' });
    await result.getByRole('heading', { name: 'Review summary', exact: true }).waitFor();
    assert.equal(await currentRunId(), summary.id);
    checks.push('Readable report export and refresh recovery preserve the saved summary');

    // A waiting-user run can be cancelled without creating a new analysis.
    await page.locator('.request-details > summary').click();
    await page.getByRole('button', { name: 'Choose task in a form →', exact: true }).click();
    await form.waitFor();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    const cancelled = await settled(summary.id, 'cancelled');
    assert.deepEqual(await api(`/api/runs/${cancelled.id}/tools`), []);
    assert.match(await page.getByLabel('Analysis progress', { exact: true }).textContent(), /Request cancelled/);
    checks.push('Waiting request cancellation terminates cleanly without inference');

    // Recover on a new browser with no localStorage. Open the saved summary explicitly.
    const mobile = await context(390), mp = await mobile.newPage();
    await mp.goto('/', { waitUntil: 'networkidle' });
    const select = mp.getByLabel('Saved session', { exact: true });
    await select.locator(`option[value="${workspace.session_id}"]`).waitFor({ state: 'attached' });
    await select.selectOption(workspace.session_id);
    await mp.locator('.history-list li').filter({ hasText: 'Review summary' }).first().getByRole('button', { name: 'Open record', exact: true }).click();
    const mr = mp.getByRole('region', { name: 'Workflow result', exact: true });
    await mr.getByRole('heading', { name: 'Review summary', exact: true }).waitFor();
    await mr.getByText('View recorded evidence for #4', { exact: true }).click();
    await mp.getByRole('button', { name: 'Use saved focus', exact: true }).click();
    assert.equal(await mp.locator('.target-options button[aria-pressed=true]').count(), 1);
    assert.equal(await mp.getByLabel('Input video', { exact: true }).getAttribute('src'), `/api/assets/${workspace.asset_id}/media`);
    assert.equal(await mp.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await mp.screenshot({ path: path.join(output, 'workspace-390.png'), fullPage: true });
    await mp.locator('.engineering-details > summary').click();
    assert.equal(await mp.getByText('Baseline connected · member algorithm pending', { exact: true }).count(), 4);
    assert.equal(await mp.locator('.node-grid li').count(), 12);
    checks.push('Fresh mobile browser restores video, report, evidence, preference and 12-node trace');
    assert.equal(posts.filter(p => p === '/api/demo/session').length, 0);
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    assert.equal((await api('/api/health')).llm, 'disabled');
    fs.writeFileSync(acceptance, JSON.stringify({ passed: true, mode: reuseVerifiedVideo ? 'actual_browser_reusing_prior_real_video_acceptance' : 'actual_browser_and_real_video',
      checks, workspace, run_id: analysis.id, report_id: analysis.report_id,
      summary_run_id: summary.id, cancelled_run_id: cancelled.id,
      maximum_score_difference: delta, real_video_analyses: 1,
      new_video_analyses_this_script_run: reuseVerifiedVideo ? 0 : 1,
      followup_tool_calls: 0, llm_calls: 0, module_algorithms_changed: false,
      widths: [1440, 390], browser_errors: errors, external_browser_requests: external.length,
    }, null, 2));
    console.log(JSON.stringify({ passed: true, checks }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
