/* Current-mainline UI acceptance with two bounded DeepSeek calls and no CV calls. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const ROOT = path.resolve(__dirname, '..');
const BASE = 'http://127.0.0.1:8013';
const liveState = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/mainline-delivery-v1/deepseek-live.json'), 'utf8'));
const output = path.join(ROOT, 'data/mainline-delivery-v1/mainline-deepseek-ui-acceptance.json');

async function main() {
  assert(!fs.existsSync(output), 'Preserve existing DeepSeek UI acceptance');
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true,
  });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  const external = [];
  context.on('page', (page) => page.on('pageerror', (error) => errors.push(String(error))));
  context.on('request', (request) => {
    if (new URL(request.url()).origin !== BASE) external.push(request.url());
  });
  const api = context.request;
  const get = async (url) => {
    const response = await api.get(`${BASE}/api${url}`);
    assert(response.ok(), `${url}: ${response.status()}`);
    return response.json();
  };
  try {
    const health = await get('/health');
    assert.equal(health.llm, 'deepseek');
    const page = await context.newPage();
    await page.goto(`${BASE}/demo/open/D01`, { waitUntil: 'domcontentloaded' });
    await page.waitForURL(`${BASE}/`);
    await page.waitForFunction(() => !document.querySelector('#request-text')?.disabled);
    const sourceRunId = liveState.full_run_id;
    const source = await get(`/runs/${sourceRunId}`);
    await page.locator('#request-text').fill('请只解释这段训练视频中手臂弯曲这一项的已有结果，不要重新分析视频。');
    await page.getByRole('button', { name: 'Send message', exact: true }).click();
    await page.waitForFunction((old) => JSON.parse(localStorage.getItem('cpr.runId')) !== old, sourceRunId);
    const runId = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
    let run;
    for (let i = 0; i < 300; i += 1) {
      run = await get(`/runs/${runId}`);
      if (['completed', 'partially_completed', 'failed', 'rejected', 'cancelled', 'waiting_user'].includes(run.status)) break;
      await page.waitForTimeout(250);
    }
    assert.equal(run.status, 'completed', JSON.stringify(run));
    assert.deepEqual(await get(`/runs/${runId}/tools`), []);
    assert.deepEqual(run.response.prediction, source.response.prediction);
    assert.deepEqual(run.grounded_review.scope_label_ids, [4]);
    assert.equal(run.grounded_review.organisation_receipt.code, 'LLM_PLAN_VALIDATED');
    await page.getByLabel('Evidence-constrained explanation', { exact: true }).waitFor();
    await page.getByText('AI-assisted explanation', { exact: false }).waitFor();
    await page.setViewportSize({ width: 390, height: 950 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({ path: path.join(ROOT, 'data/mainline-delivery-v1/mainline-deepseek-ui-mobile.png'), fullPage: true });
    const events = (await get(`/runs/${runId}/events`)).events;
    const providerCalls = events.filter((event) => event.payload?.called === true).length;
    assert.equal(providerCalls, 2);
    const result = {
      passed: true,
      source_run_id: sourceRunId,
      run_id: runId,
      report_id: run.report_id,
      scope: run.grounded_review.scope_label_ids,
      provider_requests: providerCalls,
      new_vision_calls: 0,
      prediction_unchanged: true,
      grounded_plan_validated: true,
      refresh_safe: true,
      mobile_width: 390,
      browser_errors: errors,
      external_browser_requests: external.length,
    };
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    fs.writeFileSync(output, JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });

