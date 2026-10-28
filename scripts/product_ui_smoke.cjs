/* Product-facing UI regression with real saved CPR evidence. No mocked responses or new vision work. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const output = path.join(root, process.env.CPR_UI_TEST_OUTPUT || 'test-results/product-ui-v1/ui');
const baseURL = 'http://127.0.0.1:8005';
async function main() {
  fs.mkdirSync(output, { recursive: true });
  assert(!fs.existsSync(path.join(output,'acceptance.json')), 'Preserve accepted evidence');
  const record=JSON.parse(fs.readFileSync(path.join(root,'test-results/demo-acceptance-v1/http/acceptance.json'),'utf8'));
  const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
  const errors=[], external=[], checks=[], runs=[];
  try {
    async function context(width=1440) {
      const c=await browser.newContext({baseURL,viewport:{width,height:1000},reducedMotion:'reduce'});
      c.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));
      c.on('request',r=>{if(new URL(r.url()).origin!==baseURL)external.push(r.url());});
      return c;
    }
    const c=await context(), page=await c.newPage();
    const get=async url=>{const r=await c.request.get(url);assert(r.ok(),url);return r.json();};
    const caps=await get('/api/capabilities'); assert.equal(caps.llm.mode,'disabled');
    const currentId=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('cpr.runId')));
    const findings=()=>page.getByRole('region',{name:'Practice findings',exact:true});
    const assistant=()=>page.getByRole('complementary',{name:'Practice review assistant'});
    const explanation=()=>page.getByLabel('Evidence-constrained explanation',{exact:true});
    async function noOverflow(p=page){assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'No horizontal page overflow');}
    async function seed(ws,id,p=page){await p.evaluate(({ws,id})=>{localStorage.setItem('cpr.workspace',JSON.stringify(ws));localStorage.setItem('cpr.runId',JSON.stringify(id));},{ws,id});await p.reload();}
    async function waitNew(previous,expected='completed') {
      await page.waitForFunction(old=>{const id=JSON.parse(localStorage.getItem('cpr.runId'));return id&&id!==old;},previous);
      const id=await currentId();let run;
      for(let i=0;i<150;i++){run=await get(`/api/runs/${id}`);if(run.status===expected)break;if(['failed','rejected','cancelled'].includes(run.status))throw new Error('Unexpected run '+run.status);await page.waitForTimeout(100);}
      assert.equal(run.status,expected);assert.deepEqual(await get(`/api/runs/${id}/tools`),[]);runs.push(id);
      if(expected==='completed'&&run.response?.prediction){
        assert.deepEqual(run.response.prediction,record.prediction);
        await page.locator('.export-review').filter({has:page.getByText('Export review',{exact:true})}).waitFor();
        await page.waitForFunction(report=>document.querySelector('.export-review')?.getAttribute('href')===`/api/reports/${report}/export`,run.report_id);
        await explanation().getByRole('heading').waitFor();
      }
      return run;
    }
    async function selectFocus(ids,region=page.locator('.focus-drawer')) {
      await region.getByRole('button',{name:'Clear',exact:true}).click();
      for(const id of ids)await region.getByRole('button',{name:caps.label_names[id],exact:true}).click();
    }
    await page.goto('/');await page.getByRole('heading',{name:'Start your practice review.',exact:true}).waitFor();
    await page.getByRole('button',{name:'Choose video',exact:true}).waitFor({state:'visible'});
    assert(await page.getByRole('button',{name:'Start review',exact:true}).isDisabled());
    assert(await page.getByRole('button',{name:'Send message',exact:true}).isDisabled());
    assert(!await page.locator('dialog').isVisible());
    assert(!/N0[1-9]|N1[0-2]|baseline-v|M1|M2|tool execution|LangGraph|API documentation/.test(await page.locator('body').innerText()));
    await page.screenshot({path:path.join(output,'01-start-1440.png'),fullPage:true});
    await page.locator('input[type=file]').setInputFiles({name:'broken.mp4',mimeType:'video/mp4',buffer:Buffer.from('not a video')});
    await page.getByRole('alert').filter({hasText:'This file could not be opened as a video.'}).waitFor();
    assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('cpr.workspace'))),null);
    await page.getByRole('button',{name:'Dismiss error',exact:true}).click();
    checks.push('Task-oriented empty state, guarded actions, private technical details, actionable invalid-file error');

    await page.locator(`[data-session-id="${record.workspace.session_id}"]`).click();
    async function openRecord(p,id) {
      const h=p.locator('.session-history');
      if(!await h.evaluate(e=>e.open))await h.locator('summary').click();
      const item=h.locator(`[data-run-id="${id}"] button`);
      for(let n=0;n<20&&await item.count()===0;n++){const more=h.getByRole('button',{name:'Load older reviews',exact:true});assert(await more.count(),'History record still reachable');await more.click();await p.waitForTimeout(100);}
      await item.click();
    }
    await openRecord(page,record.analysis_run_id);
    await findings().waitFor();await page.getByText('Saved source checked',{exact:false}).waitFor();
    await explanation().getByRole('heading').waitFor();
    const original=await currentId();
    await page.getByRole('button',{name:/^Focus:/}).click();await selectFocus([2,4]);
    await page.screenshot({path:path.join(output,'02-focus-picker-1440.png'),fullPage:true});
    await page.getByRole('button',{name:'Show selected focus',exact:true}).click();
    const scoped=await waitNew(original);assert.deepEqual(scoped.response.classification_claims.map(x=>x.label_id),[2,4]);
    await findings().getByRole('button',{name:'Explain Bent arms',exact:true}).click();
    const explained=await waitNew(scoped.id);assert.deepEqual(explained.response.classification_claims.map(x=>x.label_id),[4]);
    await findings().getByRole('button',{name:'Make a recap',exact:true}).click();
    const summary=await waitNew(explained.id);await explanation().getByRole('heading',{name:'Your review recap',exact:true}).waitFor();
    const exported=await c.request.get(`/api/reports/${summary.report_id}/export`);assert(exported.ok());assert((await exported.text()).includes('Evidence-constrained review'));
    await noOverflow();await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(output,'03-review-1440.png'),fullPage:true});
    checks.push('Saved video → selected focus → evidence explanation → scoped recap and real export; unchanged prediction, zero vision calls');

    await page.getByRole('button',{name:/^Focus:/}).click();await selectFocus([4]);
    await page.getByRole('button',{name:'Save focus',exact:true}).click();const remembered=await waitNew(summary.id);
    await assistant().getByText('Focus updated',{exact:true}).waitFor();await findings().waitFor();
    assert.equal(await page.locator('.export-review').getAttribute('href'),`/api/reports/${summary.report_id}/export`);
    let prefs=await get('/api/preferences');assert.deepEqual(prefs[0].label_ids,[4]);
    await page.getByRole('button',{name:/^Focus:/}).click();await selectFocus([2]);await page.getByRole('button',{name:'Use saved focus',exact:true}).click();
    assert.equal(await page.locator('.focus-drawer button[aria-pressed="true"]').count(),1);
    assert.equal((await page.locator('.focus-drawer button[aria-pressed="true"]').innerText()).replace(/\s+/g,''),'✓Bentarms');
    await page.getByRole('button',{name:'Forget focus',exact:true}).click();const forgotten=await waitNew(remembered.id);
    await assistant().getByText('Focus updated',{exact:true}).waitFor();assert.deepEqual(await get('/api/preferences'),[]);await findings().waitFor();
    checks.push('Explicit save, use and forget focus; preference receipts retain the clearly labelled saved findings');

    await page.getByLabel('Your question',{exact:true}).fill('Explain this finding.');await page.getByRole('button',{name:'Send message',exact:true}).click();
    const waiting=await waitNew(forgotten.id,'waiting_user');
    const form=page.getByRole('region',{name:'Confirm your review',exact:true});await form.waitFor();
    await form.getByLabel('I would like to',{exact:true}).selectOption('explain');
    await selectFocus([4],form);
    const sourceCheck=form.locator('input[type=checkbox]');
    if(await sourceCheck.count()){assert(await form.getByRole('button',{name:'Continue review',exact:true}).isDisabled());await sourceCheck.check();}
    await form.getByRole('button',{name:'Continue review',exact:true}).click();
    for(let i=0;i<120;i++){if((await get(`/api/runs/${waiting.id}`)).status==='completed')break;await page.waitForTimeout(100);}
    assert.equal((await get(`/api/runs/${waiting.id}`)).status,'completed');
    await explanation().getByRole('heading').waitFor();
    await page.getByLabel('Your question',{exact:true}).fill('Let me choose again.');await page.getByRole('button',{name:'Send message',exact:true}).click();
    const cancelRun=await waitNew(waiting.id,'waiting_user');await form.getByRole('button',{name:'Cancel request',exact:true}).click();
    await assistant().getByText('This request was cancelled.',{exact:false}).waitFor();assert.equal((await get(`/api/runs/${cancelRun.id}`)).response,null);await findings().waitFor();
    checks.push('Disabled language model uses honest structured confirmation; source confirmation gate, exact resume and cancellation work');

    const failed=await get(`/api/runs/${record.failed_decode_run_id}`);
    await page.locator(`[data-session-id="${failed.session_id}"]`).click();
    await page.getByRole('heading',{name:"We couldn't complete this review.",exact:true}).waitFor();
    assert.equal(await findings().count(),0);assert.equal(await explanation().count(),0);
    await page.screenshot({path:path.join(output,'04-unavailable-review.png'),fullPage:true});
    await page.locator(`[data-session-id="${record.workspace.session_id}"]`).click();
    await openRecord(page,summary.id);await explanation().getByRole('heading',{name:'Your review recap',exact:true}).waitFor();
    await page.reload();await explanation().getByRole('heading',{name:'Your review recap',exact:true}).waitFor();
    assert.equal(await currentId(),summary.id);
    checks.push('Failure cannot display previous findings; per-practice history reopens a recap and refresh restores the same source');

    await page.getByRole('button',{name:'Technical details',exact:true}).click();const dialog=page.getByRole('dialog',{name:'Technical details',exact:true});await dialog.waitFor();
    await dialog.getByText('State machine and execution',{exact:true}).click();assert.equal(await dialog.locator('.technical-nodes li').count(),12);
    await page.screenshot({path:path.join(output,'05-technical-details.png')});await page.keyboard.press('Escape');assert(!await dialog.isVisible());
    assert.equal(await page.evaluate(()=>document.activeElement?.textContent?.trim()),'Technical details');
    checks.push('Actual 12-node execution information is available only in an explicit read-only dialog; Escape restores focus');

    const mobile=await context(390), mp=await mobile.newPage();await mp.goto('/');await mp.getByRole('button',{name:'Open sessions',exact:true}).click();
    await mp.locator(`[data-session-id="${record.workspace.session_id}"]`).click();await mp.getByRole('button',{name:'Open sessions',exact:true}).waitFor({state:'visible'});
    assert(await mp.locator('.app-sidebar').evaluate(el=>el.inert));
    // Latest request is cancelled: reopen the saved recap through the explicit history.
    await mp.getByRole('button',{name:'Open sessions',exact:true}).click();await openRecord(mp,summary.id);await mp.getByRole('heading',{name:'Your review recap',exact:true}).waitFor();
    await mp.getByRole('button',{name:'Open sessions',exact:true}).click();assert(await mp.locator('.main-shell').evaluate(el=>el.inert));
    await mp.keyboard.press('Escape');assert(await mp.locator('.app-sidebar').evaluate(el=>el.inert));
    const video=mp.getByLabel('Practice video player',{exact:true});await video.scrollIntoViewIfNeeded();await video.evaluate(async e=>{e.muted=true;await e.play();});await mp.waitForTimeout(650);
    const playback=await video.evaluate(e=>{e.pause();return{time:e.currentTime,width:e.videoWidth,height:e.videoHeight,error:e.error?.code||null};});assert(playback.time>0&&playback.width>0&&!playback.error);
    await mp.getByRole('navigation',{name:'Review sections'}).getByRole('link',{name:'Assistant',exact:true}).click();
    await noOverflow(mp);await mp.screenshot({path:path.join(output,'06-review-390.png'),fullPage:true});
    for(const width of [768,1024]){await mp.setViewportSize({width,height:1000});await mp.waitForTimeout(100);await noOverflow(mp);await mp.screenshot({path:path.join(output,`07-review-${width}.png`),fullPage:true});}
    checks.push('390 / 768 / 1024 / 1440 px layouts, keyboard-safe mobile drawer, section shortcuts and actual video playback');

    // New uploads do not inherit the findings or assistant reply from the previous practice.
    await page.locator('input[type=file]').setInputFiles(path.join(root,'data/public_cpr/rgb-acceptance-v1/records/SupC04--r00--ch0/video.mp4'));
    await page.waitForFunction(old=>JSON.parse(localStorage.getItem('cpr.workspace'))?.session_id!==old,record.workspace.session_id);
    await page.getByRole('button',{name:'Start review',exact:true}).waitFor();await page.waitForFunction(()=>Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='Start review'&&!b.disabled));
    assert.equal(await findings().count(),0);assert.equal(await explanation().count(),0);assert.equal(await currentId(),null);
    assert.equal(await assistant().locator('.user-message').count(),0);
    checks.push('Adding a real new clip clears previous context, findings and conversation; upload alone never starts analysis');

    assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
    fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify({passed:true,checks,workspace:record.workspace,final_summary_run_id:summary.id,final_summary_report_id:summary.report_id,tested_run_ids:runs,real_llm_calls:0,mocked_ui_or_cv:false,widths:[390,768,1024,1440],mobile_playback:playback,errors,external_requests:external.length},null,2));
    console.log(JSON.stringify({passed:true,checks,real_llm_calls:0}));
  } catch(e) {
    for(const [i,c] of browser.contexts().entries())await c.pages()[0]?.screenshot({path:path.join(output,`failure-${i}-${Date.now()}.png`),fullPage:true}).catch(()=>{});
    throw e;
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
