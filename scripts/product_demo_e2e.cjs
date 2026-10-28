/* Real product acceptance: one new public video, real CV and explicitly allowed live DeepSeek. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),output=path.join(root,'test-results/product-demo-v2/browser');
const baseURL='http://127.0.0.1:8007',restore=process.argv.includes('--restore');
const terminal=['completed','partially_completed','failed','rejected','cancelled'];
fs.mkdirSync(output,{recursive:true});
const statePath=path.join(output,'run-state.json');
const state=fs.existsSync(statePath)?JSON.parse(fs.readFileSync(statePath,'utf8')):{runs:[],stages_observed:[],browser_errors:[],external_requests:[]};
const save=()=>fs.writeFileSync(statePath,JSON.stringify(state,null,2));
const json=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
async function main(){
 assert(restore||process.argv.includes('--allow-live'),'Explicit --allow-live is required');
 assert(!fs.existsSync(path.join(output,restore?'restore.json':'acceptance.json')),'Do not overwrite acceptance');
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 try{
  const c=await browser.newContext({baseURL,viewport:{width:1440,height:1000},reducedMotion:'reduce'});
  c.on('page',p=>p.on('pageerror',e=>{state.browser_errors.push(e.message);save();}));
  c.on('request',r=>{if(new URL(r.url()).origin!==baseURL)state.external_requests.push(r.url());});
  const get=async url=>{const r=await c.request.get('/api'+url);assert(r.ok(),url);return r.json();};
  const caps=await get('/capabilities');assert.equal(caps.mode,'experimental');
  const page=await c.newPage();await page.goto('/');
  const current=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('cpr.runId')));
  const assistant=()=>page.getByRole('complementary',{name:'Practice review assistant'});
  const findings=()=>page.getByRole('region',{name:'Practice findings',exact:true});
  const explanation=()=>page.getByLabel('Evidence-constrained explanation',{exact:true});
  async function picture(name){await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(output,name+'.png'),fullPage:true});}
  async function seed(ws,id){await page.evaluate(({ws,id})=>{localStorage.setItem('cpr.workspace',JSON.stringify(ws));localStorage.setItem('cpr.runId',JSON.stringify(id));},{ws,id});await page.reload();}
  async function waitRun(id,seconds=35){
   const start=Date.now();let last='',refreshed=false;
   while(Date.now()-start<seconds*1000){
    const run=await get('/runs/'+id),tools=await get('/runs/'+id+'/tools');
    const marker=run.status+':'+tools.map(t=>t.request.tool_id+'='+t.status).join(',');
    if(marker!==last){console.log(marker);last=marker;}
    for(const t of tools){const value=t.request.tool_id+':'+t.status;if(!state.stages_observed.includes(value))state.stages_observed.push(value);}
    if(!restore&&id===state.analysis_run_id&&tools.some(t=>t.status==='running')&&!state.running_refresh_verified&&!refreshed){
     await page.getByLabel('Analysis progress',{exact:true}).waitFor();await picture('02-real-analysis-running');
     assert(await page.getByRole('button',{name:'Change video',exact:true}).isDisabled());
     await page.reload();assert.equal(await current(),id);state.running_refresh_verified=true;save();refreshed=true;
    }
    if(terminal.includes(run.status)){assert.equal(run.status,'completed',JSON.stringify({status:run.status,code:run.error_code}));return run;}
    if(run.status==='waiting_user')throw new Error('Live request requires confirmation: '+JSON.stringify(run.wait));
    await page.waitForTimeout(500);
   }
   throw new Error('Timed out for '+id);
  }
  async function newRun(previous,action){await action();await page.waitForFunction(old=>{const id=JSON.parse(localStorage.getItem('cpr.runId'));return id&&id!==old;},previous);const id=await current();if(!state.runs.includes(id))state.runs.push(id);save();return waitRun(id);}
  async function shown(run){
   await page.waitForFunction(id=>document.querySelector('.export-review')?.getAttribute('href')===`/api/reports/${id}/export`,run.report_id);
   await explanation().getByRole('heading').waitFor();await page.getByText('Saved source checked',{exact:false}).waitFor();
  }
  if(restore){
   const beforeCalls=state.provider_requests;
   const workspaces=await get('/workspaces');assert.equal(workspaces.length,1);
   await page.locator(`[data-session-id="${state.workspace.session_id}"]`).click();
   if(await current()!==state.summary_run_id){const latest=assistant().getByRole('button',{name:'Latest recap',exact:true});await latest.click();}
   await explanation().getByRole('heading',{name:'Your review recap',exact:true}).waitFor();assert.equal(await current(),state.summary_run_id);
   await page.reload();await explanation().getByRole('heading',{name:'Your review recap',exact:true}).waitFor();
   assert.deepEqual((await get('/runs/'+state.summary_run_id)).response.prediction,state.prediction);
   assert.equal((await get('/reports/'+state.summary_report_id+'/explanation')).status,'available');
   const video=page.getByLabel('Practice video player',{exact:true});await video.evaluate(async e=>{e.muted=true;await e.play();});await page.waitForTimeout(700);
   const playback=await video.evaluate(e=>{e.pause();return{time:e.currentTime,width:e.videoWidth,height:e.videoHeight,error:e.error?.code||null};});assert(playback.time>0&&playback.width>0&&!playback.error);
   await picture('09-after-service-restart');
   let calls=0;for(const id of state.runs){for(const e of (await get('/runs/'+id+'/events')).events)if(e.payload?.called===true)calls++;}
   assert.equal(calls,beforeCalls);assert.equal((await get('/runs/'+state.analysis_run_id+'/tools')).length,4);
   fs.writeFileSync(path.join(output,'restore.json'),JSON.stringify({passed:true,service_restarted:true,fresh_browser_restored_from_server_history:true,playback,new_paid_calls:0,new_vision_calls:0},null,2));
   console.log('Restart and fresh-browser recovery passed');return;
  }
  assert.equal(caps.llm.mode,'deepseek');
  if(!state.workspace){
   assert.equal((await get('/workspaces')).length,0,'Use the separate clean demo data');
   await page.getByRole('heading',{name:'Start your practice review.',exact:true}).waitFor();
   await page.locator('input[type=file]').setInputFiles(path.join(root,'data/public_cpr/rgb-acceptance-v1/records/SupC04--r00--ch0/video.mp4'));
   await page.waitForFunction(()=>!!JSON.parse(localStorage.getItem('cpr.workspace'))?.asset_id);
   state.workspace=await page.evaluate(()=>JSON.parse(localStorage.getItem('cpr.workspace')));save();
   await page.waitForFunction(()=>Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='Start review'&&!b.disabled));
   assert.equal(await findings().count(),0);await picture('01-video-uploaded');
   await page.getByRole('button',{name:'Start review',exact:true}).click();
   await page.waitForFunction(()=>!!JSON.parse(localStorage.getItem('cpr.runId')));state.analysis_run_id=await current();state.runs.push(state.analysis_run_id);save();
  }else await seed(state.workspace,state.analysis_run_id);
  const started=Date.now(),analysed=await waitRun(state.analysis_run_id,720);await shown(analysed);
  const vision=await get('/runs/'+analysed.id+'/tools');assert.equal(vision.length,4);
  assert.deepEqual(vision.map(t=>t.request.tool_id),['pose.extract','skeleton.predict','rgb.predict','fusion.predict']);
  assert(vision.every(t=>t.status==='succeeded'&&!t.output?.is_stub));
  const reference=json('test-results/demo-acceptance-v1/http/acceptance.json').prediction;
  for(const k of ['video_sha256','bundle_sha256','decoded_frames','reported_fps','predicted_label_ids'])assert.deepEqual(analysed.response.prediction[k],reference[k]);
  state.prediction=analysed.response.prediction;state.analysis_report_id=analysed.report_id;state.analysis_wait_ms=Date.now()-started;state.vision_duration_ms=vision.reduce((n,t)=>n+(t.duration_ms||0),0);save();
  await picture('03-first-real-findings');
  const one=await newRun(analysed.id,()=>findings().getByRole('button',{name:'Explain Bent arms',exact:true}).click());await shown(one);assert.deepEqual(one.response.classification_claims.map(x=>x.label_id),[4]);
  const continued=await newRun(one.id,async()=>{await page.getByLabel('Your question',{exact:true}).fill('Continue explaining that same finding using the saved result. Do not analyse the video again.');await page.getByRole('button',{name:'Send message',exact:true}).click();});
  await shown(continued);assert.deepEqual(continued.response.classification_claims.map(x=>x.label_id),[4]);assert.equal(continued.grounded_review.mode,'model_organised_evidence');assert.deepEqual(continued.response.prediction,state.prediction);
  await picture('04-live-context-follow-up');
  const summary=await newRun(continued.id,()=>findings().getByRole('button',{name:'Make a recap',exact:true}).click());await shown(summary);await explanation().getByRole('heading',{name:'Your review recap',exact:true}).waitFor();
  state.summary_run_id=summary.id;state.summary_report_id=summary.report_id;save();
  const [popup]=await Promise.all([page.waitForEvent('popup'),page.getByRole('link',{name:'Export review',exact:true}).click()]);await popup.waitForLoadState('domcontentloaded');assert((await popup.locator('body').innerText()).includes('Evidence-constrained review'));await popup.screenshot({path:path.join(output,'05-printable-report.png'),fullPage:true});await popup.close();
  await page.getByRole('button',{name:/^Focus:/}).click();await page.locator('.focus-drawer').getByRole('button',{name:'Clear',exact:true}).click();await page.locator('.focus-drawer').getByRole('button',{name:'Bent arms',exact:true}).click();
  const memory=await newRun(summary.id,()=>page.getByRole('button',{name:'Save focus',exact:true}).click());await assistant().getByText('Focus updated',{exact:true}).waitFor();await findings().waitFor();assert.deepEqual((await get('/preferences'))[0].label_ids,[4]);
  await page.reload();await assistant().getByText('Focus updated',{exact:true}).waitFor();await page.getByRole('button',{name:/^Focus:/}).click();await page.getByRole('button',{name:'Use saved focus',exact:true}).click();assert.equal(await page.locator('.focus-drawer button[aria-pressed=true]').count(),1);
  const forgotten=await newRun(memory.id,()=>page.getByRole('button',{name:'Forget focus',exact:true}).click());assert.deepEqual(await get('/preferences'),[]);await assistant().getByText('Focus updated',{exact:true}).waitFor();
  await assistant().getByRole('button',{name:'Latest recap',exact:true}).click();await explanation().getByRole('heading',{name:'Your review recap',exact:true}).waitFor();await picture('06-saved-recap');
  await page.reload();await explanation().getByRole('heading',{name:'Your review recap',exact:true}).waitFor();assert.equal(await current(),summary.id);
  await page.setViewportSize({width:390,height:900});await picture('07-mobile-review');assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.getByRole('button',{name:'Open sessions',exact:true}).click();await page.keyboard.press('Escape');assert(await page.locator('.app-sidebar').evaluate(e=>e.inert));
  for(const id of state.runs.filter(id=>id!==analysed.id))assert.deepEqual(await get('/runs/'+id+'/tools'),[]);
  let calls=0;for(const id of state.runs){for(const e of (await get('/runs/'+id+'/events')).events)if(e.payload?.called===true)calls++;}
  assert(calls<=6,'Bounded provider use');state.provider_requests=calls;save();
  assert.deepEqual(state.browser_errors,[]);assert.deepEqual(state.external_requests,[]);
  fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify({passed:true,base_url:baseURL,workspace:state.workspace,analysis_run_id:analysed.id,summary_run_id:summary.id,summary_report_id:summary.report_id,runs:state.runs,prediction:state.prediction,real_vision_tools:vision.map(t=>({tool_id:t.request.tool_id,status:t.status,duration_ms:t.duration_ms})),vision_duration_ms:state.vision_duration_ms,real_provider_requests:calls,refreshed_during_real_analysis:state.running_refresh_verified,scope_follow_up_passed:true,explicit_preferences_passed:true,print_export_passed:true,refresh_restored_summary:true,mocked_ui_or_cv:false,browser_errors:state.browser_errors,external_browser_requests:state.external_requests.length},null,2));
  console.log(JSON.stringify({passed:true,vision_ms:state.vision_duration_ms,provider_requests:calls,summary_run_id:summary.id}));
 }catch(e){save();for(const c of browser.contexts())await c.pages()[0]?.screenshot({path:path.join(output,'failure-'+Date.now()+'.png'),fullPage:true}).catch(()=>{});throw e;}
 finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
