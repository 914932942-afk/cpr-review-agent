/* Real user options, partial start and expansion. No intercepted/mocked API. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'test-results/m4-exploration-v2/ui'),baseURL='http://127.0.0.1:8012';
const terminal=new Set(['completed','partially_completed','failed','rejected','cancelled']);
async function main(){
 fs.mkdirSync(out,{recursive:true});assert(!fs.existsSync(path.join(out,'acceptance.json')));
 const state=JSON.parse(fs.readFileSync(path.join(root,'test-results/m4-exploration-v2/http/state.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 let p;const errors=[],external=[];
 try {
  const c=await browser.newContext({baseURL,viewport:{width:1440,height:1050},reducedMotion:'reduce'});p=await c.newPage();
  p.on('pageerror',e=>errors.push(e.message));p.on('request',r=>{if(/^https?:/.test(r.url())&&new URL(r.url()).origin!==baseURL)external.push(r.url());});
  async function get(q){const r=await c.request.get('/api'+q);assert(r.ok(),await r.text());return r.json()}
  async function post(q,body){const r=await c.request.post('/api'+q,{data:body,headers:{'Idempotency-Key':crypto.randomUUID()}});assert(r.ok(),await r.text());return r.json()}
  async function restore(row,rid){await p.goto('/');await p.evaluate(({row,rid})=>{localStorage.setItem('cpr.workspace',JSON.stringify(row));if(rid)localStorage.setItem('cpr.runId',JSON.stringify(rid));else localStorage.removeItem('cpr.runId')},{row,rid});await p.reload();}
  async function wait(id){let r,mark;const until=Date.now()+930000;while(Date.now()<until){r=await get('/runs/'+id);const ts=await get('/runs/'+id+'/tools');const msg=r.status+':'+ts.map(t=>t.request.tool_id+'='+t.status).join(',');if(mark!==msg){console.log(msg);mark=msg}if(terminal.has(r.status))return {run:r,tools:ts};await p.waitForTimeout(600)}throw new Error('Run timeout')}
  let row;const saved=path.join(out,'state.json');
  if(fs.existsSync(saved))row=JSON.parse(fs.readFileSync(saved,'utf8'));else{
   const clip=await post('/clips',{asset_id:state.partial.asset_id});const session=await post('/sessions',{clip_id:clip.id});
   row={session_id:session.id,clip_id:clip.id,asset_id:state.partial.asset_id,name:'M4 bounded review demo'};fs.writeFileSync(saved,JSON.stringify(row,null,2));
  }
  await restore(row,row.run_id||null);
  if(!row.run_id){
   const start=p.getByRole('button',{name:'Start review',exact:true});await start.waitFor();await p.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Start review'&&!b.disabled));
   await p.locator('.review-options summary').click();assert.equal(await p.getByRole('checkbox',{name:'Allow a partial review if the complete check cannot fit.',exact:true}).isChecked(),false);
   await p.getByLabel('Time limit',{exact:true}).selectOption('180');await p.getByRole('checkbox',{name:'Allow a partial review if the complete check cannot fit.',exact:true}).check();
   await p.getByRole('button',{name:/^Focus:/}).click();await p.getByRole('button',{name:'Select all',exact:true}).click();await p.getByRole('button',{name:/^Focus:/}).click();
   const request=p.waitForResponse(r=>r.url().includes('/turns')&&r.request().method()==='POST');await start.click();const resp=await request;assert.equal(resp.status(),202);const payload=resp.request().postDataJSON();assert.equal(payload.goal.allow_partial,true);assert.equal(payload.goal.analysis_budget_seconds,180);assert.equal(payload.goal.target_label_ids.length,11);
   row.run_id=(await resp.json()).run_id;fs.writeFileSync(saved,JSON.stringify(row,null,2));
  }
  const partial=await wait(row.run_id);assert.equal(partial.run.status,'partially_completed');assert.deepEqual(partial.tools.map(t=>t.request.tool_id),['pose.extract','skeleton.predict','fusion.predict']);
  await p.getByText('Partial review · 4 of 11 requested categories checked',{exact:true}).waitFor();await p.getByRole('region',{name:'Scope of this analysis',exact:true}).waitFor();
  const exportRes=await c.request.get('/api/reports/'+partial.run.report_id+'/export');assert(exportRes.ok());assert((await exportRes.text()).includes('Not checked: categories'));
  for(const width of [1440,390]){await p.setViewportSize({width,height:1000});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await p.screenshot({path:path.join(out,'partial-'+width+'.png'),fullPage:true})}
  await p.setViewportSize({width:1440,height:1050});
  if(!row.expand_run_id){await p.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Expand to all categories')&&!b.disabled));const req=p.waitForResponse(r=>r.url().includes('/turns')&&r.request().method()==='POST');await p.getByRole('button',{name:'Expand to all categories',exact:true}).click();const response=await req;assert.equal(response.status(),202);row.expand_run_id=(await response.json()).run_id;fs.writeFileSync(saved,JSON.stringify(row,null,2));}
  const expanded=await wait(row.expand_run_id);assert.equal(expanded.run.status,'completed');assert.deepEqual(expanded.tools.map(t=>t.request.tool_id),['rgb.predict','fusion.predict']);assert.equal(expanded.run.response.classification_claims.length,11);
  await p.waitForFunction(()=>!document.querySelector('.partial-result-note'));await p.screenshot({path:path.join(out,'expanded.png'),fullPage:true});
  assert.equal((await get('/reports/'+partial.run.report_id+'/evidence')).status,'available');
  // Reopen the explicitly selected partial report: no silent substitution by its newer full report.
  await restore(row,row.run_id);await p.getByText('Partial review · 4 of 11 requested categories checked',{exact:true}).waitFor();
  await restore(state.full,state.full.run_id);await p.getByRole('region',{name:'Categories needing another look',exact:true}).waitFor();
  await restore(state['complete-limited'],state['complete-limited'].run_id);await p.getByRole('button',{name:'Retry with more time',exact:true}).waitFor();await p.getByText('The selected checks could not fit the time limit.',{exact:false}).waitFor();
  await p.screenshot({path:path.join(out,'budget-stop.png'),fullPage:true});
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({passed:true,real_options_and_start_click:true,partial_run_id:row.run_id,expand_run_id:row.expand_run_id,actual_successful_tool_calls:5,explicit_consent_default_off:true,actual_requested_budget_seconds:180,partial_categories_checked:4,requested_categories:11,scope_expansion_only_missing:true,old_partial_source_reopens_without_replacement:true,export_marks_unanalysed:true,budget_stop_actionable:true,full_uncertainty_preserved:true,widths:[1440,390],browser_errors:errors,provider_calls:0},null,2));console.log('Real options, partial review, expansion and source restoration passed');
 }catch(e){if(p)await p.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{});throw e}finally{await browser.close()}
}
main().catch(e=>{console.error(e);process.exitCode=1});
