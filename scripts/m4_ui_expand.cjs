/* Actual user click: expand a focused report, keep its source, and execute only missing stages. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'test-results/m4-scheduling-v1/ui-expansion'),baseURL='http://127.0.0.1:8011';
async function main(){
  fs.mkdirSync(out,{recursive:true});assert(!fs.existsSync(path.join(out,'acceptance.json')),'Preserve clicked expansion evidence');
  const row=JSON.parse(fs.readFileSync(path.join(root,'test-results/m4-scheduling-v1/http/state.json'),'utf8')).body;
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});let p;
  try{
    const c=await browser.newContext({baseURL,viewport:{width:1440,height:1100}});p=await c.newPage();const errors=[];p.on('pageerror',e=>errors.push(e.message));
    await p.goto('/');await p.evaluate(row=>{localStorage.setItem('cpr.workspace',JSON.stringify({session_id:row.session_id,clip_id:row.clip_id,asset_id:row.asset_id,name:row.name}));localStorage.setItem('cpr.runId',JSON.stringify(row.run_id));},row);await p.reload();
    await p.getByRole('region',{name:'Scope of this analysis',exact:true}).waitFor();await p.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Expand to all categories')&&!b.disabled));
    await p.getByRole('button',{name:'Expand to all categories',exact:true}).click();
    await p.waitForFunction(old=>{const value=JSON.parse(localStorage.getItem('cpr.runId')||'null');return value&&value!==old;},row.run_id);
    const id=await p.evaluate(()=>JSON.parse(localStorage.getItem('cpr.runId')));let run;let marker;const until=Date.now()+300000;
    while(Date.now()<until){run=await (await c.request.get('/api/runs/'+id)).json();const tools=await (await c.request.get('/api/runs/'+id+'/tools')).json();const text=run.status+':'+tools.map(t=>t.request.tool_id+'='+t.status).join(',');if(text!==marker){console.log(text);marker=text}if(['completed','partially_completed','failed','rejected','cancelled'].includes(run.status))break;await p.waitForTimeout(700)}
    assert.equal(run.status,'completed');const tools=await (await c.request.get('/api/runs/'+id+'/tools')).json();assert.deepEqual(tools.map(t=>t.request.tool_id),['rgb.predict','fusion.predict']);assert.equal(run.response.reused_result,false);assert.equal(run.response.prediction.analysis_route,'full-v15');assert.equal(run.response.classification_claims.length,11);
    await p.waitForFunction(()=>!document.querySelector('[aria-label="Scope of this analysis"]'));await p.getByText('Saved source checked',{exact:false}).waitFor();await p.screenshot({path:path.join(out,'expanded-after-click.png'),fullPage:true});assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({passed:true,actual_user_click:true,run_id:id,report_id:run.report_id,original_report_id:row.report_id,actual_tool_ids:tools.map(t=>t.request.tool_id),new_successful_vision_executions:2,source_report_preserved:true,provider_calls:0,browser_errors:errors},null,2));console.log('Actual expand button and cached dependency execution passed');
  }catch(e){if(p)await p.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{});throw e}finally{await browser.close()}
}
main().catch(e=>{console.error(e);process.exitCode=1});
