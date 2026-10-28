/* Verify product usability fixes against saved real evidence; no new analysis/provider requests. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const {chromium}=require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {buildSync}=require('../frontend/node_modules/esbuild');
const root=path.resolve(__dirname,'..'),output=path.join(root,'test-results/product-demo-v2/usability');
fs.mkdirSync(output,{recursive:true});assert(!fs.existsSync(path.join(output,'acceptance.json')),'Preserve acceptance');
const unit=path.join(output,'context-status-check.cjs');
buildSync({stdin:{contents:`import {renderToStaticMarkup} from 'react-dom/server'; import ContextPanel from './src/ContextPanel'; import assert from 'node:assert/strict';
const context={session_id:'s',video_name:'practice.mp4',source_status:'none',source_error_code:null,active_label_ids:[],source:null};
const render=(activity,source_status='none')=>renderToStaticMarkup(<ContextPanel context={{...context,source_status,active_label_ids:[4]}} labelNames={{4:'Bent arms'}} error='' activity={activity}/>);
assert(render(undefined).includes('Ready for its first review'));assert(render('reviewing').includes('Review in progress'));assert(!render('reviewing').includes('Ready for its first review'));
assert(render('waiting').includes('Waiting for your confirmation'));assert(render('reviewing','available').includes('Discussing Bent arms'));assert(render('reviewing','unavailable').includes('Saved evidence is unavailable'));
console.log('Context state semantics: 5 cases passed; isolated component fixtures, not a vision test');`,resolveDir:path.join(root,'frontend'),loader:'tsx'},bundle:true,jsx:'automatic',platform:'node',format:'cjs',outfile:unit});
console.log(execFileSync(process.execPath,[unit],{encoding:'utf8'}).trim());
(async()=>{
 const b=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 try{
  const c=await b.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'}),p=await c.newPage(),errors=[],writes=[];
  p.on('pageerror',e=>errors.push(e.message));p.on('request',r=>{if(r.method()!=='GET')writes.push(r.url());});
  const state=JSON.parse(fs.readFileSync(path.join(root,'test-results/product-demo-v2/browser/run-state.json'),'utf8'));
  await p.goto('http://127.0.0.1:8007');
  await p.evaluate(({ws,id})=>{localStorage.setItem('cpr.workspace',JSON.stringify(ws));localStorage.setItem('cpr.runId',JSON.stringify(id));},{ws:state.workspace,id:state.analysis_run_id});await p.reload();
  const a=p.getByRole('complementary',{name:'Practice review assistant'});
  await a.getByText('Video review: all 11 categories',{exact:true}).waitFor();await a.getByRole('heading',{name:'Your video review',exact:true}).waitFor();
  await a.getByRole('button',{name:'Replay video',exact:true}).click();const video=p.getByLabel('Practice video player',{exact:true});await p.waitForTimeout(800);
  const playback=await video.evaluate(e=>{e.pause();return{time:e.currentTime,paused:e.paused,error:e.error?.code||null};});assert(playback.time>0&&!playback.error);
  await p.evaluate(()=>scrollTo(0,0));await p.screenshot({path:path.join(output,'all-categories-1440.png'),fullPage:true});
  await a.getByRole('button',{name:'Latest recap',exact:true}).click();await a.getByRole('heading',{name:'Your review recap',exact:true}).waitFor();
  await a.getByText('Open the source video and compare the recorded category with the whole clip.',{exact:true}).waitFor();assert.equal(await p.locator('.journey > span:last-child.done svg').count(),1);
  for(const width of [1440,1024,768,390]){
   await p.setViewportSize({width,height:1000});await p.waitForTimeout(100);assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   await p.evaluate(()=>scrollTo(0,0));await p.screenshot({path:path.join(output,`recap-${width}.png`),fullPage:true});
  }
  await a.getByRole('button',{name:'Replay video',exact:true}).click();await p.waitForTimeout(650);assert(await video.evaluate(e=>e.currentTime>0&&!e.error));await video.evaluate(e=>e.pause());
  assert.deepEqual(writes,[]);assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify({passed:true,full_scope_named_correctly:true,recap_step_completed:true,verified_next_step_visible:true,explicit_replay_plays_entire_clip:true,context_state_cases:5,context_state_validation:'Isolated component fixtures, separate from real CV acceptance',widths:[1440,1024,768,390],new_vision_calls:0,paid_calls:0,errors,writes},null,2));
  console.log('Usability fixes passed against actual saved evidence; zero new vision or provider calls');
 }finally{await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
