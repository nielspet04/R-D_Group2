import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {build} from 'esbuild';
import vm from 'node:vm';

class Element {
  children=[];dataset={};value='';textContent='';hidden=false;
  append(child){this.children.push(child)}
  replaceChildren(){this.children=[];this.textContent=''}
}
const elements=new Map();
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement:()=>new Element()};
const window=new EventTarget();window.MONITOR_CONFIG={supabaseUrl:'https://example.supabase.co',publishableKey:'sb_publishable_test',projectId:'emiel-project'};
let account='niels',authChange,delayResponse;
const user=()=>({user:{id:account},access_token:'test-token'});
const client={auth:{getSession:async()=>({data:{session:user()}}),onAuthStateChange:callback=>authChange=callback},from:()=>({select:()=>({order:async()=>({data:account==='uninvited'?[]:[{id:account+'-project',name:account}]})})})};
const urls=[];
const fetch=async url=>{
  urls.push(new URL(url));
  return {ok:true,json:async()=>{if(delayResponse)await delayResponse;return {items:[],next_cursor:null}}};
};
const source=(await readFile('monitoring/hosted-entry.js','utf8')).replace("import {createClient} from '@supabase/supabase-js';",'const createClient=globalThis.__createClient;');
const bundled=await build({stdin:{contents:source,resolveDir:process.cwd()+'/monitoring'},bundle:true,write:false,format:'iife',platform:'browser'});
vm.runInNewContext(bundled.outputFiles[0].text,{window,document,URL,Event,AbortSignal,setTimeout,fetch,location:{origin:'https://dashboard.example'},__createClient:()=>client});
await window.monitorSource.ready;
await window.monitorSource.request('/api/experiments');
assert.equal(urls.at(-1).searchParams.get('project_id'),'niels-project');
assert.equal(elements.get('project-choice').children.length,1);
assert.equal(elements.get('project-choice').children[0].textContent,'niels');

// Token refresh keeps the current account's project, while identity changes clear private UI.
authChange('TOKEN_REFRESHED',user());
await window.monitorSource.request('/api/experiments');
assert.equal(urls.at(-1).searchParams.get('project_id'),'niels-project');
elements.get('new-key').value='private-key';
elements.get('detail').textContent='private-run';
let release;
delayResponse=new Promise(resolve=>release=resolve);
const oldRequest=window.monitorSource.request('/api/experiments');
await new Promise(resolve=>setTimeout(resolve,0));
account='emiel';authChange('SIGNED_IN',user());
assert.equal(elements.get('new-key').value,'');
assert.equal(elements.get('detail').textContent,'');
release();delayResponse=null;
await assert.rejects(oldRequest,/response discarded/);
await window.monitorSource.request('/api/experiments');
assert.equal(urls.at(-1).searchParams.get('project_id'),'emiel-project');
assert.equal(elements.get('project-choice').children.length,1);

account='uninvited';authChange('SIGNED_IN',user());
const before=urls.length;
await assert.rejects(window.monitorSource.request('/api/experiments'),/no project access/);
assert.equal(urls.length,before,'uninvited users must not request the default project');
authChange('SIGNED_OUT',null);
await assert.rejects(window.monitorSource.request('/api/experiments'),/Sign in/);
assert.equal(elements.get('project-picker').hidden,true);
console.log('Hosted sign-in, private project routing, token refresh, account switching and sign-out checks passed.');
