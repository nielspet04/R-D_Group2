import {createClient} from '@supabase/supabase-js';
const config=window.MONITOR_CONFIG||{};
let session=null, client=null, latestOverview=null;
const $=id=>document.getElementById(id);
function clearPrivate(){for(const id of ['jobs','detail','compare','sources','cards']){$(id).replaceChildren();delete $(id).dataset.signature}if($('new-key'))$('new-key').value='';if($('key-list'))$('key-list').replaceChildren();$('key-result').hidden=true;$('key-controls').hidden=true;latestOverview=null;window.dispatchEvent(new Event('monitor:signout'))}
async function call(functionName,path,method='GET',body){
  if(!session)throw Error('Sign in to view private results.');
  const requestUser=session.user.id;
  const url=new URL(`${config.supabaseUrl}/functions/v1/${functionName}/v1/${path}`);
  url.searchParams.set('project_id',config.projectId);
  const response=await fetch(url,{method,headers:{apikey:config.publishableKey,Authorization:'Bearer '+session.access_token,'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),cache:'no-store',signal:AbortSignal.timeout(15000)});
  const value=await response.json().catch(()=>({}));
  if(session?.user.id!==requestUser)throw Error('Signed out; private response discarded.');
  if(!response.ok){if(response.status===401)throw Error('Login expired. Please sign out and sign in again.');if(response.status===403)throw Error('Your account has no project access yet. Follow the owner bootstrap step in the setup guide.');throw Error(value.message||'Backend unavailable. The free project may be paused; resume it in Supabase and try again.')}
  return value;
}
async function fullRun(id){
  const run=await call('monitor-read','experiments/'+encodeURIComponent(id));
  const history=[];let cursor=0;
  do{const page=await call('monitor-read',`experiments/${id}/epochs?cursor=${cursor}&revision=${run.history_revision||''}`);history.push(...page.items);cursor=page.next_cursor;if(history.length>10000)throw Error('History is too large for this view; use an export.')}while(cursor!==null);
  run.history=history;return run;
}
window.monitorSource={mode:'hosted',ready:null,async request(path){
  await this.ready;
  const url=new URL(path,location.origin);
  if(url.pathname==='/api/jobs')return {items:[],next_cursor:null};
  if(url.pathname==='/api/status'){
    const items=latestOverview?.items||[],times=items.map(r=>Date.parse(r.last_measurement)).filter(Number.isFinite);
    return {sources:{api:{status:'ok',last_success:times.length?Math.max(...times)/1000:null,cluster:'last uploaded training observation'}}};
  }
  if(url.pathname==='/api/experiments'){
    const items=[];let cursor=0;
    do{const page=await call('monitor-read','overview?limit=50&cursor='+cursor);items.push(...page.items);cursor=page.next_cursor;if(items.length>=1000)break;}while(cursor!==null);
    latestOverview={items,next_cursor:cursor};return latestOverview;
  }
  if(url.pathname.startsWith('/api/experiments/'))return fullRun(decodeURIComponent(url.pathname.split('/').at(-1)));
  if(url.pathname==='/api/compare'){
    const ids=(url.searchParams.get('ids')||'').split(',');if(ids.length>6)throw Error('Select at most six experiments.');
    return {runs:await Promise.all(ids.map(fullRun))};
  }
  throw Error('Unsupported hosted request');
}};

async function initialize(){
  $('auth-panel').hidden=false;$('refresh').textContent='Refresh view';
  if(!config.supabaseUrl||!config.publishableKey||!config.projectId){$('auth-message').textContent='Hosting files are ready. Configure the public Supabase settings using the setup guide before deploying.';$('signin').disabled=true;return}
  client=createClient(config.supabaseUrl,config.publishableKey,{auth:{flowType:'pkce',detectSessionInUrl:true,persistSession:true,autoRefreshToken:true}});
  const renderSession=()=>{
    $('signin').hidden=!!session;$('signout').hidden=!session;$('owner-panel').hidden=!session;
    $('auth-message').textContent=session?'Signed in. User ID for owner setup: '+session.user.id:'Sign in to view your private training results.';
  };
  session=(await client.auth.getSession()).data.session;renderSession();
  client.auth.onAuthStateChange((_event,next)=>{const signedOut=session&&!next;session=next;renderSession();if(signedOut)clearPrivate();window.dispatchEvent(new Event('monitor:auth'))});
  $('signin').onclick=async()=>{const {error}=await client.auth.signInWithOAuth({provider:'github',options:{redirectTo:location.origin+'/'}});if(error)$('auth-message').textContent=error.message};
  $('signout').onclick=async()=>{await client.auth.signOut();clearPrivate()};
  $('open-owner').onclick=async()=>{
    try{
      const profiles=await call('monitor-admin','profiles');
      const select=$('profile');select.replaceChildren();for(const p of profiles.items){const option=document.createElement('option');option.value=p.id;option.textContent=p.label;select.append(option)}
      $('key-controls').hidden=false;await listKeys();
    }catch(error){$('owner-message').textContent=error.message}
  };
  $('create-key').onclick=async()=>{
    const button=$('create-key');button.disabled=true;
    try{
      const value=await call('monitor-admin','upload-keys','POST',{profile_id:$('profile').value,label:$('key-label').value,scopes:[$('key-scope').value]});
      $('new-key').value=value.token;$('key-result').hidden=false;await listKeys();
    }catch(error){$('owner-message').textContent=error.message}finally{button.disabled=false}
  };
  $('download-key').onclick=()=>{const token=$('new-key').value;if(!token)return;const url=URL.createObjectURL(new Blob([token+'\n'],{type:'text/plain'})),a=document.createElement('a');a.href=url;a.download='upload-key';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('new-key').value='';$('key-result').hidden=true};
}
async function listKeys(){
  const {items}=await call('monitor-admin','upload-keys');$('key-list').replaceChildren();
  for(const key of items){const row=document.createElement('p');row.textContent=key.label+' · '+key.scopes.join(', ')+' · '+(key.revoked_at?'revoked':'expires '+new Date(key.expires_at).toLocaleDateString());
    if(!key.revoked_at){const button=document.createElement('button');button.textContent='Revoke';button.onclick=async()=>{if(!confirm('Revoke this uploader key? Existing results will remain.'))return;await call('monitor-admin','upload-keys/'+key.id,'DELETE');await listKeys()};row.append(' ',button)}$('key-list').append(row)}
}
window.monitorSource.ready=initialize().catch(error=>{$('auth-message').textContent='Login setup unavailable: '+error.message});
