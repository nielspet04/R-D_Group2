/* Offline Reveal.js with local-server autosave for editable slide text. */
const printing = new URLSearchParams(location.search).has('print-pdf');
Reveal.initialize({width:1280,height:720,margin:0.04,center:false,hash:true,controls:!printing,progress:!printing,slideNumber:'c/t',transition:'fade',transitionSpeed:'fast',backgroundTransition:'none',showNotes:false,pdfSeparateFragments:false,pdfMaxPagesPerSlide:1,autoPlayMedia:false,plugins:[RevealNotes]});
let editing=false, revision=document.body.dataset.saveRevision, token=null, saveTimer=null, pending=null, blocked=false;
const changes=new Map();
const status=document.querySelector('#save-status');
function showStatus(message, failed=false){status.textContent=message;status.classList.toggle('save-error',failed);}
async function saveChanges(){
  clearTimeout(saveTimer);
  if(pending){await pending;if(changes.size&&!blocked)return saveChanges();return !changes.size;}
  if(!changes.size)return true;
  if(!token||blocked){showStatus(blocked?'Not saved — download backup before reloading':'Not saved — server unavailable',true);return false;}
  const batch=Object.fromEntries(changes);
  showStatus('Saving…');
  pending=(async()=>{
    try{
      const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({revision,changes:batch}),keepalive:true});
      const result=await response.json();
      if(!response.ok){if(response.status===409)blocked=true;throw new Error(result.error||'Save failed');}
      revision=result.revision;
      for(const [id,value] of Object.entries(batch))if(changes.get(id)===value)changes.delete(id);
      showStatus(changes.size?'Unsaved changes':'Saved');
      return true;
    }catch(error){showStatus(error.message||'Not saved — retry',true);return false;}
  })();
  const saved=await pending;pending=null;
  if(saved&&changes.size)return saveChanges();
  return saved;
}
function toggleEditing(){
  editing=!editing;document.body.classList.toggle('editing',editing);
  document.querySelectorAll('[data-edit]').forEach(el=>{el.contentEditable=String(editing);el.spellcheck=editing;});
  Reveal.configure({keyboard:!editing});
  document.querySelector('#edit').textContent=editing?'Finish editing · Esc':'Edit text · E';
  if(!editing)void saveChanges();
}
document.querySelector('#edit').onclick=toggleEditing;
document.addEventListener('input',event=>{
  const field=event.target.closest('[data-edit-id]');if(!field)return;
  changes.set(field.dataset.editId,field.innerHTML);showStatus('Unsaved changes');
  clearTimeout(saveTimer);saveTimer=setTimeout(()=>void saveChanges(),700);
});
document.addEventListener('keydown',event=>{
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();void saveChanges();}
  else if(event.key==='Escape'&&editing){event.preventDefault();toggleEditing();}
  else if(event.key.toLowerCase()==='e'&&!editing&&!event.ctrlKey&&!event.metaKey){event.preventDefault();toggleEditing();}
});
window.addEventListener('beforeunload',event=>{if(changes.size||pending){event.preventDefault();event.returnValue='';}});
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden'&&changes.size)void saveChanges();});
document.querySelector('#save').onclick=()=>{
  const clone=document.documentElement.cloneNode(true);
  clone.querySelectorAll('[data-edit]').forEach(el=>{el.removeAttribute('contenteditable');el.removeAttribute('spellcheck');});
  clone.querySelectorAll('.slides section').forEach(el=>{el.classList.remove('present','past','future');el.removeAttribute('hidden');el.removeAttribute('aria-hidden');el.removeAttribute('style');});
  clone.querySelectorAll('.fragment').forEach(el=>el.classList.remove('visible','current-fragment'));
  clone.querySelector('.slides').removeAttribute('style');
  const deck=clone.querySelector('.reveal');deck.removeAttribute('style');deck.innerHTML=clone.querySelector('.slides').outerHTML;
  clone.querySelector('body').removeAttribute('data-save-revision');clone.querySelector('body').className='';clone.className='';
  clone.querySelector('#edit').textContent='Edit text · E';clone.querySelector('#save-status').textContent='Connecting autosave…';
  const blob=new Blob(['<!doctype html>\n'+clone.outerHTML],{type:'text/html'});
  const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='index.html';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
document.querySelector('#pdf').onclick=async()=>{
  const target=window.open('about:blank','_blank');
  if(await saveChanges()){if(target)target.location=location.pathname+'?print-pdf';}
  else if(target)target.close();
};
Reveal.on('slidechanged',()=>{document.querySelectorAll('video').forEach(video=>video.pause());});
if(!printing){
  fetch('/api/editor-state',{cache:'no-store'}).then(async response=>{
    if(!response.ok)throw new Error('Server unavailable');
    const state=await response.json();
    if(revision!==state.revision){blocked=true;throw new Error('Reload to enable autosave');}
    token=state.token;showStatus(changes.size?'Unsaved changes':'Autosave on');if(changes.size)void saveChanges();
  }).catch(()=>showStatus('Autosave unavailable — use backup',true));
}
