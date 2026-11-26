import {body,db,digest,Failure,member,reply,route,serve} from '../_shared/backend.ts';
import {uuid} from '../_shared/contracts.ts';
serve(async req=>{
  const url=new URL(req.url),project=url.searchParams.get('project_id');if(!uuid(project))throw new Failure('project_required');
  await member(req,project,true);
  const path=route(req,'monitor-admin');
  if(path==='/v1/profiles'&&req.method==='GET')return reply({items:await db(`source_profiles?project_id=eq.${project}&select=id,label,cluster`)});
  if(path==='/v1/upload-keys'&&req.method==='GET')return reply({items:await db(`upload_credentials?project_id=eq.${project}&select=id,label,scopes,created_at,expires_at,revoked_at,last_used_at`)});
  if(path==='/v1/upload-keys'&&req.method==='POST'){
    const p=await body(req);if(!uuid(p.profile_id)||typeof p.label!=='string'||!p.label.trim()||p.label.length>100||!Array.isArray(p.scopes)||!p.scopes.length||p.scopes.some((s:unknown)=>!['training:write','scheduler:write'].includes(String(s))))throw new Failure('invalid_key_request');
    const profiles=await db(`source_profiles?project_id=eq.${project}&id=eq.${p.profile_id}&select=id`);if(!profiles.length)throw new Failure('profile_denied',403);
    const bytes=crypto.getRandomValues(new Uint8Array(32)),token='rnd_'+Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');
    const rows=await db('upload_credentials','POST',{project_id:project,profile_id:p.profile_id,label:p.label,scopes:p.scopes,token_hash:await digest(token),expires_at:new Date(Date.now()+180*86400000).toISOString()});
    return reply({id:rows[0].id,token,expires_at:rows[0].expires_at},201);
  }
  const id=path.split('/').at(-1);
  if(path.startsWith('/v1/upload-keys/')&&req.method==='DELETE'&&uuid(id)){
    await db(`upload_credentials?project_id=eq.${project}&id=eq.${id}`,'PATCH',{revoked_at:new Date().toISOString()});return reply({revoked:true});
  }
  if(path==='/v1/members'&&req.method==='POST'){
    const p=await body(req);if(!uuid(p.user_id)||!['owner','viewer'].includes(p.role))throw new Failure('invalid_membership');
    return reply(await db('project_members','POST',{project_id:project,user_id:p.user_id,role:p.role}),201);
  }
  throw new Failure('not_found',404);
});
