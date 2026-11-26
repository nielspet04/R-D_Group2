import {db,Failure,member,reply,route,serve} from '../_shared/backend.ts';
import {uuid} from '../_shared/contracts.ts';
serve(async req=>{
  if(req.method!=='GET')throw new Failure('method_not_allowed',405);
  const url=new URL(req.url),project=url.searchParams.get('project_id');if(!uuid(project))throw new Failure('project_required');
  const identity=await member(req,project),path=route(req,'monitor-read');
  const rpc=(name:string,body:unknown,query='')=>db('rpc/'+name+query,'POST',body,identity.jwt);
  if(path==='/v1/overview'){
    const limit=Math.min(200,Math.max(1,Number(url.searchParams.get('limit')||50))),offset=Number(url.searchParams.get('cursor')||0);
    if(!Number.isInteger(limit)||!Number.isSafeInteger(offset)||offset<0)throw new Failure('invalid_pagination');
    const items=await rpc('monitor_overview',{p_project:project,p_limit:limit,p_offset:offset});
    return reply({items,next_cursor:items.length===limit?offset+limit:null});
  }
  if(path==='/v1/compare'){
    const ids=(url.searchParams.get('ids')||'').split(',');if(!ids.length||ids.length>6||!ids.every(uuid))throw new Failure('invalid_ids');
    const runs=await Promise.all(ids.map(id=>rpc('monitor_run',{p_project:project,p_experiment:id})));
    return reply({runs:runs.filter(Boolean)});
  }
  const match=path.match(/^\/v1\/experiments\/([^/]+)(\/epochs)?$/);
  if(match){
    const id=match[1];if(!uuid(id))throw new Failure('invalid_id');
    if(!match[2]){const run=await rpc('monitor_run',{p_project:project,p_experiment:id});if(!run)throw new Failure('not_found',404);return reply(run)}
    const after=Number(url.searchParams.get('cursor')||0),revision=url.searchParams.get('revision');
    if(!Number.isSafeInteger(after)||after<0||(revision&&!uuid(revision)))throw new Failure('invalid_cursor');
    const rows=await rpc('monitor_history',{p_project:project,p_experiment:id,p_attempt:revision||null},`?epoch=gt.${after}&limit=200&order=epoch.asc`);
    return reply({items:rows.map((r:{metrics:unknown})=>r.metrics),next_cursor:rows.length===200?rows.at(-1).epoch:null});
  }
  const telemetry=path.match(/^\/v1\/attempts\/([^/]+)\/(telemetry|log)$/);
  if(telemetry){
    const id=telemetry[1];if(!uuid(id))throw new Failure('invalid_id');
    if(telemetry[2]==='log')return reply({available:false,reason:'Cloud log upload is disabled; inspect original VSC logs.'});
    const resolution=url.searchParams.get('resolution')||'raw';
    if(!['raw','hourly'].includes(resolution))throw new Failure('invalid_resolution');
    const rows=await db(`${resolution==='raw'?'telemetry_samples':'telemetry_hourly'}?project_id=eq.${project}&attempt_id=eq.${id}&order=${resolution==='raw'?'observed_at':'hour'}.desc&limit=300`,'GET',undefined,identity.jwt);
    return reply({items:rows,resolution});
  }
  if(path==='/v1/export'){
    const id=url.searchParams.get('experiment_id');if(!uuid(id))throw new Failure('invalid_id');
    const experiment=await rpc('monitor_run',{p_project:project,p_experiment:id});if(!experiment)throw new Failure('not_found',404);
    const rows=await rpc('monitor_history',{p_project:project,p_experiment:id},'?limit=10000&order=epoch.asc');
    if(rows.length===10000)throw new Failure('use_paginated_export',413);
    return reply({...experiment,history:rows.map((r:{metrics:unknown})=>r.metrics)});
  }
  throw new Failure('not_found',404);
});
