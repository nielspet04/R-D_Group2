import {body,db,digest,Failure,reply,route,serve} from '../_shared/backend.ts';
import {validateBatch} from '../_shared/contracts.ts';
serve(async req=>{
  if(req.method!=='POST')throw new Failure('method_not_allowed',405);
  const token=req.headers.get('X-RND-Upload-Key')||'';
  if(token.length<40||token.length>200)throw new Failure('credential_required',401);
  const rows=await db('upload_credentials?token_hash=eq.'+await digest(token)+'&select=id,revoked_at,expires_at');
  const credential=rows[0];if(!credential||credential.revoked_at||Date.parse(credential.expires_at)<=Date.now())throw new Failure('credential_denied',401);
  const path=route(req,'monitor-ingest');
  if(path==='/v1/probe')return reply({ok:true});
  if(path!=='/v1/batches')throw new Failure('not_found',404);
  const payload=await body(req);let events;
  try{events=validateBatch(payload)}catch{throw new Failure('invalid_batch',422)}
  return reply(await db('rpc/monitor_ingest','POST',{p_credential:credential.id,p_events:events}));
});
