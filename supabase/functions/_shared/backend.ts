export class Failure extends Error { constructor(public code:string,public status=400){super(code)} }
export const env=(key:string)=>Deno.env.get(key)||'';
function secret(){return env('SUPABASE_SERVICE_ROLE_KEY')||JSON.parse(env('SUPABASE_SECRET_KEYS')||'{}').default||env('SUPABASE_SECRET_KEY')}
export async function db(path:string,method='GET',body?:unknown,jwt?:string){
  const key=secret();if(!key||!env('SUPABASE_URL'))throw new Failure('server_configuration',503);
  const response=await fetch(env('SUPABASE_URL')+'/rest/v1/'+path,{method,headers:{apikey:key,Authorization:'Bearer '+(jwt||key),'Content-Type':'application/json',Prefer:'return=representation'},body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(10000)});
  if(!response.ok){const error=await response.json().catch(()=>({}));const message=String(error.message||'');
    const known=['event_conflict','sequence_conflict','identity_conflict','attempt_identity_conflict','epoch_conflict','job_identity_conflict','resume_conflict','parent_not_registered','branch_conflict','experiment_not_registered','credential_denied','scope_denied','rate_limited'];
    const code=known.find(k=>message.includes(k));throw new Failure(code||'database_request_failed',code==='rate_limited'?429:code?.endsWith('denied')?403:code?409:response.status===403?403:503)}
  return response.status===204?null:await response.json();
}
export async function user(req:Request){
  const authorization=req.headers.get('Authorization')||'';
  if(!authorization.startsWith('Bearer '))throw new Failure('login_required',401);
  const response=await fetch(env('SUPABASE_URL')+'/auth/v1/user',{headers:{apikey:secret(),Authorization:authorization},signal:AbortSignal.timeout(10000)});
  if(!response.ok)throw new Failure('login_expired',401);
  const identity=await response.json();if(!identity.id)throw new Failure('login_required',401);
  return {id:identity.id as string,jwt:authorization.slice(7)};
}
export async function member(req:Request,project:string,owner=false){
  const identity=await user(req);
  const rows=await db(`project_members?project_id=eq.${project}&user_id=eq.${identity.id}&select=role`,'GET',undefined,identity.jwt);
  if(!rows.length||(owner&&rows[0].role!=='owner'))throw new Failure('membership_required',403);
  return identity;
}
export async function body(req:Request){
  const reader=req.body?.getReader();if(!reader)throw new Failure('body_required');
  const chunks:Uint8Array[]=[];let size=0;
  try{while(true){const r=await reader.read();if(r.done)break;size+=r.value.length;if(size>262144){await reader.cancel();throw new Failure('body_too_large',413)}chunks.push(r.value)}}finally{reader.releaseLock()}
  const bytes=new Uint8Array(size);let offset=0;for(const c of chunks){bytes.set(c,offset);offset+=c.length}
  try{return JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes))}catch{throw new Failure('invalid_json')}
}
export async function digest(value:string){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),b=>b.toString(16).padStart(2,'0')).join('')}
export function route(req:Request,name:string){return new URL(req.url).pathname.split('/'+name).at(-1)||'/'}
export function reply(value:unknown,status=200){return Response.json(value,{status,headers:{'Cache-Control':'no-store',...(status===429?{'Retry-After':'60'}:{})}})}
export function serve(handler:(req:Request)=>Promise<Response>){
  Deno.serve(async req=>{
    const request_id=crypto.randomUUID(),origin=req.headers.get('Origin');
    const allowed=env('MONITOR_ALLOWED_ORIGINS').split(',').map(s=>s.trim()).filter(Boolean);
    let response:Response;
    try{
      if(origin&&!allowed.includes(origin))throw new Failure('origin_denied',403);
      response=req.method==='OPTIONS'?new Response(null,{status:204}):await handler(req);
    }catch(error){const e=error instanceof Failure?error:new Failure('request_failed',503);response=reply({code:e.code,message:e.code.replaceAll('_',' '),request_id},e.status)}
    response.headers.set('X-Request-Id',request_id);
    if(origin&&allowed.includes(origin)){response.headers.set('Access-Control-Allow-Origin',origin);response.headers.set('Vary','Origin');response.headers.set('Access-Control-Allow-Headers','authorization,apikey,content-type,x-client-info');response.headers.set('Access-Control-Allow-Methods','GET,POST,DELETE,OPTIONS')}
    return response;
  });
}
