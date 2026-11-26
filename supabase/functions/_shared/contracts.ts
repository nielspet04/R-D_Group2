import schema from './schema.json' with {type:'json'};
export type Event = {event_id:string;experiment_id:string;attempt_id:string;sequence:number;kind:string;observed_at:string;payload:Record<string,unknown>};
export const uuid = (v:unknown):v is string => typeof v==='string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(v);
const object = (v:unknown):v is Record<string,unknown> => !!v && typeof v==='object' && !Array.isArray(v);
const integer = (v:unknown) => Number.isSafeInteger(v) && typeof v==='number';
export function validateEvent(input:unknown):Event {
  if(!object(input)||Object.keys(input).sort().join(',')!=='attempt_id,event_id,experiment_id,kind,observed_at,payload,sequence')throw Error('invalid_envelope');
  const e=input as Event;
  if(!uuid(e.event_id)||!uuid(e.experiment_id)||!uuid(e.attempt_id)||!integer(e.sequence)||e.sequence<=0)throw Error('invalid_identity');
  if(typeof e.observed_at!=='string'||!/(Z|\+00:00)$/.test(e.observed_at)||!Number.isFinite(Date.parse(e.observed_at))||Date.parse(e.observed_at)>Date.now()+600000)throw Error('invalid_timestamp');
  const fields=(schema.fields as Record<string,string[]>)[e.kind];
  if(!fields||!object(e.payload)||Object.keys(e.payload).some(k=>!fields.includes(k)))throw Error('invalid_fields');
  const p=e.payload;
  if(e.kind==='experiment_registered'&&(typeof p.model!=='string'||!p.model||p.model.length>100))throw Error('model_required');
  if(e.kind==='epoch_committed'&&(!integer(p.epoch)||(p.epoch as number)<1||p.history_revision!==e.attempt_id))throw Error('invalid_epoch');
  if(e.kind==='resume_confirmed'&&(!uuid(p.parent_attempt_id)||!integer(p.cutoff)||(p.cutoff as number)<0))throw Error('invalid_resume');
  if(e.kind==='batch_exit_reported'&&(!integer(p.exit_code)||(p.exit_code as number)<0||(p.exit_code as number)>255))throw Error('invalid_exit');
  for(const key of schema.metrics){const n=p[key];if(n!==undefined&&n!==null){if(typeof n!=='number'||!Number.isFinite(n))throw Error('invalid_metric');if(key.endsWith('mt5r')&&(n<0||n>1))throw Error('invalid_recall')}}
  if(p.job_id!==undefined&&p.job_id!==null&&(typeof p.job_id!=='string'||!/^[0-9_+.[\]-]{1,100}$/.test(p.job_id)))throw Error('invalid_job');
  for(const key of ['cluster','phase','reason','error_code','completion_reason'])if(p[key]!==undefined&&(typeof p[key]!=='string'||(p[key] as string).length>100))throw Error('invalid_text');
  if(p.configuration!==undefined&&(!object(p.configuration)||JSON.stringify(p.configuration).length>32768))throw Error('invalid_configuration');
  if(p.gpus!==undefined){if(!Array.isArray(p.gpus)||p.gpus.length>16)throw Error('invalid_gpus');for(const g of p.gpus){if(!object(g)||Object.keys(g).some(k=>!['uuid','index','utilization','memory_used','memory_total'].includes(k))||typeof g.uuid!=='string'||g.uuid.length>100)throw Error('invalid_gpu');for(const k of ['utilization','memory_used','memory_total'])if(typeof g[k]!=='number'||!Number.isFinite(g[k])||(g[k] as number)<0)throw Error('invalid_gpu');if((g.utilization as number)>100)throw Error('invalid_gpu')}}
  if(e.kind==='scheduler_snapshot'){if(typeof p.cluster!=='string'||!Array.isArray(p.jobs)||p.jobs.length>200)throw Error('invalid_scheduler');for(const j of p.jobs){if(!object(j)||typeof j.job_id!=='string'||typeof j.state!=='string'||Object.keys(j).some(k=>!['job_id','state','name','partition','elapsed','node','exit_code','reason','submitted_at','started_at','ended_at','time_limit'].includes(k)))throw Error('invalid_scheduler')}}
  return e;
}
export function validateBatch(input:unknown){
  if(!object(input)||input.schema_version!==1||!uuid(input.batch_id)||!Array.isArray(input.events)||!input.events.length||input.events.length>50||Object.keys(input).some(k=>!['schema_version','batch_id','events'].includes(k)))throw Error('invalid_batch');
  return input.events.map(validateEvent);
}
