create or replace function public.monitor_run(p_project uuid,p_experiment uuid)
returns jsonb language sql stable security invoker set search_path=public as $$
 select jsonb_build_object(
  'run_id',e.id,'model',e.model,'config',e.configuration || jsonb_build_object('comparison_identity',e.comparison_identity),
  'attempt_id',a.id,'job_id',a.job_id,'reported_state',a.producer_state,
  'last_measurement',a.last_measured_at,'last_upload',a.received_at,
  'history_revision',e.current_attempt,'metric_source','api',
  'submitted',extract(epoch from e.created_at),
  'telemetry',coalesce(a.telemetry,'{}') || jsonb_build_object('heartbeat',extract(epoch from a.telemetry_at),'completion_reason',a.completion_reason),
  'attempts',coalesce((select jsonb_agg(jsonb_build_object('attempt_id',t.id,'job_id',t.job_id,
    'parent_attempt_id',t.parent_attempt_id,'resumed_from_epoch',t.resume_cutoff,'reported_state',t.producer_state,
    'last_measurement',t.last_measured_at,'last_upload',t.received_at,'batch_exit_code',t.batch_exit_code) order by t.started_at nulls first,t.received_at)
    from attempts t where t.project_id=e.project_id and t.experiment_id=e.id),'[]'),
  'history',coalesce((select jsonb_agg(h.metrics order by h.epoch) from
    (select * from monitor_history(e.project_id,e.id) limit 200) h),'[]'),
  'samples',coalesce((select jsonb_agg(s.payload || jsonb_build_object('heartbeat',extract(epoch from s.observed_at)) order by s.observed_at)
    from (select * from telemetry_samples where project_id=e.project_id and attempt_id=a.id order by observed_at desc limit 300) s),'[]'),
  'scheduler', (select jsonb_build_object('state',o.payload->>'state','observed_at',o.observed_at,'payload',o.payload)
    from scheduler_observations o where o.project_id=e.project_id and o.profile_id=e.profile_id and o.job_id=a.job_id)
 ) from experiments e left join attempts a on a.project_id=e.project_id and a.id=e.current_attempt
   where e.project_id=p_project and e.id=p_experiment;
$$;
revoke all on function public.monitor_run(uuid,uuid) from public;
grant execute on function public.monitor_run(uuid,uuid) to authenticated;

create or replace function public.monitor_overview(p_project uuid,p_limit integer default 50,p_offset integer default 0)
returns jsonb language sql stable security invoker set search_path=public as $$
 select coalesce(jsonb_agg(row order by created_at desc),'[]') from (
   select e.created_at, jsonb_build_object('run_id',e.id,'model',e.model,'attempt_id',a.id,'job_id',a.job_id,
     'reported_state',a.producer_state,'last_measurement',a.last_measured_at,'last_upload',a.received_at,
     'latest',(select metrics from monitor_history(e.project_id,e.id) order by epoch desc limit 1),
     'best',(select metrics from monitor_history(e.project_id,e.id) where metrics->>'action_mt5r' is not null order by (metrics->>'action_mt5r')::double precision desc,epoch limit 1),
     'scheduler',(select jsonb_build_object('state',o.payload->>'state','observed_at',o.observed_at) from scheduler_observations o
       where o.project_id=e.project_id and o.profile_id=e.profile_id and o.job_id=a.job_id)) as row
   from experiments e left join attempts a on a.project_id=e.project_id and a.id=e.current_attempt
   where e.project_id=p_project order by e.created_at desc limit least(200,greatest(1,p_limit)) offset greatest(0,p_offset)
 ) limited;
$$;
revoke all on function public.monitor_overview(uuid,integer,integer) from public;
grant execute on function public.monitor_overview(uuid,integer,integer) to authenticated;
