-- Private research data; only the ingestion function has server write credentials.
create table public.monitor_projects (
  id uuid primary key default gen_random_uuid(), name text not null,
  created_at timestamptz not null default now()
);
create table public.project_members (
  project_id uuid references public.monitor_projects(id), user_id uuid not null,
  role text not null check (role in ('owner','viewer')), primary key(project_id,user_id)
);
create or replace function public.monitor_member(p_project uuid, owner_only boolean default false)
returns boolean language sql stable security definer set search_path = public as $$
  select exists(select 1 from project_members where project_id=p_project and user_id=auth.uid()
    and (not owner_only or role='owner'));
$$;
revoke all on function public.monitor_member(uuid,boolean) from public;
grant execute on function public.monitor_member(uuid,boolean) to authenticated;

create table public.source_profiles (
  project_id uuid references public.monitor_projects(id), id uuid default gen_random_uuid(),
  label text not null, cluster text not null default 'wice', primary key(project_id,id)
);
create table public.upload_credentials (
  id uuid primary key default gen_random_uuid(), project_id uuid not null,
  profile_id uuid not null, token_hash text not null unique, label text not null,
  scopes text[] not null default array['training:write'],
  created_at timestamptz default now(), expires_at timestamptz not null,
  revoked_at timestamptz, last_used_at timestamptz,
  rate_window timestamptz, rate_count integer not null default 0,
  foreign key(project_id,profile_id) references public.source_profiles(project_id,id),
  check(scopes <@ array['training:write','scheduler:write'])
);
create table public.experiments (
  project_id uuid references public.monitor_projects(id), id uuid not null,
  profile_id uuid not null, model text not null, configuration jsonb not null default '{}',
  comparison_identity jsonb, current_attempt uuid, created_at timestamptz default now(),
  primary key(project_id,id), foreign key(project_id,profile_id) references public.source_profiles(project_id,id)
);
create table public.attempts (
  project_id uuid not null, id uuid not null, experiment_id uuid not null,
  parent_attempt_id uuid, resume_cutoff integer check(resume_cutoff>=0),
  job_id text, cluster text, producer_state text not null default 'registered',
  started_at timestamptz, last_measured_at timestamptz, received_at timestamptz default now(),
  last_sequence bigint not null default 0, telemetry jsonb not null default '{}',
  completion_reason text, batch_exit_code integer, telemetry_at timestamptz,
  primary key(project_id,id),
  foreign key(project_id,experiment_id) references public.experiments(project_id,id),
  foreign key(project_id,parent_attempt_id) references public.attempts(project_id,id)
);
alter table public.experiments add foreign key(project_id,current_attempt) references public.attempts(project_id,id);
create table public.epoch_records (
  project_id uuid not null, attempt_id uuid not null, epoch integer not null check(epoch>0),
  metrics jsonb not null, observed_at timestamptz not null,
  primary key(project_id,attempt_id,epoch),
  foreign key(project_id,attempt_id) references public.attempts(project_id,id)
);
create table public.telemetry_samples (
  project_id uuid not null, attempt_id uuid not null, sequence bigint not null,
  observed_at timestamptz not null, payload jsonb not null,
  primary key(project_id,attempt_id,sequence),
  foreign key(project_id,attempt_id) references public.attempts(project_id,id)
);
create index telemetry_by_time on public.telemetry_samples(project_id,attempt_id,observed_at desc);
create table public.telemetry_hourly (
  project_id uuid not null, attempt_id uuid not null, hour timestamptz not null,
  device text not null, samples integer not null, min_utilization real, mean_utilization real, max_utilization real,
  primary key(project_id,attempt_id,hour,device), foreign key(project_id,attempt_id) references public.attempts(project_id,id)
);
create table public.scheduler_observations (
  project_id uuid not null, profile_id uuid not null, job_id text not null,
  observed_at timestamptz not null, payload jsonb not null,
  primary key(project_id,profile_id,job_id),
  foreign key(project_id,profile_id) references public.source_profiles(project_id,id)
);
create table public.ingest_receipts (
  project_id uuid not null references public.monitor_projects(id), event_id uuid not null,
  attempt_id uuid not null, sequence bigint not null, content_hash text not null,
  received_at timestamptz default now(), primary key(project_id,event_id),
  unique(project_id,attempt_id,sequence)
);
create index receipts_by_attempt on public.ingest_receipts(project_id,attempt_id,sequence);

-- No grants are inherited from provider defaults. Anonymous readers see no data.
do $$ declare tab text; begin
  foreach tab in array array['monitor_projects','project_members','source_profiles','upload_credentials',
    'experiments','attempts','epoch_records','telemetry_samples','telemetry_hourly','scheduler_observations','ingest_receipts'] loop
    execute format('alter table public.%I enable row level security',tab);
    execute format('revoke all on public.%I from anon, authenticated',tab);
    execute format('grant all on public.%I to service_role',tab);
    if tab not in ('upload_credentials','ingest_receipts') then
      execute format('grant select on public.%I to authenticated',tab);
      if tab='monitor_projects' then
        execute 'create policy member_read on public.monitor_projects for select to authenticated using(public.monitor_member(id))';
      else
        execute format('create policy member_read on public.%I for select to authenticated using(public.monitor_member(project_id))',tab);
      end if;
    end if;
  end loop;
end $$;

create or replace function public.monitor_ingest(p_credential uuid, p_events jsonb)
returns jsonb language plpgsql security definer set search_path = public as $$
declare
  cred upload_credentials%rowtype; ev jsonb; body jsonb; event_uuid uuid; experiment_uuid uuid;
  attempt_uuid uuid; parent_uuid uuid; seq bigint; kind text; stamp timestamptz;
  hashed text; previous_hash text; accepted jsonb := '[]'; duplicates jsonb := '[]';
  exp experiments%rowtype; att attempts%rowtype; existing_metrics jsonb; job jsonb;
begin
  select * into cred from upload_credentials where id=p_credential for update;
  if not found or cred.revoked_at is not null or cred.expires_at<=now() then
    raise exception 'credential_denied' using errcode='42501';
  end if;
  if jsonb_typeof(p_events)<>'array' or jsonb_array_length(p_events) not between 1 and 50
      or octet_length(p_events::text)>262144 then raise exception 'invalid_batch'; end if;
  if cred.rate_window is not null and cred.rate_window>now()-interval '1 minute' and cred.rate_count>=10 then
    raise exception 'rate_limited';
  end if;
  update upload_credentials set
    rate_count=case when rate_window>now()-interval '1 minute' then rate_count+1 else 1 end,
    rate_window=case when rate_window>now()-interval '1 minute' then rate_window else now() end,
    last_used_at=now() where id=cred.id;
  -- Serialize projection changes for a small project, independent of Edge worker lifetime.
  perform 1 from monitor_projects where id=cred.project_id for update;
  for ev in select value from jsonb_array_elements(p_events) loop
    event_uuid := (ev->>'event_id')::uuid; experiment_uuid := (ev->>'experiment_id')::uuid;
    attempt_uuid := (ev->>'attempt_id')::uuid; seq := (ev->>'sequence')::bigint;
    kind := ev->>'kind'; body := ev->'payload'; stamp := (ev->>'observed_at')::timestamptz;
    if seq<=0 or seq>=9007199254740992 or stamp>now()+interval '10 minutes' or jsonb_typeof(body)<>'object' then
      raise exception 'invalid_event';
    end if;
    if (kind='scheduler_snapshot' and not 'scheduler:write'=any(cred.scopes)) or
       (kind<>'scheduler_snapshot' and not 'training:write'=any(cred.scopes)) then
      raise exception 'scope_denied' using errcode='42501';
    end if;
    hashed := encode(sha256(convert_to(ev::text,'UTF8')),'hex');
    select content_hash into previous_hash from ingest_receipts where project_id=cred.project_id and event_id=event_uuid;
    if found then
      if previous_hash<>hashed then raise exception 'event_conflict'; end if;
      duplicates := duplicates || jsonb_build_array(event_uuid); continue;
    end if;
    if exists(select 1 from ingest_receipts where project_id=cred.project_id and attempt_id=attempt_uuid and sequence=seq) then
      raise exception 'sequence_conflict';
    end if;
    if kind='scheduler_snapshot' then
      for job in select value from jsonb_array_elements(body->'jobs') loop
        insert into scheduler_observations values(cred.project_id,cred.profile_id,job->>'job_id',stamp,job)
        on conflict(project_id,profile_id,job_id) do update set observed_at=excluded.observed_at,payload=excluded.payload
          where scheduler_observations.observed_at<excluded.observed_at;
      end loop;
    else
      select * into exp from experiments where project_id=cred.project_id and id=experiment_uuid;
      if kind='experiment_registered' then
        if found and (exp.profile_id<>cred.profile_id or exp.model<>body->>'model') then raise exception 'identity_conflict'; end if;
        insert into experiments(project_id,id,profile_id,model,configuration,comparison_identity)
          values(cred.project_id,experiment_uuid,cred.profile_id,body->>'model',coalesce(body->'configuration','{}'),body->'comparison_identity')
          on conflict(project_id,id) do update set
            configuration=case when experiments.configuration='{}' then excluded.configuration else experiments.configuration end,
            comparison_identity=coalesce(experiments.comparison_identity,excluded.comparison_identity);
      elsif not found or exp.profile_id<>cred.profile_id then raise exception 'experiment_not_registered'; end if;
      select * into att from attempts where project_id=cred.project_id and id=attempt_uuid;
      if not found then
        insert into attempts(project_id,id,experiment_id) values(cred.project_id,attempt_uuid,experiment_uuid);
      elsif att.experiment_id<>experiment_uuid then raise exception 'attempt_identity_conflict'; end if;
      if kind in ('attempt_started','attempt_submitted') then
        if att.job_id is not null and body->>'job_id' is not null and att.job_id<>body->>'job_id' then raise exception 'job_identity_conflict'; end if;
        update attempts set job_id=coalesce(job_id,body->>'job_id'),cluster=coalesce(cluster,body->>'cluster'),
          started_at=case when kind='attempt_started' then coalesce(started_at,stamp) else started_at end
          where project_id=cred.project_id and id=attempt_uuid;
        update experiments set current_attempt=attempt_uuid where project_id=cred.project_id and id=experiment_uuid and current_attempt is null;
      elsif kind='resume_confirmed' then
        parent_uuid := (body->>'parent_attempt_id')::uuid;
        if parent_uuid=attempt_uuid or not exists(select 1 from attempts where project_id=cred.project_id and id=parent_uuid and experiment_id=experiment_uuid) then
          raise exception 'parent_not_registered';
        end if;
        if exists(with recursive ancestors as (
          select id,parent_attempt_id,0 as depth from attempts where project_id=cred.project_id and id=parent_uuid
          union all select a.id,a.parent_attempt_id,p.depth+1 from ancestors p join attempts a
            on a.project_id=cred.project_id and a.id=p.parent_attempt_id where p.depth<100
        ) select 1 from ancestors where id=attempt_uuid) then raise exception 'branch_conflict'; end if;
        -- Parent must be current or an idempotent declaration on this branch. Prevent late branch activation.
        select * into exp from experiments where project_id=cred.project_id and id=experiment_uuid;
        if exp.current_attempt not in (parent_uuid,attempt_uuid) then raise exception 'branch_conflict'; end if;
        if att.parent_attempt_id is not null and (att.parent_attempt_id<>parent_uuid or att.resume_cutoff<>(body->>'cutoff')::int) then raise exception 'resume_conflict'; end if;
        update attempts set parent_attempt_id=parent_uuid,resume_cutoff=(body->>'cutoff')::int where project_id=cred.project_id and id=attempt_uuid;
        update experiments set current_attempt=attempt_uuid where project_id=cred.project_id and id=experiment_uuid;
      elsif kind='epoch_committed' then
        if (body->>'history_revision')::uuid<>attempt_uuid or (body->>'epoch')::int<1 then raise exception 'invalid_epoch'; end if;
        select metrics into existing_metrics from epoch_records where project_id=cred.project_id and attempt_id=attempt_uuid and epoch=(body->>'epoch')::int;
        if found and existing_metrics<>body then raise exception 'epoch_conflict'; end if;
        insert into epoch_records values(cred.project_id,attempt_uuid,(body->>'epoch')::int,body,stamp) on conflict do nothing;
      elsif kind='heartbeat' then
        insert into telemetry_samples values(cred.project_id,attempt_uuid,seq,stamp,body);
      elsif kind not in ('experiment_registered','progress','submission_unknown','training_finished','training_failed','batch_exit_reported') then
        raise exception 'unknown_kind';
      end if;
      -- Older observations are retained but cannot regress lifecycle or resource projections.
      update attempts set received_at=now(), last_measured_at=stamp,last_sequence=seq,
        producer_state=case when kind in ('attempt_started','attempt_submitted','submission_unknown','training_finished','training_failed','batch_exit_reported') then kind else producer_state end,
        telemetry=case when kind in ('heartbeat','progress') then telemetry || body else telemetry end,
        telemetry_at=case when kind in ('heartbeat','progress') then stamp else telemetry_at end,
        completion_reason=case when kind='training_finished' then body->>'completion_reason' else completion_reason end,
        batch_exit_code=case when kind='batch_exit_reported' then (body->>'exit_code')::int else batch_exit_code end
        where project_id=cred.project_id and id=attempt_uuid and last_sequence<seq;
    end if;
    insert into ingest_receipts(project_id,event_id,attempt_id,sequence,content_hash) values(cred.project_id,event_uuid,attempt_uuid,seq,hashed);
    accepted := accepted || jsonb_build_array(event_uuid);
  end loop;
  return jsonb_build_object('accepted',accepted,'duplicates',duplicates);
end $$;
revoke all on function public.monitor_ingest(uuid,jsonb) from public,anon,authenticated;
grant execute on function public.monitor_ingest(uuid,jsonb) to service_role;

-- Resolve the current branch, retaining only the inherited checkpoint prefix.
create or replace function public.monitor_history(p_project uuid,p_experiment uuid,p_attempt uuid default null)
returns table(epoch integer,metrics jsonb,attempt_id uuid,observed_at timestamptz)
language sql stable security invoker set search_path=public as $$
  with recursive lineage as (
    select a.id,a.parent_attempt_id,a.resume_cutoff,2147483647 as cutoff,0 as depth
      from attempts a join experiments e on e.project_id=a.project_id and coalesce(p_attempt,e.current_attempt)=a.id and a.experiment_id=e.id
      where e.project_id=p_project and e.id=p_experiment
    union all
    select p.id,p.parent_attempt_id,p.resume_cutoff,least(l.cutoff,l.resume_cutoff),l.depth+1
      from lineage l join attempts p on p.project_id=p_project and p.id=l.parent_attempt_id where l.depth<100
  ) select distinct on (r.epoch) r.epoch,r.metrics,r.attempt_id,r.observed_at
      from lineage l join epoch_records r on r.project_id=p_project and r.attempt_id=l.id and r.epoch<=l.cutoff
      order by r.epoch,l.depth;
$$;
revoke all on function public.monitor_history(uuid,uuid,uuid) from public;
grant execute on function public.monitor_history(uuid,uuid,uuid) to authenticated,service_role;

create or replace function public.monitor_retention() returns void
language plpgsql security definer set search_path=public as $$
begin
  insert into telemetry_hourly
    select s.project_id,s.attempt_id,date_trunc('hour',s.observed_at),g->>'uuid',count(*)::int,
      min((g->>'utilization')::real),avg((g->>'utilization')::real),max((g->>'utilization')::real)
      from telemetry_samples s cross join lateral jsonb_array_elements(coalesce(s.payload->'gpus','[]')) g
      where s.observed_at<now()-interval '7 days' and g->>'uuid' is not null
      group by s.project_id,s.attempt_id,date_trunc('hour',s.observed_at),g->>'uuid'
    on conflict(project_id,attempt_id,hour,device) do update set
      samples=telemetry_hourly.samples+excluded.samples,
      min_utilization=least(telemetry_hourly.min_utilization,excluded.min_utilization),
      mean_utilization=(telemetry_hourly.mean_utilization*telemetry_hourly.samples+excluded.mean_utilization*excluded.samples)/(telemetry_hourly.samples+excluded.samples),
      max_utilization=greatest(telemetry_hourly.max_utilization,excluded.max_utilization);
  delete from telemetry_samples where observed_at<now()-interval '7 days';
  delete from telemetry_hourly where hour<now()-interval '30 days';
end $$;
revoke all on function public.monitor_retention() from public,anon,authenticated;
grant execute on function public.monitor_retention() to service_role;
