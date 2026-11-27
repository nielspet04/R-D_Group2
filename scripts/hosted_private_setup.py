"""Prepare SQL for a teammate's private dashboard project; no credentials required."""
import argparse
import json
from pathlib import Path
import uuid


def private_sql(namespace, user_id, name):
    user = str(uuid.UUID(user_id))
    project = str(uuid.uuid5(uuid.UUID(namespace), 'private:' + user))
    profile = str(uuid.uuid5(uuid.UUID(project), 'wice'))
    label = name.replace("'", "''")
    sql = f"""-- Private project: only the named user becomes a member.
begin;
do $$ begin
  if not exists(select 1 from auth.users where id='{user}') then
    raise exception 'User must sign in to the dashboard first';
  end if;
  if exists(select 1 from public.project_members where project_id='{project}' and user_id<>'{user}') then
    raise exception 'Project already has other members; inspect before continuing';
  end if;
end $$;
insert into public.monitor_projects(id,name) values ('{project}','{label}') on conflict(id) do nothing;
insert into public.source_profiles(project_id,id,label,cluster)
  values ('{project}','{profile}','VSC wICE','wice') on conflict(project_id,id) do nothing;
insert into public.project_members(project_id,user_id,role)
  values ('{project}','{user}','owner') on conflict(project_id,user_id) do update set role='owner';
commit;
"""
    return project, sql


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-id', required=True, help='Supabase user UUID after GitHub sign-in')
    parser.add_argument('--name', required=True, help='Private project display name')
    parser.add_argument('--config', type=Path, default=Path('configs/monitoring-hosted.local.json'))
    parser.add_argument('--output', type=Path, default=Path('results/monitoring/private-project.sql'))
    args = parser.parse_args()
    if not args.name.strip() or len(args.name) > 100 or '$$' in args.name:
        parser.error('Use a project name of 1–100 characters without $$')
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        project, sql = private_sql(config['projectId'], args.user_id, args.name.strip())
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sql, encoding='utf-8')
    print('Prepared private-project SQL: ' + str(args.output))
    print('Private project ID: ' + project)
    print('Apply this SQL in the existing Supabase project. Public website config is unchanged.')
