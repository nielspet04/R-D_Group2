"""Prepare PUBLIC website config and owner-bootstrap SQL. Never accepts secret keys."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
import uuid

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supabase-url',required=True)
    parser.add_argument('--publishable-key',required=True)
    parser.add_argument('--owner-id',help='Supabase user UUID shown after browser sign-in')
    args=parser.parse_args()
    parts=urlsplit(args.supabase_url)
    if parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.path not in ('','/') or parts.query or parts.fragment:
        parser.error('Use the HTTPS Supabase project URL without credentials or paths')
    if not args.publishable_key.startswith('sb_publishable_'):
        parser.error('Use the public sb_publishable_ key, never the secret/service-role key')
    path=Path('configs/monitoring-hosted.local.json')
    previous=json.loads(path.read_text()) if path.exists() else {}
    project=previous.get('projectId') or str(uuid.uuid4())
    profile=str(uuid.uuid5(uuid.UUID(project),'wice'))
    path.write_text(json.dumps({'supabaseUrl':args.supabase_url.rstrip('/'),'publishableKey':args.publishable_key,'projectId':project},indent=2)+'\n')
    output=Path('results/monitoring/hosted-bootstrap.sql');output.parent.mkdir(parents=True,exist_ok=True)
    sql=f"begin;\ninsert into public.monitor_projects(id,name) values ('{project}','Training Observatory') on conflict do nothing;\ninsert into public.source_profiles(project_id,id,label,cluster) values ('{project}','{profile}','VSC wICE','wice') on conflict do nothing;\n"
    if args.owner_id:
        owner=str(uuid.UUID(args.owner_id))
        sql+=f"insert into public.project_members(project_id,user_id,role) values ('{project}','{owner}','owner') on conflict(project_id,user_id) do update set role='owner';\n"
    sql+='commit;\n'
    output.write_text(sql)
    print('Prepared public config: '+str(path))
    print('Apply owner bootstrap in Supabase SQL Editor: '+str(output))
    print('Project ID: '+project)
