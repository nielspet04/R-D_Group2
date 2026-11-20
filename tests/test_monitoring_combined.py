"""Offline regression checks for combined monitoring; no Torch or cloud account."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

from src.monitoring.store import connect, ingest, read_data
from src.monitoring.tracking import collect as cloud_collect
from src.run_monitor import RunMonitor
from scripts.monitor_snapshot import collect, bounded_text
from scripts.dashboard import Collector


def snapshot(stamp=100, **values):
    return {'collected_at':stamp, 'cluster':'test','jobs':[], 'runs':[], 'errors':[], **values}


class CombinedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'cache.sqlite3'

    def test_partial_read_preserves_history_and_success_time(self):
        ingest(self.db,snapshot(runs=[{'run_id':'r','history':[{'epoch':1,'action_mt5r':0}]}]))
        ingest(self.db,snapshot(200,runs=[{'run_id':'r'}], errors=['history unreadable']))
        data = read_data(self.db)
        self.assertEqual(data['runs'][0]['history'][0]['action_mt5r'],0)
        self.assertEqual(data['meta']['vsc']['last_success'],100)
        self.assertEqual(data['meta']['vsc']['status'],'partial')

    def test_out_of_order_does_not_restore_abandoned_epochs(self):
        ingest(self.db,snapshot(200,runs=[{'run_id':'r','history':[{'epoch':1}]}]))
        ingest(self.db,snapshot(100,runs=[{'run_id':'r','history':[{'epoch':1},{'epoch':2}]}]))
        self.assertEqual(len(read_data(self.db)['runs'][0]['history']),1)

    def test_cloud_cannot_override_local_metrics_or_scheduler(self):
        ingest(self.db,snapshot(jobs=[{'job_id':'1','state':'RUNNING'}],runs=[{'run_id':'r','history':[{'epoch':1,'action_mt5r':0.1}]}]))
        ingest(self.db,snapshot(200,runs=[{'run_id':'r','history':[{'epoch':1,'action_mt5r':0.8}],'wandb':{'state':'finished'}}]),source='wandb')
        data=read_data(self.db)
        self.assertEqual(data['runs'][0]['history'][0]['action_mt5r'],0.1)
        self.assertEqual(data['jobs'][0]['state'],'RUNNING')

    def test_legacy_backup_and_migration_are_repeatable(self):
        with closing(sqlite3.connect(self.db)) as db:
            db.execute('CREATE TABLE records (kind TEXT,id TEXT,payload TEXT,PRIMARY KEY(kind,id))')
            db.execute('INSERT INTO records VALUES (?,?,?)',('runs','r','{"run_id":"r"}'))
            db.commit()
        for _ in range(2):
            with connect(self.db):
                pass
        self.assertTrue(Path(str(self.db)+'.before-v2.sqlite3').exists())
        self.assertEqual(len(read_data(self.db)['runs']),1)

    def test_newer_database_is_not_modified(self):
        with closing(sqlite3.connect(self.db)) as db:
            db.execute('PRAGMA user_version=99')
        with self.assertRaises(ValueError):
            with connect(self.db):
                pass

    def test_future_snapshot_is_rejected(self):
        with self.assertRaises(ValueError):
            ingest(self.db,snapshot(schema_version=9))

    def test_project_switch_requires_separate_database(self):
        ingest(self.db,snapshot(project='team/one'),source='wandb')
        with self.assertRaises(ValueError):
            ingest(self.db,snapshot(200,project='team/two'),source='wandb')

    def test_metric_strings_invalid_values_and_zero(self):
        ingest(self.db,snapshot(runs=[{'run_id':'r','history':[
            {'epoch':'1','action_mt5r':'0','train_loss':'broken'},
            {'epoch':'bad','action_mt5r':1}]}]))
        history=read_data(self.db)['runs'][0]['history']
        self.assertEqual(history,[{'epoch':1,'action_mt5r':0.0,'train_loss':None}])

    def test_sta2_submission_dry_run_never_submits(self):
        from scripts import submit_monitored
        with patch('sys.argv',['submit_monitored.py','--job-type','sta2-full-h100','--dry-run']), patch('scripts.submit_monitored.subprocess.run') as execute:
            submit_monitored.main()
        execute.assert_not_called()

    def test_ambiguous_submission_is_recorded(self):
        from scripts import submit_monitored
        root=self.root / 'R-D_Group2'
        (root / 'scripts').mkdir(parents=True)
        replies=[subprocess.CompletedProcess([],0,'commit',''),subprocess.CompletedProcess([],0,'',''),subprocess.CompletedProcess([],0,'ambiguous','')]
        with patch.object(submit_monitored,'__file__',str(root / 'scripts/submit_monitored.py')), patch.dict('os.environ',{'VSC_DATA':str(root.parent)}), patch('sys.argv',['submit_monitored.py','--model','gru']), patch('scripts.submit_monitored.subprocess.run',side_effect=replies):
            with self.assertRaises(SystemExit):
                submit_monitored.main()
        metadata=json.loads(next((root / 'results/runs').glob('*/run.json')).read_text())
        self.assertEqual(metadata['state'],'SUBMISSION_UNKNOWN')

    def test_sta2_native_history_and_attempts(self):
        run=self.root / 'results/sta2/full-h100-123'
        run.mkdir(parents=True)
        (run / 'history.json').write_text('[{"epoch":1,"action_mt5r":0.25}]')
        (run / 'run-manifest-123.json').write_text('{"configuration":{"training":{"seed":42}}}')
        replies=[subprocess.CompletedProcess([],0,'123|sta2|RUNNING|gpu|1:00|node|0:0\n',''),subprocess.CompletedProcess([],0,'123|sta2|RUNNING|gpu|1:00|node|None\n','')]
        with patch('scripts.monitor_snapshot.subprocess.run',side_effect=replies):
            result=collect(self.root)
        self.assertEqual(result['runs'][0]['history'][0]['action_mt5r'],0.25)
        self.assertEqual(result['runs'][0]['job_id'],'123')

    def test_file_limit_is_enforced_before_decode(self):
        path=self.root/'large'
        path.write_bytes(b'x'*10)
        with self.assertRaises(ValueError):
            bounded_text(path,limit=5)

    def test_rank_one_does_not_write(self):
        monitor=RunMonitor()
        with patch.dict('os.environ',{'RANK':'1'}):
            monitor.configure(self.root/'output','sta2')
        self.assertIsNone(monitor.path)
        self.assertFalse((self.root/'output').exists())

    def test_resume_keeps_experiment_but_creates_attempt(self):
        monitor=RunMonitor()
        with patch.dict('os.environ',{'RANK':'0','RND_WANDB_MODE':'disabled'}):
            monitor.configure(self.root,'sta2')
            first=dict(monitor.metadata)
            monitor.configure(self.root,'sta2',resume='last.pt')
        self.assertEqual(first['experiment_id'],monitor.metadata['experiment_id'])
        self.assertNotEqual(first['attempt_id'],monitor.metadata['attempt_id'])

    def test_publisher_failure_does_not_escape_epoch(self):
        monitor=RunMonitor()
        with patch.dict('os.environ',{'RANK':'0','RND_WANDB_MODE':'online'}):
            monitor.configure(self.root,'sta2')
            with patch('src.run_monitor.subprocess.Popen',side_effect=OSError('failure')):
                with self.assertWarns(UserWarning):
                    monitor.epoch({'epoch':1,'action_mt5r':0})
        self.assertEqual(json.loads((monitor.attempt_dir/'epochs.json').read_text())[0]['epoch'],1)

    def test_cloud_requires_noninteractive_credentials(self):
        with patch.dict('os.environ',{'WANDB_API_KEY':''}):
            with self.assertRaises(ValueError):
                cloud_collect('team/project')

    def test_cloud_only_never_uses_ssh(self):
        args=Namespace(db=self.db,wandb_only=True,wandb_project=None)
        collector=Collector(args)
        with patch('scripts.dashboard.subprocess.run') as command:
            collector.refresh()
        command.assert_not_called()

    def test_cloud_resume_projects_correct_suffix(self):
        class Remote:
            def __init__(self,attempt,stamp,cutoff,epochs):
                self.config={'experiment_id':'r','attempt_id':attempt,'submitted':stamp,'resumed_from_epoch':cutoff}
                self.url='https://example.test/run'
                self.state='finished'
                self.path=['team','project',attempt]
                self.rows=[{'epoch':e,'validation/action_mt5r':e/10} for e in epochs]
            def scan_history(self,**kwargs):
                return iter(self.rows)
        from unittest.mock import Mock
        client=Mock()
        client.runs.return_value=[Remote('a',1,None,[1,2,3]),Remote('b',2,1,[2])]
        with patch('src.monitoring.tracking.api',return_value=client):
            result=cloud_collect('team/project')
        self.assertEqual([r['epoch'] for r in result['runs'][0]['history']],[1,2])


if __name__=='__main__':
    unittest.main()
