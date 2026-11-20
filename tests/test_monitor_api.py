"""API publication is optional, durable, bounded and isolated from training."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from src.monitoring.api_contracts import new_event,identifier,public_config,validate_event
from src.monitoring.outbox import Outbox,file_lock
from src.monitoring.api_transport import configuration,post,UploadError
from src.run_monitor import RunMonitor
from scripts.monitor_upload import upload_once
from scripts.monitor_backfill import prepare


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.outbox=Outbox(self.root,'legacy-experiment','attempt')

    def test_golden_contracts(self):
        fixtures=json.loads((Path(__file__).parent/'fixtures/monitoring/api-contracts.json').read_text())
        for value in fixtures['valid']:
            validate_event(value)
        for value in fixtures['invalid']:
            with self.assertRaises((ValueError,KeyError,TypeError)):
                validate_event(value)

    def test_identity_and_secret_path_filtering(self):
        self.assertEqual(identifier('old'),identifier('old'))
        config=public_config({'seed':42,'api_key':'SECRET','path':'/data/private','nested':{'token':'secret','model':'gru'},'file':'C:\\private'})
        self.assertEqual(config,{'seed':42,'nested':{'model':'gru'},'file':'[local path]'})

    def test_event_immutable_across_worker_restart(self):
        event=self.outbox.append('experiment_registered',{'model':'gru'})
        fresh=Outbox(self.root)
        self.assertEqual(fresh.batch()['events'][0],event)
        self.assertEqual(len(list(self.outbox.events.glob('*.json'))),1)

    def test_acknowledges_only_explicit_ids(self):
        first=self.outbox.append('experiment_registered',{'model':'gru'})
        second=self.outbox.append('attempt_started',{'job_id':'123'})
        batch=self.outbox.batch()
        self.outbox.acknowledge(batch,{'accepted':[first['event_id']]})
        self.assertEqual(self.outbox.batch()['events'],[second])
        with self.assertRaises(ValueError):
            self.outbox.acknowledge(batch,{'accepted':[str(uuid.uuid4())]})

    def test_two_producers_share_monotonic_sequence(self):
        other=Outbox(self.root,'legacy-experiment','attempt')
        a=self.outbox.append('progress',{'phase':'training'})
        b=other.append('progress',{'phase':'validation'})
        c=self.outbox.append('progress',{'phase':'training'})
        self.assertEqual([a['sequence'],b['sequence'],c['sequence']],[1,2,3])

    def test_crash_after_event_write_recovers_sequence(self):
        self.outbox.append('progress',{'phase':'training'})
        (self.root/'api-producer.json').unlink()
        fresh=Outbox(self.root,'legacy-experiment','attempt')
        self.assertEqual(fresh.append('progress',{'phase':'validation'})['sequence'],2)

    def test_compaction_preserves_scientific_events(self):
        self.outbox.append('experiment_registered',{'model':'sta2'})
        for _ in range(10):
            self.outbox.append('heartbeat',{'phase':'training'})
        self.outbox.append('epoch_committed',{'epoch':1,'history_revision':identifier('attempt'),'action_mt5r':0})
        self.outbox.compact(budget=1)
        self.assertEqual([r['kind'] for r in self.outbox.batch()['events']],['experiment_registered','epoch_committed'])

    def test_failed_network_never_acknowledges_events(self):
        self.outbox.append('progress',{'phase':'training'})
        with patch('scripts.monitor_upload.post',side_effect=UploadError('unreachable',True)):
            with self.assertRaises(UploadError):
                upload_once(self.outbox)
        self.assertEqual(len(self.outbox.batch()['events']),1)

    def test_network_replay_receipt_removes_only_committed_events(self):
        event=self.outbox.append('progress',{'phase':'training'})
        with patch('scripts.monitor_upload.post',return_value={'duplicates':[event['event_id']]}):
            upload_once(self.outbox)
        self.assertFalse(self.outbox.batch()['events'])

    def test_rejects_plain_http_and_url_credentials(self):
        key=self.root/'key';key.write_text('a'*64);key.chmod(0o600)
        for endpoint in ['http://example.test','https://secret@example.test','https://example.test?key=secret']:
            with patch.dict(os.environ,{'RND_MONITOR_API_URL':endpoint,'RND_MONITOR_UPLOAD_KEY_FILE':str(key)}):
                with self.assertRaises(UploadError):
                    configuration()

    def test_nonzero_rank_cannot_create_api_worker(self):
        monitor=RunMonitor()
        with patch.dict(os.environ,{'RANK':'1','RND_MONITOR_API_URL':'https://example.test','RND_MONITOR_UPLOAD_KEY_FILE':'/private'}),patch('src.run_monitor.subprocess.Popen') as spawn:
            monitor.configure(self.root/'nonzero','sta2')
        spawn.assert_not_called()

    def test_uploader_start_failure_cannot_fail_training(self):
        monitor=RunMonitor()
        with patch.dict(os.environ,{'RANK':'0','RND_MONITOR_API_URL':'https://example.test','RND_MONITOR_UPLOAD_KEY_FILE':'/private'}),patch('src.run_monitor.subprocess.Popen',side_effect=OSError('network')):
            with self.assertWarns(UserWarning):
                monitor.configure(self.root/'training','sta2')
        self.assertEqual(len(monitor.outbox.batch()['events']),2)

    def test_historical_import_is_repeatable_without_completion_claim(self):
        run=self.root/'run';run.mkdir()
        (run/'history.json').write_text('[{"epoch":1,"train_loss":2.5,"action_mt5r":0}]')
        (run/'run-manifest-123.json').write_text('{"configuration":{"seed":42,"path":"/private"}}')
        directory,count=prepare(run)
        before=Outbox(directory).batch()['events']
        prepare(run)
        self.assertEqual(Outbox(directory).batch()['events'],before)
        self.assertEqual(count,1)
        self.assertFalse(any(e['kind']=='training_finished' for e in before))

    def test_csv_import_converts_counts_and_missing_metrics_and_retries(self):
        run=self.root/'csv-run'
        logs=run/'logs/gru';logs.mkdir(parents=True)
        history=logs/'history.csv'
        history.write_text('epoch,train_loss,action_mt5r,train_usable_events,validation_events,best_epoch,rank_peak_reserved_gpu_bytes\n'
                           '1,2.5,0.0122,100,20,,nan\n')
        directory,count=prepare(run)
        events=Outbox(directory).batch()['events']
        payload=events[-1]['payload']
        self.assertEqual(payload['train_usable_events'],100)
        self.assertEqual(payload['validation_events'],20)
        self.assertEqual(payload['action_mt5r'],0.0122)
        self.assertIsNone(payload['best_epoch'])
        self.assertIsNone(payload['rank_peak_reserved_gpu_bytes'])
        self.assertEqual(count,1)
        # A previous failed import can already have queued lifecycle events.
        ledger=directory/'imported.json'
        ledger.write_text(json.dumps(json.loads(ledger.read_text())[:2]))
        epoch=next((directory/'api-outbox').glob('0000000003.json'))
        epoch.unlink()
        prepare(run)
        retried=Outbox(directory).batch()['events']
        self.assertEqual([e['event_id'] for e in retried],[e['event_id'] for e in events])
        self.assertEqual(retried[-1]['payload'],payload)


if __name__=='__main__':
    unittest.main()
