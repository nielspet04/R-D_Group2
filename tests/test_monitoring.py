"""Monitoring must preserve scheduler truth and never break training."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.dashboard import ingest, read_data
from scripts.monitor_snapshot import collect
from scripts import submit_monitored
from src.run_monitor import RunMonitor


class MonitoringTests(unittest.TestCase):
    def test_submission_exports_isolated_output_and_records_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "R-D_Group2"
            (root / "scripts").mkdir(parents=True)
            replies = [subprocess.CompletedProcess([], 0, "abc123\n", ""),
                       subprocess.CompletedProcess([], 0, "", ""),
                       subprocess.CompletedProcess([], 0, "123;wice\n", "")]
            with patch.object(submit_monitored, "__file__", str(root / "scripts/submit_monitored.py")), \
                 patch.dict("os.environ", {"VSC_DATA": str(root.parent)}), \
                 patch("sys.argv", ["submit_monitored.py", "--model", "gru"]), \
                 patch("scripts.submit_monitored.subprocess.run", side_effect=replies) as execute:
                submit_monitored.main()
            metadata = json.loads(next((root / "results/runs").glob("*/run.json")).read_text())
            self.assertEqual(metadata["job_id"], "123")
            self.assertEqual(metadata["state"], "PENDING")
            env = execute.call_args.kwargs["env"]
            self.assertEqual(env["RND_OUTPUT_ROOT"], metadata["output_root"])
            self.assertEqual(env["RND_MONITOR_RUN_DIR"], metadata["output_root"])

    def test_ingestion_keeps_terminal_jobs_and_deduplicates_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.sqlite3"
            snapshot = {"collected_at": 10, "cluster": "test", "errors": [],
                "jobs": [{"job_id": "123", "state": "RUNNING"}, {"job_id": "124", "state": "FAILED"}],
                "runs": [{"run_id": "r", "telemetry": {"heartbeat": 9}, "submitted": 1}]}
            ingest(path, snapshot)
            ingest(path, snapshot)
            self.assertEqual(len(read_data(path)["runs"][0]["samples"]), 1)
            ingest(path, {**snapshot, "jobs": [], "runs": []})
            states = {j["job_id"]: j["state"] for j in read_data(path)["jobs"]}
            self.assertEqual(states, {"123": "UNKNOWN", "124": "FAILED"})

    def test_snapshot_queue_overrides_accounting_and_reads_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "results/runs/r"
            (run / "logs/gru").mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"run_id": "r", "job_id": "123", "model": "gru"}))
            (run / "logs/gru/history.csv").write_text("epoch,train_loss\n1,2.5\n")
            results = [subprocess.CompletedProcess([], 0, "123|sta|PENDING|gpu|00:00|node|0:0\n", ""),
                       subprocess.CompletedProcess([], 0, "123|sta|RUNNING|gpu|00:01|node|None\n", "")]
            with patch("scripts.monitor_snapshot.subprocess.run", side_effect=results):
                snapshot = collect(root)
            self.assertEqual(snapshot["jobs"][0]["state"], "RUNNING")
            self.assertEqual(snapshot["runs"][0]["history"][0]["train_loss"], "2.5")

    def test_scheduler_failures_report_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("scripts.monitor_snapshot.subprocess.run", side_effect=OSError("not installed")):
                result = collect(Path(temporary))
            self.assertEqual(len(result["errors"]), 2)

    def test_telemetry_failure_does_not_escape_and_context_preserves_error(self):
        monitor = RunMonitor()
        monitor.path = "unwritable"
        with patch("src.run_monitor.subprocess.run", side_effect=OSError()), patch("pathlib.Path.mkdir", side_effect=OSError("disk")):
            with self.assertWarns(UserWarning):
                monitor.write()
        monitor.path = None
        with self.assertRaisesRegex(RuntimeError, "training problem"):
            with monitor:
                raise RuntimeError("training problem")


if __name__ == "__main__":
    unittest.main()
