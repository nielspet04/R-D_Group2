import subprocess
import unittest
from unittest.mock import patch

from scripts import monitor_follow as follow


class FollowTests(unittest.TestCase):
    @patch.object(follow.subprocess, 'run')
    def test_scheduler_failure_is_unknown(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, '', 'Unavailable')
        self.assertIsNone(follow.job_active('123', 'wice'))
        run.side_effect = subprocess.TimeoutExpired('squeue', 20)
        self.assertIsNone(follow.job_active('123', 'wice'))

    @patch.object(follow.subprocess, 'run')
    def test_exact_job_match_and_absence(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, '123\n', '')
        self.assertTrue(follow.job_active('123', 'wice'))
        run.return_value = subprocess.CompletedProcess([], 0, '', '')
        self.assertFalse(follow.job_active('123', 'wice'))

    @patch.object(follow.time, 'sleep')
    @patch.object(follow, 'upload_history', side_effect=[True, False, True, True])
    @patch.object(follow, 'job_active', side_effect=[None, False, False, False])
    def test_unknown_scheduler_and_failed_final_upload_are_retried(self, active, upload, sleep):
        self.assertEqual(follow.follow('run', '123', 'wice', 300, 1), 0)
        self.assertEqual(upload.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    @patch.object(follow.time, 'monotonic', side_effect=[0, 3601])
    @patch.object(follow, 'upload_history')
    def test_bounded_runtime(self, upload, monotonic):
        self.assertEqual(follow.follow('run', '123', 'wice', 300, 1), 1)
        upload.assert_not_called()


if __name__ == '__main__':
    unittest.main()
