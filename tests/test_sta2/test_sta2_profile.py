import unittest
from unittest.mock import patch

import torch

from scripts.profile_sta2 import precision_context


class ProfilePrecisionTests(unittest.TestCase):
    def test_fp32_does_not_enable_cuda_autocast(self):
        with patch("scripts.profile_sta2.torch.autocast", side_effect=AssertionError("unexpected autocast")):
            with precision_context("fp32"):
                self.assertEqual(torch.ones(1).dtype, torch.float32)

    def test_bf16_selects_cuda_bfloat16(self):
        with patch("scripts.profile_sta2.torch.autocast") as autocast:
            precision_context("bf16")
        autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)

    def test_unknown_precision_is_rejected(self):
        with self.assertRaises(ValueError):
            precision_context("fp16")
