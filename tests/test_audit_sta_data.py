"""Tests for the STA data-audit helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.audit_sta_data import (
    count_context_statuses,
    validate_split_integrity,
    validate_video_coverage,
)


def annotation(
    narration_id: str,
    participant_id: str,
    video_id: str,
    start_timestamp: str,
) -> dict:
    return {
        "narration_id": narration_id,
        "participant_id": participant_id,
        "video_id": video_id,
        "start_timestamp": start_timestamp,
    }


class STAAuditTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            [
                annotation("train_full", "P01", "P01_01", "00:00:20.000"),
                annotation("train_partial", "P01", "P01_01", "00:00:04.000"),
            ]
        )
        self.validation = pd.DataFrame(
            [
                annotation(
                    "validation_none",
                    "P02",
                    "P02_01",
                    "00:00:00.800",
                )
            ]
        )
        self.all_annotations = pd.concat(
            [self.train, self.validation],
            ignore_index=True,
        )

    def test_valid_disjoint_split_passes(self):
        validate_split_integrity(
            self.all_annotations,
            self.train,
            self.validation,
        )

    def test_participant_overlap_is_rejected(self):
        invalid_validation = self.validation.copy()
        invalid_validation.loc[0, "participant_id"] = "P01"

        with self.assertRaises(ValueError):
            validate_split_integrity(
                self.all_annotations,
                self.train,
                invalid_validation,
            )

    def test_narration_overlap_is_rejected(self):
        invalid_validation = self.validation.copy()
        invalid_validation.loc[0, "narration_id"] = "train_full"

        with self.assertRaises(ValueError):
            validate_split_integrity(
                self.all_annotations,
                self.train,
                invalid_validation,
            )

    def test_context_counts(self):
        counts = count_context_statuses(self.all_annotations)

        self.assertEqual(
            counts,
            {
                "full": 1,
                "partial": 1,
                "no_context": 1,
            },
        )

    def test_complete_video_coverage_passes(self):
        videos = {
            "P01_01": Path("/dataset/P01_01.mp4"),
            "P02_01": Path("/dataset/P02_01.mp4"),
        }

        validate_video_coverage(self.all_annotations, videos)

    def test_missing_video_is_rejected(self):
        videos = {
            "P01_01": Path("/dataset/P01_01.mp4"),
        }

        with self.assertRaises(ValueError):
            validate_video_coverage(self.all_annotations, videos)


if __name__ == "__main__":
    unittest.main()