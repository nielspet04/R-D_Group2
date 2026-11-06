"""Hand-checkable tests for the project evaluation utilities."""

import unittest

import numpy as np

from src.evaluation import (
    action_labels_from_pairs,
    build_action_pairs,
    class_mean_top_k_recall,
    joint_action_scores,
    top_k_accuracy,
)
from src.sta_models import FrequencyPriorSTA


class EvaluationTests(unittest.TestCase):
    def test_class_mean_recall_weights_classes_equally(self):
        labels = np.array([0, 0, 0, 1])
        scores = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.8, 0.2],
            ]
        )
        self.assertEqual(top_k_accuracy(labels, scores, k=1), 0.75)
        self.assertEqual(class_mean_top_k_recall(labels, scores, k=1), 0.5)

    def test_joint_action_scores_use_only_known_pairs(self):
        action_pairs = build_action_pairs(
            np.array([0, 1]),
            np.array([1, 0]),
        )
        verb_scores = np.array([[0.8, 0.2]])
        noun_scores = np.array([[0.25, 0.75]])
        scores = joint_action_scores(verb_scores, noun_scores, action_pairs)
        labels = action_labels_from_pairs(
            np.array([0]), np.array([1]), action_pairs
        )

        self.assertEqual(scores.shape, (1, 2))
        self.assertEqual(top_k_accuracy(labels, scores, k=1), 1.0)

    def test_frequency_prior_returns_requested_shapes(self):
        model = FrequencyPriorSTA.fit(
            verb_labels=np.array([0, 0, 1]),
            noun_labels=np.array([2, 2, 3]),
            num_verbs=3,
            num_nouns=4,
        )
        verb_scores, noun_scores = model.predict(sample_count=5)

        self.assertEqual(verb_scores.shape, (5, 3))
        self.assertEqual(noun_scores.shape, (5, 4))
        np.testing.assert_allclose(verb_scores.sum(axis=1), 1.0)
        np.testing.assert_allclose(noun_scores.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()

