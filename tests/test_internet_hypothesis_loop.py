"""
Unit tests for Autonomous Internet-Driven Hypothesis Research Daemon (src/internet_hypothesis_autoresearch.py).
"""
import os
import shutil
import unittest
import torch

from src.internet_hypothesis_autoresearch import InternetResearchLoopManager, ResearchHypothesis


class TestInternetHypothesisLoop(unittest.TestCase):

    def setUp(self):
        self.manager = InternetResearchLoopManager()

    def test_fetch_literature_hypotheses(self):
        hypotheses = self.manager.fetch_literature_hypotheses()
        self.assertGreater(len(hypotheses), 0)
        h1 = hypotheses[0]
        self.assertIsInstance(h1, ResearchHypothesis)
        self.assertIn("EAGLE-3", h1.literature_source)

    def test_evaluate_hypothesis_promotion(self):
        self.manager.run_unit_tests = lambda: True
        h = ResearchHypothesis(
            hypothesis_id=99,
            title="Test Promotion Hypothesis",
            literature_source="Unit Test",
            rationale="Valid test patch",
            patch_config={"TAU_ENTROPY": 1.95}
        )

        res = self.manager.evaluate_hypothesis(h)
        self.assertTrue(res["tests_passed"])
        self.assertEqual(res["status"], "PROMOTED_PARETO_BEST")
        self.assertGreater(self.manager.best_score, -float("inf"))

    def test_evaluate_hypothesis_rollback_on_test_failure(self):
        # Mock run_unit_tests failure
        self.manager.run_unit_tests = lambda: False

        h = ResearchHypothesis(
            hypothesis_id=100,
            title="Failing Hypothesis",
            literature_source="Unit Test",
            rationale="Triggers test failure",
            patch_config={"TAU_ENTROPY": -1.0}
        )

        res = self.manager.evaluate_hypothesis(h)
        self.assertFalse(res["tests_passed"])
        self.assertEqual(res["status"], "REJECTED_TEST_FAILURE")


if __name__ == "__main__":
    unittest.main()
