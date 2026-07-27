"""
Unit tests for Interactive CLI and Real Weight Evaluation Harness.
"""
import unittest
import torch
from src.inference_engine import ElasticMTPInferenceEngine
from real_weight_eval import BENCHMARK_DATASETS

class TestCLIAndRealWeightEval(unittest.TestCase):

    def test_benchmark_dataset_definitions(self):
        self.assertEqual(len(BENCHMARK_DATASETS), 3)
        self.assertIn("Wikitext", BENCHMARK_DATASETS[0]["domain"])
        self.assertIn("HumanEval", BENCHMARK_DATASETS[1]["domain"])
        self.assertIn("GSM8K", BENCHMARK_DATASETS[2]["domain"])

    def test_synthetic_engine_generation_for_datasets(self):
        engine = ElasticMTPInferenceEngine(model_name="synthetic")
        for item in BENCHMARK_DATASETS:
            res = engine.generate(item["prompt"], max_new_tokens=10, mode="elastic")
            self.assertGreater(res["tokens_generated"], 0)
            self.assertGreater(res["tokens_per_sec"], 0)

if __name__ == "__main__":
    unittest.main()
