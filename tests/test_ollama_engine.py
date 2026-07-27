"""
Unit tests for Ollama Engine Bridge module.
"""
import unittest
from src.ollama_engine import OllamaElasticEngine

class TestOllamaEngine(unittest.TestCase):

    def test_ollama_engine_initialization(self):
        engine = OllamaElasticEngine(model_name="llama3.2")
        self.assertEqual(engine.model_name, "llama3.2")
        self.assertIsInstance(engine.is_available(), bool)

    def test_ollama_model_list(self):
        engine = OllamaElasticEngine()
        models = engine.list_models()
        self.assertIsInstance(models, list)

if __name__ == "__main__":
    unittest.main()
