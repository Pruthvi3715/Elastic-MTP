"""
Configuration parameters for Elastic-MTP Research Prototype.
"""
import os
import torch

class ElasticMTPConfig:
    # Model Selection
    DEFAULT_MODEL_NAME: str = "synthetic"
    
    # Device & Precision
    DEVICE: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DTYPE: torch.dtype = torch.float16 if DEVICE != "cpu" else torch.float32
    
    # Entropy & Horizon Thresholds (AutoResearch Optimized)
    ENTROPY_LOW_THRESHOLD: float = 1.50   # H(P) < 1.50 -> High confidence -> k = 8
    ENTROPY_HIGH_THRESHOLD: float = 2.0  # H(P) > 2.0 -> High uncertainty -> k = 1
    
    # Dynamic Horizons (Draft depths)
    K_MAX: int = 8   # Max spec depth for highly predictable text
    K_MED: int = 4   # Standard spec depth
    K_MIN: int = 1   # Baseline next-token prediction
    
    # Hallucination / Contradiction Safeguard (AutoResearch Optimized)
    CONTRADICTION_THRESHOLD: float = 0.30 
    
    # Output Directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR: str = os.path.join(BASE_DIR, "benchmark", "results")
    PLOTS_DIR: str = os.path.join(BASE_DIR, "benchmark", "plots")

# Ensure output directories exist
os.makedirs(ElasticMTPConfig.RESULTS_DIR, exist_ok=True)
os.makedirs(ElasticMTPConfig.PLOTS_DIR, exist_ok=True)
