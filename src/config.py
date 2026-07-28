"""
Configuration parameters for Elastic-MTP Research Prototype.
"""
import os
import torch

class ElasticMTPConfig:
    # Model Selection
    DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Device & Precision
    DEVICE: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DTYPE: torch.dtype = torch.float16 if DEVICE != "cpu" else torch.float32
    
    # Entropy & Horizon Thresholds (Single Source of Truth)
    TAU_ENTROPY: float = 1.50      # Calibrated threshold H(P) < 1.50 -> K scaling up to K_MAX
    TAU_DIVERGENCE: float = 0.30   # D_KL threshold -> Fallback safeguard
    ENTROPY_LOW_THRESHOLD: float = 1.50
    ENTROPY_HIGH_THRESHOLD: float = 2.00
    CONTRADICTION_THRESHOLD: float = 0.30 
    
    # Dynamic Horizons (Draft depths)
    K_MAX: int = 8   # Max spec depth for highly predictable text
    K_MED: int = 4   # Standard spec depth
    K_MIN: int = 1   # Baseline next-token prediction 
    
    # Output Directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR: str = os.path.join(BASE_DIR, "benchmark", "results")
    PLOTS_DIR: str = os.path.join(BASE_DIR, "benchmark", "plots")

# Ensure output directories exist
os.makedirs(ElasticMTPConfig.RESULTS_DIR, exist_ok=True)
os.makedirs(ElasticMTPConfig.PLOTS_DIR, exist_ok=True)
