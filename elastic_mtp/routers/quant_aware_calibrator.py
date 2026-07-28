"""
Quantization-Aware Horizon Recalibrator for Elastic-MTP.
"""
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from elastic_mtp.config import ElasticMTPConfig
from elastic_mtp.routers.elastic_horizon_router import DynamicHorizonRouter
from elastic_mtp.core.registry import register_router

@register_router("quant_calibrator")
class QuantizationAwareCalibrator:
    """Recalibrates entropy thresholds based on KV quantization bit depth and noise characteristics."""
    
    QUANTIZATION_ENTROPY_SHIFTS = {
        "FP16": 0.00,
        "INT8": 0.12,
        "INT4": 0.45,
        "TurboQuant_3.5bit": 0.38,
        "TurboQuant_3bit": 0.52
    }
    
    def __init__(self, base_tau_entropy: float = ElasticMTPConfig.TAU_ENTROPY):
        self.base_tau_entropy = base_tau_entropy
        
    def compute_adjusted_threshold(self, precision_mode: str, measured_entropy_shift: Optional[float] = None) -> float:
        if measured_entropy_shift is not None:
            delta_h = measured_entropy_shift
        else:
            delta_h = self.QUANTIZATION_ENTROPY_SHIFTS.get(precision_mode, 0.0)
            
        alpha = 1.15
        tau_adjusted = self.base_tau_entropy + alpha * delta_h
        return round(tau_adjusted, 3)

    def get_recalibrated_router(self, precision_mode: str, max_k: int = ElasticMTPConfig.K_MAX) -> DynamicHorizonRouter:
        tau_adj = self.compute_adjusted_threshold(precision_mode)
        router = DynamicHorizonRouter(
            tau_entropy=tau_adj,
            tau_divergence=ElasticMTPConfig.TAU_DIVERGENCE,
            max_k=max_k
        )
        return router

    def evaluate_quantization_decay(self, precision_modes: List[str] = None) -> Dict[str, Any]:
        if precision_modes is None:
            precision_modes = ["FP16", "INT8", "INT4", "TurboQuant_3.5bit"]
            
        results = {}
        for mode in precision_modes:
            uncal_router = DynamicHorizonRouter(tau_entropy=self.base_tau_entropy)
            cal_router = self.get_recalibrated_router(mode)
            
            delta_h = self.QUANTIZATION_ENTROPY_SHIFTS.get(mode, 0.0)
            base_h = 0.50
            quant_h = base_h + delta_h
            
            k_uncal = uncal_router.determine_horizon(quant_h)["target_k"]
            k_cal = cal_router.determine_horizon(quant_h)["target_k"]
            
            results[mode] = {
                "precision": mode,
                "entropy_shift_delta_h": delta_h,
                "uncalibrated_tau": self.base_tau_entropy,
                "uncalibrated_k": k_uncal,
                "recalibrated_tau": cal_router.tau_entropy,
                "recalibrated_k": k_cal,
                "horizon_recovery": k_cal - k_uncal
            }
            
        return results
