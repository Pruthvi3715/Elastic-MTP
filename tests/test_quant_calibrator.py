"""
Unit tests for QuantizationAwareCalibrator.
"""
import pytest
from src.quant_aware_calibrator import QuantizationAwareCalibrator

def test_adjusted_threshold_calculation():
    calibrator = QuantizationAwareCalibrator(base_tau_entropy=1.50)
    
    # FP16 should equal base tau
    assert calibrator.compute_adjusted_threshold("FP16") == 1.50
    
    # INT4 should have higher tau to compensate for entropy shift
    tau_int4 = calibrator.compute_adjusted_threshold("INT4")
    assert tau_int4 > 1.50
    assert tau_int4 == round(1.50 + 1.15 * 0.45, 3)

def test_quantization_decay_evaluation():
    calibrator = QuantizationAwareCalibrator(base_tau_entropy=1.50)
    telemetry = calibrator.evaluate_quantization_decay()
    
    assert "FP16" in telemetry
    assert "INT4" in telemetry
    assert "TurboQuant_3.5bit" in telemetry
    
    # Recalibrated K should maintain or recover horizon under INT4
    assert telemetry["INT4"]["recalibrated_k"] >= telemetry["INT4"]["uncalibrated_k"]
