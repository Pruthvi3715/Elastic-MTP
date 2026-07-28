"""
Abstract Base Classes (ABCs) and Core Interfaces for Elastic-MTP.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import torch

class BaseRouter(ABC):
    """Abstract Base Class for all speculative horizon and tree routers."""
    
    @abstractmethod
    def evaluate_entropy(self, logits: torch.Tensor) -> float:
        """Evaluate token logit entropy H(P_t)."""
        pass
        
    @abstractmethod
    def determine_horizon(self, entropy_input: Union[float, torch.Tensor], aux_logits_list: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """Determine speculative prediction horizon or tree structure."""
        pass
        
    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return research telemetry metrics summary."""
        pass

class BaseAdapter(ABC):
    """Abstract Base Class for GLoRA multi-token prediction adapters."""
    
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Produce MTP candidate draft logits for speculative steps."""
        pass

class BaseCompressor(ABC):
    """Abstract Base Class for KV cache quantization compressors."""
    
    @abstractmethod
    def compress_kv(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compress Key and Value cache tensors."""
        pass

    @abstractmethod
    def decompress_kv(self, compressed_key: torch.Tensor, compressed_value: torch.Tensor, meta: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress Key and Value cache tensors."""
        pass
