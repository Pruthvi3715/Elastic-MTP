"""
Elastic-MTP Core Interfaces and Registry.
"""
from elastic_mtp.core.interfaces import BaseRouter, BaseAdapter, BaseCompressor
from elastic_mtp.core.registry import ComponentRegistry, register_router, register_adapter, register_compressor, build_router, build_compressor

__all__ = [
    "BaseRouter",
    "BaseAdapter",
    "BaseCompressor",
    "ComponentRegistry",
    "register_router",
    "register_adapter",
    "register_compressor",
    "build_router",
    "build_compressor"
]
