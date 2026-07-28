"""
Component Registry Pattern for Elastic-MTP Routers, Adapters, and Compressors.
"""
from typing import Dict, Type, Any, Callable
from elastic_mtp.core.interfaces import BaseRouter, BaseAdapter, BaseCompressor

class ComponentRegistry:
    """Central registry mapping component names to their implementation classes."""
    _ROUTERS: Dict[str, Type[BaseRouter]] = {}
    _ADAPTERS: Dict[str, Type[BaseAdapter]] = {}
    _COMPRESSORS: Dict[str, Type[BaseCompressor]] = {}

    @classmethod
    def register_router(cls, name: str) -> Callable:
        def decorator(router_cls: Type[BaseRouter]):
            cls._ROUTERS[name.lower()] = router_cls
            return router_cls
        return decorator

    @classmethod
    def register_adapter(cls, name: str) -> Callable:
        def decorator(adapter_cls: Type[BaseAdapter]):
            cls._ADAPTERS[name.lower()] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def register_compressor(cls, name: str) -> Callable:
        def decorator(compressor_cls: Type[BaseCompressor]):
            cls._COMPRESSORS[name.lower()] = compressor_cls
            return compressor_cls
        return decorator

    @classmethod
    def build_router(cls, name: str, **kwargs) -> BaseRouter:
        key = name.lower()
        if key not in cls._ROUTERS:
            raise KeyError(f"Router '{name}' not found in registry. Available: {list(cls._ROUTERS.keys())}")
        return cls._ROUTERS[key](**kwargs)

    @classmethod
    def build_compressor(cls, name: str, **kwargs) -> BaseCompressor:
        key = name.lower()
        if key not in cls._COMPRESSORS:
            raise KeyError(f"Compressor '{name}' not found in registry. Available: {list(cls._COMPRESSORS.keys())}")
        return cls._COMPRESSORS[key](**kwargs)

# Convenient Decorator Shortcuts
register_router = ComponentRegistry.register_router
register_adapter = ComponentRegistry.register_adapter
register_compressor = ComponentRegistry.register_compressor
build_router = ComponentRegistry.build_router
build_compressor = ComponentRegistry.build_compressor
