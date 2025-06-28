from typing import Dict
import torch

from ...torchkde import KernelDensity

# type aliases
KDECache = Dict[str, Dict[torch.device, KernelDensity]]

class KDEManager:
    """Singleton manager for KDE models to prevent redundant loading"""
    
    _instance = None
    _kde_cache: KDECache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KDEManager, cls).__new__(cls)
        return cls._instance
    
    def get_kde(self, kde_file: str, device: torch.device) -> KernelDensity:
        """Get or load a KDE model from cache using the provided file path and device"""
        if kde_file not in self._kde_cache:
            self._kde_cache[kde_file] = {}
        
        if device not in self._kde_cache[kde_file]:
            self._kde_cache[kde_file][device] = torch.load(kde_file, map_location=device)
        
        return self._kde_cache[kde_file][device]
    
    def clear_cache(self) -> None:
        """Clear the KDE model cache to free memory"""
        self._kde_cache.clear()