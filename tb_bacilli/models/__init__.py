"""
models/ — Model architectures for TB Bacilli Segmentation.

Primary: CaVMamba (cavmamba.py)
Fallback: Switch-UMamba (switch_umamba.py)
"""

from .model_factory import get_model

__all__ = ["get_model"]
