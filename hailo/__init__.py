"""Hailo AI Accelerator Integration Module"""

try:
    from .hailo_inference import HailoInfer
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    HailoInfer = None

from .toolbox import load_json_file, get_labels, default_preprocess

# Import the instance segmentation module if available
try:
    from . import instance_segmentation_edge
except ImportError:
    instance_segmentation_edge = None

__all__ = ["HailoInfer", "load_json_file", "get_labels", "default_preprocess", "HAILO_AVAILABLE"]