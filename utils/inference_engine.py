#!/usr/bin/env python3
"""Base inference engine interface and implementations."""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any


class BaseInferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load a model from the given path."""
        pass
    
    @abstractmethod
    def run_inference(self, frame: np.ndarray, timeout_ms: int = 1000) -> Optional[Any]:
        """Run inference on a frame."""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get the input shape (height, width, channels)."""
        pass
    
    def run_sync(self, frames: List[np.ndarray], timeout_ms: int = 1000) -> List[Any]:
        """Run synchronous inference on multiple frames."""
        results = []
        for frame in frames:
            result = self.run_inference(frame, timeout_ms)
            results.append(result)
        return results


class HailoInferenceEngine(BaseInferenceEngine):
    """Hailo AI accelerator inference engine."""
    
    def __init__(self):
        self.hailo_inference = None
        self._input_shape = None
    
    def load_model(self, model_path: str) -> bool:
        """Load Hailo model."""
        try:
            from hailo.hailo_inference import HailoInfer
            self.hailo_inference = HailoInfer(model_path)
            self._input_shape = self.hailo_inference.get_input_shape()
            return True
        except Exception as e:
            print(f"Failed to load Hailo model: {e}")
            return False
    
    def run_inference(self, frame: np.ndarray, timeout_ms: int = 1000) -> Optional[Any]:
        """Run Hailo inference."""
        if self.hailo_inference is None:
            return None
        
        try:
            return self.hailo_inference.run_sync([frame], timeout_ms)[0]
        except Exception as e:
            print(f"Hailo inference error: {e}")
            return None
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get Hailo input shape."""
        return self._input_shape if self._input_shape else (640, 640, 3)


class DemoInferenceEngine(BaseInferenceEngine):
    """Demo inference engine for testing without real hardware."""
    
    def __init__(self):
        self._input_shape = (640, 640, 3)
        self.model_loaded = False
    
    def load_model(self, model_path: str) -> bool:
        """Simulate model loading."""
        print(f"Demo: Loading model from {model_path}")
        self.model_loaded = True
        return True
    
    def run_inference(self, frame: np.ndarray, timeout_ms: int = 1000) -> Optional[Any]:
        """Generate mock inference results."""
        if not self.model_loaded:
            return None
        
        height, width = frame.shape[:2]
        
        # Generate random detections for demo
        num_detections = np.random.randint(0, 5)
        detections = []
        
        for _ in range(num_detections):
            # Random bounding box
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = x1 + np.random.randint(50, width // 4)
            y2 = y1 + np.random.randint(30, height // 4)
            
            # Mock detection with bbox and confidence
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': np.random.uniform(0.5, 0.95),
                'class_id': 0,  # Vehicle class
                'mask': np.random.rand(y2-y1, x2-x1) > 0.5  # Random mask
            }
            detections.append(detection)
        
        return detections
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get demo input shape."""
        return self._input_shape


def create_inference_engine(model_path: str, prefer_hailo: bool = True) -> BaseInferenceEngine:
    """Create an inference engine based on availability and preference."""
    
    if prefer_hailo:
        try:
            from hailo import HAILO_AVAILABLE
            if HAILO_AVAILABLE:
                engine = HailoInferenceEngine()
                if engine.load_model(model_path):
                    print("✅ Using Hailo inference engine")
                    return engine
        except ImportError:
            pass
    
    # Fallback to demo engine
    print("⚠️ Using demo inference engine (no real inference)")
    engine = DemoInferenceEngine()
    engine.load_model(model_path)
    return engine