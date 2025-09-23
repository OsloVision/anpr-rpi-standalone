"""ANPR Standalone - Complete License Plate Recognition System"""

__version__ = "1.0.0"
__author__ = "OsloVision"
__description__ = "Standalone ANPR system with Streamlit UI and Norwegian registry integration"

# Optional imports based on availability
try:
    from hailo import HAILO_AVAILABLE
except ImportError:
    HAILO_AVAILABLE = False