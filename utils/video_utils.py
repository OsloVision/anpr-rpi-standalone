#!/usr/bin/env python3
"""Video processing utilities for ANPR system."""

import cv2
import os
import numpy as np
import glob
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from hailo.toolbox import default_preprocess
from inference.postprocessing import decode_and_postprocess


def save_detection_crops(frame: np.ndarray, detections: Any, config_data: Dict, 
                        architecture: str, output_dir: str, frame_number: int) -> int:
    """
    Save detection crops from a frame.
    
    Args:
        frame: Input video frame
        detections: Inference results
        config_data: Configuration for postprocessing
        architecture: Model architecture (fast, v5, v8)
        output_dir: Base output directory
        frame_number: Current frame number
        
    Returns:
        Number of crops saved
    """
    try:
        # Create crops directory
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        
        if detections is None:
            return 0
        
        # Decode detections using postprocessing
        try:
            decoded_detections = decode_and_postprocess(detections, config_data, architecture)
        except Exception as e:
            print(f"Postprocessing error: {e}")
            return 0
        
        if not decoded_detections:
            return 0
        
        crops_saved = 0
        
        # Process each detection
        for i, detection in enumerate(decoded_detections):
            try:
                # Extract bounding box
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                else:
                    # Try alternative bbox formats
                    x1 = int(detection.get('x1', detection.get('left', 0)))
                    y1 = int(detection.get('y1', detection.get('top', 0)))
                    x2 = int(detection.get('x2', detection.get('right', frame.shape[1])))
                    y2 = int(detection.get('y2', detection.get('bottom', frame.shape[0])))
                
                # Validate bbox
                x1 = max(0, min(x1, frame.shape[1]))
                y1 = max(0, min(y1, frame.shape[0]))
                x2 = max(x1, min(x2, frame.shape[1]))
                y2 = max(y1, min(y2, frame.shape[0]))
                
                if x2 > x1 and y2 > y1:
                    # Extract crop
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        # Generate filename
                        confidence = detection.get('confidence', 0.0)
                        crop_filename = f"crop_frame_{frame_number:06d}_{i:02d}_conf_{confidence:.2f}.jpg"
                        crop_path = os.path.join(crops_dir, crop_filename)
                        
                        # Save crop
                        cv2.imwrite(crop_path, crop)
                        crops_saved += 1
                        
            except Exception as e:
                print(f"Error saving crop {i}: {e}")
                continue
        
        return crops_saved
        
    except Exception as e:
        print(f"Error in save_detection_crops: {e}")
        return 0


def process_video_file(video_path: str, inference_engine: Any, config_data: Dict, 
                      architecture: str, output_dir: str) -> Dict[str, Any]:
    """
    Process a single video file and extract detection crops.
    
    Args:
        video_path: Path to video file
        inference_engine: Inference engine instance
        config_data: Configuration data
        architecture: Model architecture
        output_dir: Output directory for crops
        
    Returns:
        Dictionary with processing results
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video file"}
        
        frame_count = 0
        total_crops = 0
        processed_frames = 0
        
        # Get input shape from inference engine
        height, width, _ = inference_engine.get_input_shape()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 30th frame (approximately 1 per second at 30fps)
            if frame_count % 30 == 0:
                try:
                    # Preprocess frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preprocessed_frame = default_preprocess(rgb_frame, width, height)
                    
                    # Run inference
                    result = inference_engine.run_inference(preprocessed_frame)
                    
                    if result is not None:
                        # Save detection crops
                        crops = save_detection_crops(
                            frame, result, config_data, architecture, output_dir, frame_count
                        )
                        total_crops += crops
                    
                    processed_frames += 1
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        return {
            "success": True,
            "total_frames": frame_count,
            "processed_frames": processed_frames,
            "total_crops": total_crops,
            "video_filename": os.path.basename(video_path)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def setup_camera_recording(device_id: int = 0, output_dir: str = "~/video-capture") -> Dict[str, Any]:
    """
    Setup camera recording using v4l2-ctl and ffmpeg.
    
    Args:
        device_id: Camera device ID
        output_dir: Output directory for recordings
        
    Returns:
        Dictionary with setup status
    """
    try:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if required tools are available
        try:
            subprocess.run(["v4l2-ctl", "--version"], capture_output=True, check=True)
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                "success": False, 
                "error": "v4l2-ctl or ffmpeg not found. Please install v4l-utils and ffmpeg."
            }
        
        # Set camera controls
        try:
            subprocess.run([
                "v4l2-ctl", f"--device=/dev/video{device_id}",
                "--set-ctrl=focus_automatic_continuous=0",
                "--set-ctrl=focus_absolute=0"
            ], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            # Camera controls may not be supported, continue anyway
            pass
        
        return {"success": True, "output_dir": output_dir}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def start_ffmpeg_recording(device_id: int = 0, output_dir: str = "~/video-capture") -> Optional[subprocess.Popen]:
    """
    Start ffmpeg recording process.
    
    Args:
        device_id: Camera device ID
        output_dir: Output directory
        
    Returns:
        FFmpeg process handle or None if failed
    """
    try:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"output_{timestamp}.mp4")
        
        # Start ffmpeg recording
        cmd = [
            "ffmpeg",
            "-f", "v4l2",
            "-framerate", "30",
            "-video_size", "1920x1080",
            "-i", f"/dev/video{device_id}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-y",  # Overwrite output file
            output_file
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return process
        
    except Exception as e:
        print(f"Error starting ffmpeg recording: {e}")
        return None


def stop_ffmpeg_recording(process: subprocess.Popen) -> bool:
    """
    Stop ffmpeg recording process gracefully.
    
    Args:
        process: FFmpeg process handle
        
    Returns:
        True if stopped successfully, False otherwise
    """
    try:
        if process and process.poll() is None:
            # Send SIGTERM to gracefully stop recording
            process.terminate()
            
            # Wait for process to finish (with timeout)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop gracefully
                process.kill()
                process.wait()
            
            return True
        return False
        
    except Exception as e:
        print(f"Error stopping ffmpeg recording: {e}")
        return False