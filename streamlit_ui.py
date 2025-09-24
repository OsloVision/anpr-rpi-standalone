#!/usr/bin/env python3
import streamlit as st
import cv2
import numpy as np
import os
import sys
import subprocess
import signal
import time
import shutil
import glob
from pathlib import Path
from PIL import Image
import tempfile
from functools import partial
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Shared state file for cross-session communication
SHARED_STATE_FILE = os.path.join(tempfile.gettempdir(), 'anpr_shared_state.json')

def get_shared_recording_state():
    """Get the current recording state shared across all sessions"""
    try:
        if os.path.exists(SHARED_STATE_FILE):
            with open(SHARED_STATE_FILE, 'r') as f:
                data = json.load(f)
                return data.get('recording', False), data.get('timestamp', 0), data.get('ffmpeg_pid', None)
        return False, 0, None
    except (json.JSONDecodeError, FileNotFoundError, PermissionError, IOError):
        return False, 0, None

def set_shared_recording_state(recording, ffmpeg_pid=None):
    """Set the recording state shared across all sessions"""
    try:
        data = {
            'recording': recording, 
            'timestamp': time.time(),
            'ffmpeg_pid': ffmpeg_pid
        }
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(SHARED_STATE_FILE), exist_ok=True)
        
        # Write atomically by writing to temp file then moving
        temp_file = SHARED_STATE_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        os.rename(temp_file, SHARED_STATE_FILE)
        
    except (PermissionError, IOError, OSError):
        pass  # Silently fail if we can't write to temp directory

def stop_recording_by_pid(pid):
    """Stop recording process by PID (works across sessions)"""
    if pid is None:
        return False
    
    try:
        # Try to terminate the process gracefully
        os.kill(pid, signal.SIGINT)
        
        # Wait a bit for graceful shutdown
        time.sleep(2)
        
        # Check if process is still running
        try:
            os.kill(pid, 0)  # This doesn't kill, just checks if process exists
            # Process still exists, force kill
            os.kill(pid, signal.SIGKILL)
        except OSError:
            # Process is already dead
            pass
            
        return True
    except (OSError, ProcessLookupError):
        # Process doesn't exist or we don't have permission
        return False

def cleanup_shared_state():
    """Clean up shared state file"""
    try:
        if os.path.exists(SHARED_STATE_FILE):
            os.remove(SHARED_STATE_FILE)
    except PermissionError:
        pass

# Import from our standalone package structure
try:
    from hailo.hailo_inference import HailoInfer
    from hailo.toolbox import load_json_file, get_labels, default_preprocess
    from hailo import HAILO_AVAILABLE
except ImportError:
    HailoInfer = None
    HAILO_AVAILABLE = False
    # Fallback functions for when Hailo is not available
    def load_json_file(path):
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_labels(config, arch):
        return ["vehicle", "license_plate"]
    
    def default_preprocess(image, width, height):
        return cv2.resize(image, (width, height))

from inference.postprocessing import inference_result_handler, decode_and_postprocess

# Import ANPR functionality
from anpr.license_plate_reader import read_license_plate, check_both_norwegian_services, upsert_loan_status
from anpr.loan_db_utils import LoanStatusDB

# Import utility functions
from utils.inference_engine import create_inference_engine
from utils.video_utils import (
    process_video_file, 
    setup_camera_recording, 
    start_ffmpeg_recording, 
    stop_ffmpeg_recording,
    save_detection_crops
)

# Translations dictionary
translations = {
    "en": {
        "page_title": "Car Loan Detector",
        "main_title": "🚗 Car Loan Detector",
        "subtitle": "Plug in a USB camera and drive around in traffic to get lots of videos to process. Then you can run inference on the videos, and then postprocessing which includes OCR with OpenAI and lookups to Norwegian vehicle databases.",
        "language": "Language",
        "step1_capture": "1️⃣ Capture",
        "step2_infer": "2️⃣ Infer", 
        "step3_postprocess": "3️⃣ Postprocess",
        "recording_in_progress": "🔴 Recording in progress...",
        "click_stop": "Click 'Stop Recording' to finish capture",
        "ready_to_record": "📹 Ready to start recording",
        "start_recording": "📸 Start Recording",
        "stop_recording": "⏹️ Stop Recording",
        "setting_camera": "Setting camera focus controls...",
        "starting_video": "Starting video recording...",
        "recording_started": "✅ Recording started successfully!",
        "camera_error": "❌ Camera control error:",
        "not_found_error": "❌ v4l2-ctl or ffmpeg not found. Please install v4l-utils and ffmpeg.",
        "start_error": "❌ Error starting recording:",
        "recording_stopped": "✅ Recording stopped!",
        "video_saved": "📁 Video files saved to ~/video-capture/output_*.mp4",
        "stop_error": "❌ Error stopping recording:",
        "recent_captures": "Recent Captures",
        "no_videos": "No video files found",
        "output_not_found": "Output directory not found",
        "list_error": "Error listing files:",
        "run_inference": "🧠 Run Inference",
        "capture_first": "❌ Please capture an image first!",
        "processing_videos": "🎬 Processing video files...",
        "no_videos_to_process": "ℹ️ No video files found to process",
        "processing_video": "Processing video:",
        "video_processed": "✅ Video processed successfully:",
        "crops_saved": "📸 Saved crops:",
        "moved_to_processed": "📁 Moved to processed directory:",
        "processing_error": "❌ Error processing video:",
        "demo_inference": "ℹ️ This is a demo UI. Connect your actual inference code here.",
        "inference_completed": "✅ Inference completed!",
        "inference_status": "**Inference Status:** ✅ Ready for postprocessing",
        "postprocess_results": "🎨 Postprocess Results",
        "run_postprocessing": "🎨 Run ANPR Postprocessing",
        "processing_crops": "🔍 Processing crops with ANPR...",
        "crops_processed": "📊 Crops processed:",
        "plates_found": "🔢 License plates found:",
        "registry_matches": "✅ Registry matches:",
        "database_records": "💾 Database records updated:",
        "crops_moved": "📁 Crops moved to processed directory:",
        "anpr_completed": "✅ ANPR postprocessing completed!",
        "no_crops_found": "ℹ️ No crop images found. Please run inference first.",
        "openai_key_missing": "❌ OpenAI API key required for ANPR. Please set OPENAI_API_KEY environment variable.",
        "run_inference_first": "❌ Please run inference first!",
        "no_image": "❌ No captured image available!",
        "demo_postprocess": "ℹ️ This is a demo UI. Connect your actual postprocessing code here.",
        "postprocess_completed": "✅ Postprocessing completed!",
        "results": "📊 Results",
        "original_image": "Original Image",
        "segmentation_result": "Instance Segmentation Result",
        "download_result": "💾 Download Result",
        "download_processed": "⬇️ Download Processed Image",
        "clear_results": "🗑️ Clear All Results",
        "footer": "**Car Loan Detector** | Built with Streamlit 🚀"
    },
    "no": {
        "page_title": "Billån Detektor",
        "main_title": "🚗 Billån Detektor",
        "subtitle": "Last opp et bilde og kjør instanssegmentering med tre-trinns prosess",
        "language": "Språk",
        "step1_capture": "1️⃣ Opptak",
        "step2_infer": "2️⃣ Slutt",
        "step3_postprocess": "3️⃣ Etterbehandle",
        "recording_in_progress": "🔴 Opptak pågår...",
        "click_stop": "Klikk 'Stopp opptak' for å fullføre opptak",
        "ready_to_record": "📹 Klar til å starte opptak",
        "start_recording": "📸 Start opptak",
        "stop_recording": "⏹️ Stopp opptak",
        "setting_camera": "Setter kamerafokuskontroller...",
        "starting_video": "Starter videoopptak...",
        "recording_started": "✅ Opptak startet vellykket!",
        "camera_error": "❌ Kamerakontrollfeil:",
        "not_found_error": "❌ v4l2-ctl eller ffmpeg ikke funnet. Vennligst installer v4l-utils og ffmpeg.",
        "start_error": "❌ Feil ved start av opptak:",
        "recording_stopped": "✅ Opptak stoppet!",
        "video_saved": "📁 Videofiler lagret til ~/video-capture/output_*.mp4",
        "stop_error": "❌ Feil ved stopp av opptak:",
        "recent_captures": "Nylige opptak",
        "no_videos": "Ingen videofiler funnet",
        "output_not_found": "Utdatamappe ikke funnet",
        "list_error": "Feil ved listing av filer:",
        "run_inference": "🧠 Kjør inference",
        "capture_first": "❌ Vennligst ta opp et bilde først!",
        "processing_videos": "🎬 Behandler videofiler...",
        "no_videos_to_process": "ℹ️ Ingen videofiler funnet å behandle",
        "processing_video": "Behandler video:",
        "video_processed": "✅ Video behandlet vellykket:",
        "crops_saved": "📸 Lagret utsnitt:",
        "moved_to_processed": "📁 Flyttet til behandlet mappe:",
        "processing_error": "❌ Feil ved behandling av video:",
        "demo_inference": "ℹ️ Dette er en demo-UI. Koble til din faktiske inferenceskode her.",
        "inference_completed": "✅ inference fullført!",
        "inference_status": "**inferencesstatus:** ✅ Klar for etterbehandling",
        "postprocess_results": "🎨 Etterbehandle resultater",
        "run_postprocessing": "🎨 Kjør ANPR etterbehandling",
        "processing_crops": "🔍 Behandler beskjæringer med ANPR...",
        "crops_processed": "📊 Beskjæringer behandlet:",
        "plates_found": "🔢 Skiltnumre funnet:",
        "registry_matches": "✅ Registertreff:",
        "database_records": "💾 Databaseposter oppdatert:",
        "crops_moved": "📁 Beskjæringer flyttet til behandlet mappe:",
        "anpr_completed": "✅ ANPR etterbehandling fullført!",
        "no_crops_found": "ℹ️ Ingen beskjæringsbilder funnet. Kjør slutning først.",
        "openai_key_missing": "❌ OpenAI API-nøkkel påkrevd for ANPR. Vennligst sett OPENAI_API_KEY miljøvariabel.",
        "run_inference_first": "❌ Vennligst kjør inference først!",
        "no_image": "❌ Ingen tatt bilde tilgjengelig!",
        "demo_postprocess": "ℹ️ Dette er en demo-UI. Koble til din faktiske etterbehandlingskode her.",
        "postprocess_completed": "✅ Etterbehandling fullført!",
        "results": "📊 Resultater",
        "original_image": "Originalt bilde",
        "segmentation_result": "Instanssegmenteringsresultat",
        "download_result": "💾 Last ned resultat",
        "download_processed": "⬇️ Last ned behandlet bilde",
        "clear_results": "🗑️ Tøm alle resultater",
        "footer": "**Billån Detektor** | Bygget med Streamlit 🚀"
    }
}

def calculate_perceptual_hash(image, hash_size=8):
    """
    Calculate perceptual hash for duplicate detection.
    Similar images will have similar hashes.
    """
    # Resize to hash_size x hash_size
    resized = cv2.resize(image, (hash_size, hash_size))
    
    # Convert to grayscale if needed
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Calculate average pixel value
    avg = resized.mean()
    
    # Create binary hash based on whether pixels are above/below average
    diff = resized > avg
    
    # Convert to hex string
    return ''.join(['1' if pixel else '0' for pixel in diff.flatten()])

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two binary hash strings."""
    if len(hash1) != len(hash2):
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def save_detection_crops(original_frame, inference_result, config_data, arch, output_dir, frame_id):
    """
    Extract detection bounding boxes from inference results and save cropped regions as image files.
    Avoids saving duplicate crops based on image content similarity using perceptual hashing.
    """
    import cv2
    import hashlib
    
    # Process the raw inference results to get detection boxes
    try:
        decoded_detections = decode_and_postprocess(inference_result, config_data, arch)
        
        # Debug: Check postprocessing results
        if isinstance(decoded_detections, dict):
            print(f"Postprocessing returned dict with keys: {list(decoded_detections.keys())}")
            if 'detection_boxes' in decoded_detections:
                print(f"Found {len(decoded_detections['detection_boxes'])} detection boxes")
        else:
            print(f"Postprocessing returned: {type(decoded_detections)}")
        
    except Exception as e:
        print(f"Error in decode_and_postprocess: {e}")
        return 0
    
    # Check if we have valid detections
    if not isinstance(decoded_detections, dict) or 'detection_boxes' not in decoded_detections:
        print("No valid detections found")
        return 0
    
    boxes = decoded_detections['detection_boxes']
    scores = decoded_detections.get('detection_scores', [])
    class_ids = decoded_detections.get('detection_classes', [])
    
    if len(boxes) == 0:
        return 0
    
    # Create crops directory
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    # Load existing image hashes to avoid duplicates
    hash_file = os.path.join(crops_dir, "image_hashes.txt")
    existing_exact_hashes = set()
    existing_perceptual_hashes = []
    
    if os.path.exists(hash_file):
        try:
            with open(hash_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            existing_exact_hashes.add(parts[0])
                            existing_perceptual_hashes.append(parts[1])
        except:
            pass  # Continue if can't read hash file
    
    h, w = original_frame.shape[:2]
    model_h, model_w = 640, 640  # Model input size
    
    # Calculate preprocessing scaling and padding
    scale = min(model_w / w, model_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2
    
    crop_count = 0
    
    for i, box in enumerate(boxes):
        # Get confidence score and class if available
        confidence = scores[i] if i < len(scores) else 1.0
        class_id = int(class_ids[i]) if i < len(class_ids) else 0
        
        # Boxes are normalized to model input size (640x640)
        xmin_norm, ymin_norm, xmax_norm, ymax_norm = box
        
        # Scale to model coordinates (640x640)
        xmin_model = xmin_norm * model_w
        ymin_model = ymin_norm * model_h
        xmax_model = xmax_norm * model_w
        ymax_model = ymax_norm * model_h
        
        # Remove padding to get scaled image coordinates
        xmin_scaled = xmin_model - x_offset
        ymin_scaled = ymin_model - y_offset
        xmax_scaled = xmax_model - x_offset
        ymax_scaled = ymax_model - y_offset
        
        # Scale back to original image coordinates
        xmin = int(xmin_scaled / scale)
        ymin = int(ymin_scaled / scale)
        xmax = int(xmax_scaled / scale)
        ymax = int(ymax_scaled / scale)
        
        # Clamp coordinates to image bounds
        xmin = max(0, min(xmin, w-1))
        xmax = max(0, min(xmax, w-1))
        ymin = max(0, min(ymin, h-1))
        ymax = max(0, min(ymax, h-1))
        
        # Skip if crop is too small
        if (xmax - xmin) < 10 or (ymax - ymin) < 10:
            continue
            
        # Extract crop
        crop = original_frame[ymin:ymax, xmin:xmax]
        
        if crop.size > 0:
            # Calculate both exact and perceptual hashes for duplicate detection
            crop_resized = cv2.resize(crop, (64, 64))
            crop_gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            
            # Exact hash (MD5 of normalized image data)
            exact_hash = hashlib.md5(crop_gray.tobytes()).hexdigest()
            
            # Perceptual hash for similarity detection
            perceptual_hash = calculate_perceptual_hash(crop_gray)
            
            # Check for exact duplicates first
            if exact_hash in existing_exact_hashes:
                continue  # Skip exact duplicate
            
            # Check for similar images using perceptual hash
            is_similar = False
            similarity_threshold = 8  # Allow up to 8 bit differences (out of 64 bits)
            
            for existing_phash in existing_perceptual_hashes:
                if hamming_distance(perceptual_hash, existing_phash) <= similarity_threshold:
                    is_similar = True
                    break
            
            if is_similar:
                continue  # Skip similar image
            
            # Create filename with frame_id, crop_id, class_id, and confidence
            crop_filename = f"frame_{frame_id:06d}_crop_{i:03d}_class_{class_id}_conf_{confidence:.3f}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            
            # Save crop
            cv2.imwrite(crop_path, crop)
            
            # Add hashes to existing sets and save to file
            existing_exact_hashes.add(exact_hash)
            existing_perceptual_hashes.append(perceptual_hash)
            
            try:
                with open(hash_file, 'a') as f:
                    f.write(f"{exact_hash}|{perceptual_hash}\n")
            except:
                pass  # Continue if can't write hash file
            
            crop_count += 1
    
    # Periodically clean up hash file to prevent it from growing too large
    if crop_count > 0:
        cleanup_hash_file(hash_file)
    
    return crop_count

def cleanup_hash_file(hash_file, max_entries=10000):
    """
    Clean up hash file if it gets too large to maintain performance.
    Keeps only the most recent entries.
    """
    try:
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > max_entries:
                # Keep only the last max_entries
                with open(hash_file, 'w') as f:
                    f.writelines(lines[-max_entries:])
    except:
        pass  # Ignore cleanup errors

def process_video_files():
    """
    Process all MP4 files in ~/video-capture directory and move them to processed folder after processing.
    """
    video_dir = os.path.expanduser("~/video-capture")
    processed_dir = os.path.join(video_dir, "processed")
    crops_dir = os.path.join(video_dir, "crops")
    
    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    
    # Find all MP4 files in video directory (not in processed subdirectory)
    mp4_files = [f for f in os.listdir(video_dir) 
                 if f.endswith('.mp4') and os.path.isfile(os.path.join(video_dir, f))]
    
    if not mp4_files:
        return [], 0
    
    processed_files = []
    total_crops = 0
    
    # Load configuration
    try:
        config_data = load_json_file("config.json")
    except:
        # Default config for demo mode
        config_data = {
            "fast": {
                "score_threshold": 0.5,
                "nms_threshold": 0.4
            },
            "v5": {
                "score_threshold": 0.5,
                "nms_threshold": 0.4
            },
            "v8": {
                "score_threshold": 0.5,
                "nms_threshold": 0.4
            }
        }
    
    # Check if we have a loaded model
    hailo_inference = st.session_state.hailo_inference
    architecture = st.session_state.architecture
    use_real_inference = hailo_inference is not None
    
    # Debug model status
    print(f"Processing videos - Model loaded: {use_real_inference}, Architecture: {architecture}")
    if hasattr(st.session_state, 'model_path'):
        print(f"Model path: {st.session_state.model_path}")
    
    if use_real_inference:
        try:
            height, width, _ = hailo_inference.get_input_shape()
            print(f"Model input shape: {width}x{height}")
        except Exception as e:
            print(f"Error getting model shape: {e}")
            use_real_inference = False
    
    # Set OpenCV environment variables for better video handling
    os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'
    
    # Process each video file
    for video_file in mp4_files:
        video_path = os.path.join(video_dir, video_file)
        
        try:
            # Open video with specific backend for better compatibility
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"Failed to open video: {video_file}")
                continue
                
            # Get video properties for debugging
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Processing video: {video_file}")
            print(f"  Resolution: {width_orig}x{height_orig}")
            print(f"  FPS: {fps}")
            print(f"  Total frames: {frame_count_total}")
            
            # Skip if video properties are invalid
            if fps <= 0 or frame_count_total <= 0:
                print(f"  Invalid video properties, skipping")
                cap.release()
                continue
            
            frame_count = 0
            video_crops = 0
            
            # Calculate frame skip based on actual FPS (aim for 1 frame per second)
            frame_skip = max(1, int(fps)) if fps > 0 else 30
            print(f"  Processing every {frame_skip} frames")
            
            # Process frames with better error handling
            consecutive_failures = 0
            max_consecutive_failures = 100
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > max_consecutive_failures:
                        print(f"  Too many consecutive read failures, stopping")
                        break
                    continue
                else:
                    consecutive_failures = 0
                
                if frame_count % frame_skip == 0:  # Process frames at ~1 per second
                    if use_real_inference:
                        try:
                            # Real inference with loaded model
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            preprocessed_frame = default_preprocess(rgb_frame, width, height)
                            
                            # Run inference
                            result = hailo_inference.run_sync([preprocessed_frame])
                            
                            # Debug: Check if we got results
                            if result and len(result) > 0:
                                # Save detection crops
                                crops = save_detection_crops(
                                    frame, result[0], config_data, architecture, video_dir, frame_count
                                )
                                video_crops += crops
                                
                                # Debug info (only for first few frames)
                                if frame_count < 90:  # First 3 processed frames
                                    print(f"Frame {frame_count}: Found {crops} crops, result shape: {result[0].shape if hasattr(result[0], 'shape') else 'unknown'}")
                            else:
                                print(f"Frame {frame_count}: No inference results returned")
                            
                        except Exception as e:
                            st.error(f"Inference error on frame {frame_count}: {str(e)}")
                            print(f"Full inference error on frame {frame_count}: {str(e)}")
                    else:
                        # Demo mode - simulate detection crops
                        demo_crops = min(np.random.randint(0, 5), 3)  # 0-3 random crops per frame
                        video_crops += demo_crops
                
                frame_count += 1
            
            cap.release()
            total_crops += video_crops
            
            # Move processed video to processed directory
            processed_path = os.path.join(processed_dir, video_file)
            shutil.move(video_path, processed_path)
            
            processed_files.append({
                'filename': video_file,
                'crops': video_crops,
                'frames': frame_count,
                'inference_mode': 'real' if use_real_inference else 'demo'
            })
            
        except Exception as e:
            st.error(f"Error processing {video_file}: {str(e)}")
            continue
    
    return processed_files, total_crops


def process_crops_with_anpr():
    """Process all crop images with ANPR and Norwegian registry lookup"""
    try:
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        
        output_dir = os.path.expanduser("~/video-capture")
        crops_dir = os.path.join(output_dir, "crops")
        processed_crops_dir = os.path.join(output_dir, "processed-crops")
        
        # Create processed-crops directory if it doesn't exist
        os.makedirs(processed_crops_dir, exist_ok=True)
        
        if not os.path.exists(crops_dir):
            return False, "No crops directory found. Please run inference first."
        
        # Find all crop images
        crop_files = glob.glob(os.path.join(crops_dir, "*.jpg"))
        
        if not crop_files:
            return False, "No crop images found. Please run inference first."
        
        # Initialize database
        db_path = os.path.join(output_dir, "anpr_results.db")
        
        # Process each crop
        total_crops = len(crop_files)
        plates_found = 0
        registry_matches = 0
        database_records = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, crop_file in enumerate(crop_files):
            try:
                crop_filename = os.path.basename(crop_file)
                status_text.text(f"Processing {crop_filename} ({i+1}/{total_crops})")
                
                # Read license plate using OpenAI
                license_plate_text = read_license_plate(crop_file, api_key)
                
                if license_plate_text and license_plate_text.strip():
                    plates_found += 1
                    st.write(f"📋 Found plate: {license_plate_text} in {crop_filename}")
                    
                    # Check both Norwegian registry services
                    registry_status, registry_info = check_both_norwegian_services(license_plate_text)
                    
                    if registry_status in ["yes", "partial"]:
                        registry_matches += 1
                        st.write(f"✅ Registry match: {license_plate_text} -> {registry_status}")
                        
                        # Only save to database if we have actual vehicle data
                        success = upsert_loan_status(license_plate_text, registry_status, registry_info, db_path)
                        if success:
                            database_records += 1
                    else:
                        st.write(f"❌ No registry match: {license_plate_text} -> {registry_status}")
                        # Don't add to database if no vehicle data is available
                
                # Move processed crop to processed-crops directory
                dest_path = os.path.join(processed_crops_dir, crop_filename)
                shutil.move(crop_file, dest_path)
                
                # Update progress
                progress_bar.progress((i + 1) / total_crops)
                
            except Exception as e:
                st.error(f"❌ Error processing {crop_filename}: {str(e)}")
                continue
        
        status_text.text("ANPR processing completed!")
        
        # Summary statistics
        summary = {
            "total_crops": total_crops,
            "plates_found": plates_found,
            "registry_matches": registry_matches,
            "database_records": database_records,
            "processed_crops_moved": total_crops
        }
        
        return True, summary
        
    except Exception as e:
        return False, f"Error in ANPR processing: {str(e)}"


def check_api_keys():
    """Check if required API keys are configured"""
    openai_key = os.getenv("OPENAI_API_KEY")
    vehicle_key = os.getenv("VEHICLE_API_KEY")
    
    return {
        "openai_configured": openai_key is not None and openai_key != "your-openai-api-key-here",
        "vehicle_configured": vehicle_key is not None and vehicle_key != "your-norwegian-vehicle-api-key-here",
        "openai_key": openai_key,
        "vehicle_key": vehicle_key
    }

# Page configuration
st.set_page_config(
    page_title="Car Loan Detector",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'inference_result' not in st.session_state:
    st.session_state.inference_result = None
if 'processed_result' not in st.session_state:
    st.session_state.processed_result = None
if 'ffmpeg_process' not in st.session_state:
    st.session_state.ffmpeg_process = None
if 'recording' not in st.session_state:
    # Check shared state to sync with other sessions
    shared_recording, _, _ = get_shared_recording_state()
    st.session_state.recording = shared_recording
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'model_path' not in st.session_state:
    st.session_state.model_path = '~/anprv2.hef'
if 'architecture' not in st.session_state:
    st.session_state.architecture = 'fast'
if 'hailo_inference' not in st.session_state:
    st.session_state.hailo_inference = None
if 'anpr_result' not in st.session_state:
    st.session_state.anpr_result = None

# Auto-load model with defaults if not already loaded
if st.session_state.hailo_inference is None and HAILO_AVAILABLE:
    model_path_expanded = os.path.expanduser(st.session_state.model_path)
    print(f"Attempting to load model from: {model_path_expanded}")
    print(f"Model file exists: {os.path.exists(model_path_expanded)}")
    
    if os.path.exists(model_path_expanded):
        try:
            print("Loading Hailo model...")
            st.session_state.hailo_inference = HailoInfer(
                model_path_expanded,
                batch_size=1,
                output_type="FLOAT32"
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Continue without model
            pass
    else:
        print(f"❌ Model file not found: {model_path_expanded}")
else:
    if not HAILO_AVAILABLE:
        print("❌ Hailo not available")
    if st.session_state.hailo_inference is not None:
        print("✅ Model already loaded")

# Language selector
language = st.selectbox(
    "Language / Språk",
    options=['en', 'no'],
    format_func=lambda x: 'English' if x == 'en' else 'Norsk (Bokmål)',
    index=0 if st.session_state.language == 'en' else 1,
    key='language_selector'
)
st.session_state.language = language

# Get current translations
t = translations[st.session_state.language]

# Update page title and content with translations
st.title(t["main_title"])
st.markdown(t["subtitle"])

# Configuration
arch_type = "fast"  # Default architecture type

# Add refresh button to sync state across sessions\ncol_refresh, col_debug, col_empty = st.columns([1, 1, 3])\nwith col_refresh:\n    if st.button(\"🔄 Refresh\", help=\"Sync with other browser tabs\"):\n        st.rerun()\n\nwith col_debug:\n    # Debug info\n    shared_recording, shared_timestamp = get_shared_recording_state()\n    if st.button(\"🐛 Debug\", help=\"Show debug info\"):\n        st.write(f\"Local recording: {st.session_state.recording}\")\n        st.write(f\"Shared recording: {shared_recording}\")\n        st.write(f\"Shared timestamp: {shared_timestamp}\")\n        st.write(f\"File exists: {os.path.exists(SHARED_STATE_FILE)}\")\n        if os.path.exists(SHARED_STATE_FILE):\n            st.write(f\"File content: {open(SHARED_STATE_FILE).read()}\")

# Main UI with three columns for the three steps
col1, col2, col3 = st.columns(3)

# Step 1: Capture
with col1:
    st.header(t["step1_capture"])
    
    # Sync with shared state on each page load
    shared_recording, shared_timestamp, shared_pid = get_shared_recording_state()
    
    # Update local state if shared state has changed
    if shared_recording != st.session_state.recording:
        st.session_state.recording = shared_recording
        if not shared_recording:
            # Recording was stopped in another session
            st.session_state.ffmpeg_process = None
    
    # Validate that recording process is actually still running (only if we have a process object)
    if st.session_state.recording and hasattr(st.session_state, 'ffmpeg_process') and st.session_state.ffmpeg_process:
        try:
            # Check if process is still alive
            if st.session_state.ffmpeg_process.poll() is not None:
                # Process has terminated
                st.session_state.recording = False
                set_shared_recording_state(False)
                st.session_state.ffmpeg_process = None
        except (AttributeError, OSError):
            # Process object is invalid or no longer exists
            st.session_state.recording = False
            set_shared_recording_state(False)
            st.session_state.ffmpeg_process = None
    
    # Recording status
    if st.session_state.recording:
        st.success(t["recording_in_progress"])
        st.info(t["click_stop"])
    else:
        st.info(t["ready_to_record"])
    
    # Camera capture controls
    col_start, col_stop = st.columns(2)
    
    with col_start:
        if st.button(t["start_recording"], use_container_width=True, disabled=st.session_state.recording):
            try:
                # Set camera focus controls
                st.info(t["setting_camera"])
                subprocess.run([
                    "/usr/bin/v4l2-ctl", 
                    "--set-ctrl=focus_automatic_continuous=0"
                ], check=True)
                
                subprocess.run([
                    "/usr/bin/v4l2-ctl", 
                    "--set-ctrl=focus_absolute=1"
                ], check=True)
                
                # Start ffmpeg recording
                st.info(t["starting_video"])
                
                # Create output directory if it doesn't exist
                output_dir = os.path.expanduser("~/video-capture")
                os.makedirs(output_dir, exist_ok=True)
                
                ffmpeg_cmd = [
                    "/usr/bin/ffmpeg",
                    "-f", "v4l2",
                    "-input_format", "h264", 
                    "-framerate", "30",
                    "-video_size", "1920x1080",
                    "-i", "/dev/video0",
                    "-c:v", "copy",
                    "-f", "segment",
                    "-segment_time", "120",
                    "-segment_format", "mp4",
                    "-reset_timestamps", "1",
                    "-strftime", "1",
                    f"{output_dir}/output_%Y%m%d_%H%M%S.mp4"
                ]
                
                # Start the process
                st.session_state.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                st.session_state.recording = True
                ffmpeg_pid = st.session_state.ffmpeg_process.pid if st.session_state.ffmpeg_process else None
                set_shared_recording_state(True, ffmpeg_pid)
                st.success(t["recording_started"])
                st.rerun()
                
            except subprocess.CalledProcessError as e:
                st.error(f"{t['camera_error']} {str(e)}")
            except FileNotFoundError:
                st.error(t["not_found_error"])
            except Exception as e:
                st.error(f"{t['start_error']} {str(e)}")
    
    with col_stop:
        if st.button(t["stop_recording"], use_container_width=True, disabled=not st.session_state.recording):
            try:
                # Get shared state to find the PID
                shared_recording, shared_timestamp, shared_pid = get_shared_recording_state()
                
                stopped = False
                
                # Try to stop using local process object first
                if hasattr(st.session_state, 'ffmpeg_process') and st.session_state.ffmpeg_process:
                    try:
                        # Send SIGINT to ffmpeg for clean shutdown
                        st.session_state.ffmpeg_process.send_signal(signal.SIGINT)
                        
                        # Wait for process to finish (with timeout)
                        try:
                            st.session_state.ffmpeg_process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            # Force kill if it doesn't stop gracefully
                            st.session_state.ffmpeg_process.kill()
                            st.session_state.ffmpeg_process.wait()
                        
                        st.session_state.ffmpeg_process = None
                        stopped = True
                    except Exception:
                        pass  # Fall back to PID method
                
                # If local process didn't work, try using PID from shared state
                if not stopped and shared_pid:
                    stopped = stop_recording_by_pid(shared_pid)
                
                if stopped:
                    st.session_state.recording = False
                    set_shared_recording_state(False)
                    st.success(t["recording_stopped"])
                    st.info(t["video_saved"])
                    st.rerun()
                else:
                    st.error("Failed to stop recording - process may have already ended")
                
            except Exception as e:
                st.error(f"{t['stop_error']} {str(e)}")
    
    # Show recent captures
    if not st.session_state.recording:
        st.subheader(t["recent_captures"])
        try:
            # List recent video files
            output_dir = os.path.expanduser("~/video-capture")
            if os.path.exists(output_dir):
                video_files = sorted([
                    f for f in os.listdir(output_dir) 
                    if f.startswith("output_") and f.endswith(".mp4")
                ], reverse=True)[:5]  # Show last 5 files
                
                if video_files:
                    for video_file in video_files:
                        file_path = os.path.join(output_dir, video_file)
                        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                        st.write(f"📹 {video_file} ({file_size:.1f} MB)")
                else:
                    st.write(t["no_videos"])
            else:
                st.write(t["output_not_found"])
                
        except FileNotFoundError:
            st.write(t["output_not_found"])
        except Exception as e:
            st.write(f"{t['list_error']} {str(e)}")
    
    # Display captured/uploaded image
    if st.session_state.captured_image is not None:
        # Convert BGR to RGB for display
        display_image = cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB)
        st.image(display_image, caption="Current Image", use_column_width=True)

# Step 2: Infer
with col2:
    st.header(t["step2_infer"])
    
    if st.button(t["run_inference"], use_container_width=True):
        # Check for video files to process
        video_dir = os.path.expanduser("~/video-capture")
        if not os.path.exists(video_dir):
            st.error(t["output_not_found"])
        else:
            # Find MP4 files to process
            mp4_files = [f for f in os.listdir(video_dir) 
                        if f.endswith('.mp4') and os.path.isfile(os.path.join(video_dir, f))]
            
            if not mp4_files:
                st.info(t["no_videos_to_process"])
            else:
                # Show inference mode
                if st.session_state.hailo_inference is not None:
                    st.info(f"🧠 Using real inference with {st.session_state.architecture} architecture")
                
                with st.spinner(t["processing_videos"]):
                    processed_files, total_crops = process_video_files()
                
                if processed_files:
                    st.success(t["inference_completed"])
                    st.session_state.inference_result = {
                        "processed_files": processed_files,
                        "total_crops": total_crops
                    }
                    
                    # Display processing results
                    st.write(f"**{t['video_processed']}**")
                    for file_info in processed_files:
                        inference_mode = "🧠 Real" if file_info.get('inference_mode') == 'real' else "🎭 Demo"
                        st.write(f"📹 {file_info['filename']} ({inference_mode})")
                        st.write(f"  └─ {t['crops_saved']} {file_info['crops']}")
                        st.write(f"  └─ {t['moved_to_processed']} processed/")
                    
                    st.info(f"**Total:** {len(processed_files)} videos, {total_crops} crops")
                else:
                    st.warning(t["processing_error"])
    
    # Display inference status
    if st.session_state.inference_result is not None:
        if isinstance(st.session_state.inference_result, dict) and "processed_files" in st.session_state.inference_result:
            files_count = len(st.session_state.inference_result["processed_files"])
            crops_count = st.session_state.inference_result["total_crops"]
            st.write(f"**Status:** ✅ Processed {files_count} videos, {crops_count} crops saved")
        else:
            st.write(t["inference_status"])

# Step 3: Postprocess
with col3:
    st.header(t["step3_postprocess"])
    
    # Check API keys status
    api_status = check_api_keys()
    
    # Show API key configuration status
    with st.expander("🔑 API Configuration", expanded=not (api_status["openai_configured"] and api_status["vehicle_configured"])):
        col_api1, col_api2 = st.columns(2)
        
        with col_api1:
            if api_status["openai_configured"]:
                st.success("✅ OpenAI API Key: Configured")
            else:
                st.error("❌ OpenAI API Key: Not configured")
                st.write("Required for license plate text extraction")
        
        with col_api2:
            if api_status["vehicle_configured"]:
                st.success("✅ Vehicle API Key: Configured")
            else:
                st.warning("⚠️ Vehicle API Key: Not configured")
                st.write("Optional - will use web scraping fallback")
        
        if not api_status["openai_configured"]:
            st.info("📝 To configure API keys:")
            st.code("""
# Edit the .env file in this directory:
nano .env

# Add your OpenAI API key:
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Optionally add Norwegian Vehicle API key:
VEHICLE_API_KEY=your-vehicle-api-key-here
            """)
    
    # ANPR Processing button
    can_process = api_status["openai_configured"]
    
    if st.button(t["run_postprocessing"], use_container_width=True, disabled=not can_process):
        if not can_process:
            st.error(t["openai_key_missing"])
        else:
            st.info(t["processing_crops"])
            
            # Process crops with ANPR
            success, result = process_crops_with_anpr()
            
            if success:
                # Display summary statistics
                st.success(t["anpr_completed"])
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric(t["crops_processed"], result["total_crops"])
                    st.metric(t["plates_found"], result["plates_found"])
                    st.metric(t["registry_matches"], result["registry_matches"])
                
                with col_stat2:
                    st.metric(t["database_records"], result["database_records"])
                    st.metric(t["crops_moved"], result["processed_crops_moved"])
                
                # Store result in session state
                st.session_state.anpr_result = result
                
            else:
                if "No crop images found" in str(result):
                    st.warning(t["no_crops_found"])
                else:
                    st.error(f"❌ {result}")
    
    # Display ANPR results if available
    if 'anpr_result' in st.session_state and st.session_state.anpr_result:
        st.subheader("📊 ANPR Results Summary")
        result = st.session_state.anpr_result
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Crops", result["total_crops"])
        with col2:
            st.metric("License Plates Found", result["plates_found"])
        with col3:
            st.metric("Registry Matches", result["registry_matches"])
        
        # Database location
        output_dir = os.path.expanduser("~/video-capture")
        db_path = os.path.join(output_dir, "anpr_results.db")
        st.info(f"💾 Results saved to: {db_path}")
        
        # Show processed crops directory
        processed_crops_dir = os.path.join(output_dir, "processed-crops")
        st.info(f"📁 Processed crops moved to: {processed_crops_dir}")

# Results section
if st.session_state.processed_result is not None:
    st.header(t["results"])
    
    # Side-by-side comparison
    col_orig, col_result = st.columns(2)
    
    with col_orig:
        st.subheader(t["original_image"])
        orig_display = cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB)
        st.image(orig_display, use_column_width=True)
    
    with col_result:
        st.subheader(t["segmentation_result"])
        result_display = cv2.cvtColor(st.session_state.processed_result, cv2.COLOR_BGR2RGB)
        st.image(result_display, use_column_width=True)
    
    # Download processed image
    if st.button(t["download_result"]):
        # Convert to PIL for saving
        result_pil = Image.fromarray(cv2.cvtColor(st.session_state.processed_result, cv2.COLOR_BGR2RGB))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            result_pil.save(tmp_file.name, "JPEG")
            
            # Provide download link
            with open(tmp_file.name, "rb") as file:
                st.download_button(
                    label=t["download_processed"],
                    data=file.read(),
                    file_name="instance_segmentation_result.jpg",
                    mime="image/jpeg"
                )

# Clear results button
if st.button(t["clear_results"]):
    st.session_state.captured_image = None
    st.session_state.inference_result = None
    st.session_state.processed_result = None
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(t["footer"])