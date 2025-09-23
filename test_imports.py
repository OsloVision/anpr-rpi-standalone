#!/usr/bin/env python3
"""
Test script to verify all imports and basic functionality work correctly.
"""

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test ANPR module imports
        from anpr import license_plate_reader
        from anpr import loan_db_utils
        from anpr import norwegian_vehicle_api
        print("✅ ANPR modules imported successfully")
    except ImportError as e:
        print(f"❌ ANPR import failed: {e}")
        return False
    
    try:
        # Test utility imports
        from utils import inference_engine
        from utils import video_utils
        print("✅ Utility modules imported successfully")
    except ImportError as e:
        print(f"❌ Utility import failed: {e}")
        return False
    
    try:
        # Test Hailo module imports (should work with fallbacks)
        from hailo import instance_segmentation_edge
        print("✅ Hailo modules imported successfully")
    except ImportError as e:
        print(f"❌ Hailo import failed: {e}")
        return False
    
    try:
        # Test main UI import
        import streamlit_ui
        print("✅ Streamlit UI imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit UI import failed: {e}")
        return False
    
    return True

def test_inference_engines():
    """Test that inference engines can be instantiated."""
    print("\nTesting inference engines...")
    
    try:
        from utils.inference_engine import DemoInferenceEngine, create_inference_engine
        
        # Test demo engine
        demo_engine = DemoInferenceEngine()
        print("✅ Demo inference engine created successfully")
        
        # Test engine factory
        engine = create_inference_engine("dummy_model_path.hef", prefer_hailo=False)
        print(f"✅ Inference engine factory returned: {type(engine).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Inference engine test failed: {e}")
        return False

def test_environment_setup():
    """Test environment configuration."""
    print("\nTesting environment setup...")
    
    try:
        import os
        from dotenv import load_dotenv
        
        # Try to load .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print("✅ .env file loaded successfully")
        else:
            print("ℹ️  No .env file found (this is optional)")
        
        return True
    except Exception as e:
        print(f"❌ Environment setup test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running ANPR Standalone Package Tests\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_inference_engines()
    all_passed &= test_environment_setup()
    
    # Summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! The ANPR standalone package is working correctly.")
        print("\n📝 To run the application:")
        print("   streamlit run streamlit_ui.py")
        print("\n🔧 To configure API keys:")
        print("   cp .env.example .env")
        print("   # Edit .env with your API keys")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)