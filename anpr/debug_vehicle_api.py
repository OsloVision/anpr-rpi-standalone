#!/usr/bin/env python3
"""
Debug script for Norwegian Vehicle API
"""

import os
import json
from norwegian_vehicle_api import NorwegianVehicleAPI, VehicleAPIError

def debug_api_setup():
    """Debug the API configuration"""
    print("=== API Configuration Debug ===")
    
    # Check environment variables
    api_key = os.getenv('VEHICLE_API_KEY')
    print(f"API Key from env: {'✓ Set' if api_key else '✗ Not set'}")
    if api_key:
        print(f"API Key preview: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")
    
    # Test API initialization
    try:
        with NorwegianVehicleAPI(enable_logging=True) as api:
            config = api.debug_config()
            print("\nAPI Client Configuration:")
            print(json.dumps(config, indent=2))
            
            # Test basic connection
            print(f"\nConnection test: {'✓ Success' if api.test_connection() else '✗ Failed'}")
            
    except Exception as e:
        print(f"Error initializing API: {e}")

def test_simple_request():
    """Test a simple API request with detailed error info"""
    print("\n=== Simple Request Test ===")
    
    try:
        with NorwegianVehicleAPI(enable_logging=True) as api:
            # Try looking up a simple license plate
            license_plate = "AB12345"
            print(f"Testing lookup for: {license_plate}")
            
            
                vehicle_info = api.lookup_by_license_plate(license_plate)
                print("✓ Request successful!")
                print(f"Result: {vehicle_info}")
                
            except VehicleAPIError as e:
                print(f"✗ API Error: {e.message}")
                print(f"  Status Code: {e.status_code}")
                print(f"  Details: {e.details}")
                
                # Additional debugging for 403 errors
                if e.status_code == 403:
                    print("\n--- 403 Troubleshooting ---")
                    print("This usually means:")
                    print("1. Invalid API key")
                    print("2. API key doesn't have required permissions")
                    print("3. Service requires additional authentication")
                    print("4. IP address not whitelisted (if applicable)")
                    print("5. Account not properly activated")
                    
    except Exception as e:
        print(f"Unexpected error: {e}")

def check_api_documentation():
    """Check API endpoint directly"""
    print("\n=== Direct API Check ===")
    
    import requests
    
    api_key = os.getenv('VEHICLE_API_KEY')
    base_url = "https://akfell-datautlevering.atlas.vegvesen.no"
    
    # Test different endpoints
    endpoints = [
        "/",
        "/swagger-ui/index.html",
        "/v3/api-docs",
        "/enkeltoppslag/kjoretoydata"
    ]
    
    for endpoint in endpoints:
        url = base_url + endpoint
        print(f"\nTesting: {url}")
        
        headers = {'User-Agent': 'Debug-Script/1.0'}
        if api_key and endpoint == "/enkeltoppslag/kjoretoydata":
            headers['SVV-Authorization'] = f'Apikey {api_key}'
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"  Status: {response.status_code}")
            print(f"  Content-Type: {response.headers.get('content-type', 'N/A')}")
            
            if response.status_code == 403:
                print(f"  Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"  Error: {e}")

def main():
    print("Norwegian Vehicle API Debug Tool")
    print("================================")
    
    debug_api_setup()
    test_simple_request()
    check_api_documentation()
    
    print("\n=== Recommendations ===")
    print("1. Verify your API key is correct")
    print("2. Check if you need to register your IP address")
    print("3. Ensure your account has the required permissions")
    print("4. Contact Statens Vegvesen if the issue persists")
    print("5. Check their documentation for any recent changes")

if __name__ == "__main__":
    main()