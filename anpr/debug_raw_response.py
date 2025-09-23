#!/usr/bin/env python3
"""
Raw response debugging script for Norwegian Vehicle API
"""

import os
import json
import sys
from norwegian_vehicle_api import NorwegianVehicleAPI, VehicleAPIError


def debug_raw_response():
    """Debug the raw API response in detail"""

    # Test with a license plate
    license_plate = input(
        "Enter license plate to test (or press Enter for DR21468): "
    ).strip()
    if not license_plate:
        license_plate = "DR21468"

    print(f"\n=== Debugging Raw Response for {license_plate} ===")

    try:
        with NorwegianVehicleAPI(enable_logging=True) as api:
            # Show API configuration
            print("\n1. API Configuration:")
            config = api.debug_config()
            print(json.dumps(config, indent=2))

            # Make the raw request
            print(f"\n2. Making request for license plate: {license_plate}")
            raw_response = api.get_raw_data(license_plate=license_plate)

            # Print the complete raw response
            print("\n3. Complete Raw API Response:")
            print("=" * 60)
            print(json.dumps(raw_response, indent=2, ensure_ascii=False, default=str))
            print("=" * 60)

            # Analyze the response structure
            print("\n4. Response Analysis:")
            print(f"Response type: {type(raw_response)}")

            if isinstance(raw_response, dict):
                print(f"Top-level keys: {list(raw_response.keys())}")

                if "feilmelding" in raw_response:
                    print(f"Error message: {raw_response['feilmelding']}")

                if "kjoretoydataListe" in raw_response:
                    vehicles = raw_response["kjoretoydataListe"]
                    print(f"Number of vehicles found: {len(vehicles)}")

                    if vehicles:
                        print("\n5. First Vehicle Structure:")
                        first_vehicle = vehicles[0]
                        print(
                            f"Vehicle keys: {list(first_vehicle.keys()) if isinstance(first_vehicle, dict) else 'Not a dict'}"
                        )

                        # Show each section
                        for key, value in first_vehicle.items():
                            if isinstance(value, dict):
                                print(f"  {key}: {list(value.keys())}")
                            elif isinstance(value, list):
                                print(f"  {key}: List with {len(value)} items")
                            else:
                                print(f"  {key}: {type(value).__name__} = {value}")

            # Try parsing with the API
            print("\n6. Parsed Vehicle Info:")
            try:
                vehicle_info = api.lookup_by_license_plate(license_plate)
                print("Parsing successful!")

                # Show all non-None fields
                for field_name in vehicle_info.__dataclass_fields__:
                    value = getattr(vehicle_info, field_name)
                    if value is not None:
                        print(f"  {field_name}: {value}")

            except Exception as parse_error:
                print(f"Parsing failed: {parse_error}")

    except VehicleAPIError as e:
        print(f"\nAPI Error: {e.message}")
        print(f"Status Code: {e.status_code}")
        print(f"Details: {e.details}")

    except Exception as e:
        print(f"\nUnexpected Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Norwegian Vehicle API - Raw Response Debugger")
    print("=" * 50)

    # Check API key
    api_key = os.getenv("VEHICLE_API_KEY")
    if not api_key:
        print("⚠️  No VEHICLE_API_KEY environment variable found!")
        print("Please set it before running this script.")
        sys.exit(1)

    print(f"Using API key: {api_key[:8]}...{api_key[-4:]}")

    debug_raw_response()
