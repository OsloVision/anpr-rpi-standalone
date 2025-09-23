#!/usr/bin/env python3
"""
Example usage of the Norwegian Vehicle API client
"""

import os
import sys
import json
from norwegian_vehicle_api import (
    NorwegianVehicleAPI,
    VehicleAPIError,
    lookup_vehicle_by_plate,
    lookup_vehicle_by_vin,
)

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def example_license_plate_lookup():
    """Example: Look up vehicle by license plate"""
    print("=== License Plate Lookup Example ===")

    # Example license plate (replace with real one for testing)
    license_plate = "AB12345"

    try:
        # Method 1: Using the convenience function
        print(f"Looking up vehicle with license plate: {license_plate}")
        vehicle_info = lookup_vehicle_by_plate(license_plate)

        if vehicle_info.feilmelding:
            print(f"Error: {vehicle_info.feilmelding}")
        else:
            print(f"KUID: {vehicle_info.kuid}")
            print(f"VIN: {vehicle_info.understellsnummer}")
            print(f"License Plate: {vehicle_info.kjennemerke}")
            print(f"First Registered: {vehicle_info.registrert_forstgang}")
            print(f"Inspection Due: {vehicle_info.kontrollfrist}")

    except VehicleAPIError as e:
        print(f"API Error: {e.message}")
        if e.status_code:
            print(f"Status Code: {e.status_code}")
        if e.details:
            print(f"Details: {e.details}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def example_vin_lookup():
    """Example: Look up vehicle by VIN"""
    print("\n=== VIN Lookup Example ===")

    # Example VIN (replace with real one for testing)
    vin = "1HGBH41JXMN109186"

    try:
        # Method 2: Using the API class directly
        with NorwegianVehicleAPI(enable_logging=True) as api:
            print(f"Looking up vehicle with VIN: {vin}")
            vehicle_info = api.lookup_by_vin(vin)

            if vehicle_info.feilmelding:
                print(f"Error: {vehicle_info.feilmelding}")
            else:
                print(f"KUID: {vehicle_info.kuid}")
                print(f"VIN: {vehicle_info.understellsnummer}")
                print(f"License Plate: {vehicle_info.kjennemerke}")
                print(f"First Registered: {vehicle_info.registrert_forstgang}")
                print(f"Inspection Due: {vehicle_info.kontrollfrist}")

    except VehicleAPIError as e:
        print(f"API Error: {e.message}")
        if e.status_code:
            print(f"Status Code: {e.status_code}")
        if e.details:
            print(f"Details: {e.details}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def example_detailed_lookup():
    """Example: Detailed lookup with raw response data"""
    print("\n=== Detailed Lookup with Raw Response ===")

    # Use a real Norwegian license plate format
    license_plate = "DR21468"  # Update this with a real plate for testing

    try:
        with NorwegianVehicleAPI(enable_logging=True) as api:
            print(f"Looking up vehicle with license plate: {license_plate}")

            # First, get the raw response
            print("\n--- RAW API RESPONSE ---")
            raw_data = api.get_raw_data(license_plate=license_plate)
            print("Full raw response:")
            print(json.dumps(raw_data, indent=2, ensure_ascii=False, default=str))

            print("\n--- PARSED VEHICLE INFO ---")
            # Then get the parsed vehicle info
            vehicle_info = api.lookup_by_license_plate(license_plate)

            if vehicle_info.feilmelding:
                print(f"Error: {vehicle_info.feilmelding}")
            else:
                print("Parsed vehicle information:")
                for field_name in vehicle_info.__dataclass_fields__:
                    value = getattr(vehicle_info, field_name)
                    if value is not None:
                        print(f"  {field_name}: {value}")

    except VehicleAPIError as e:
        print(f"API Error: {e.message}")
        if e.status_code:
            print(f"Status Code: {e.status_code}")
        if e.details:
            print(f"Details: {e.details}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def example_raw_data():
    """Example: Get raw API response"""
    print("\n=== Raw Data Example ===")

    license_plate = "AB12345"

    try:
        with NorwegianVehicleAPI() as api:
            print(f"Getting raw data for license plate: {license_plate}")
            raw_data = api.get_raw_data(license_plate=license_plate)

            print("Raw API Response:")
            print(f"  Error Message: {raw_data.get('feilmelding', 'None')}")
            print(f"  Number of vehicles: {len(raw_data.get('kjoretoydataListe', []))}")

            if raw_data.get("kjoretoydataListe"):
                vehicle = raw_data["kjoretoydataListe"][0]
                print(
                    f"  First vehicle KUID: {vehicle.get('kjoretoyId', {}).get('kuid', 'N/A')}"
                )

    except VehicleAPIError as e:
        print(f"API Error: {e.message}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def example_connection_test():
    """Example: Test API connection"""
    print("\n=== Connection Test Example ===")

    try:
        with NorwegianVehicleAPI() as api:
            if api.test_connection():
                print("✓ API connection successful")
            else:
                print("✗ API connection failed")

    except Exception as e:
        print(f"Connection test error: {str(e)}")


def interactive_lookup():
    """Interactive vehicle lookup"""
    print("\n=== Interactive Vehicle Lookup ===")

    while True:
        print("\nChoose lookup method:")
        print("1. License plate")
        print("2. VIN")
        print("3. Exit")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "3":
            break
        elif choice == "1":
            plate = input("Enter license plate: ").strip()
            if plate:
                try:
                    vehicle_info = lookup_vehicle_by_plate(plate)
                    print_vehicle_info(vehicle_info)
                except Exception as e:
                    print(f"Error: {str(e)}")
        elif choice == "2":
            vin = input("Enter VIN: ").strip()
            if vin:
                try:
                    vehicle_info = lookup_vehicle_by_vin(vin)
                    print_vehicle_info(vehicle_info)
                except Exception as e:
                    print(f"Error: {str(e)}")
        else:
            print("Invalid choice")


def print_vehicle_info(vehicle_info):
    """Print vehicle information in a formatted way"""
    print("\n--- Vehicle Information ---")
    if vehicle_info.feilmelding:
        print(f"Error: {vehicle_info.feilmelding}")
    else:
        print(f"KUID: {vehicle_info.kuid or 'N/A'}")
        print(f"VIN: {vehicle_info.understellsnummer or 'N/A'}")
        print(f"License Plate: {vehicle_info.kjennemerke or 'N/A'}")
        print(f"Brand: {vehicle_info.merke or 'N/A'}")
        print(f"Model: {vehicle_info.modell or 'N/A'}")
        print(f"Own Weight: {vehicle_info.egenvekt or 'N/A'} kg")
        print(f"Total Weight: {vehicle_info.totalvekt or 'N/A'} kg")
        print(f"Fuel Type: {vehicle_info.drivstoff or 'N/A'}")
        print(f"First Registered: {vehicle_info.registrert_forstgang or 'N/A'}")
        print(f"Inspection Due: {vehicle_info.kontrollfrist or 'N/A'}")
        print(f"Owner: {vehicle_info.eier_navn or 'N/A'}")


def main():
    """Main function"""
    print("Norwegian Vehicle API Client Examples")
    print("=====================================")

    # Check if API key is configured
    api_key = os.getenv("VEHICLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("\n⚠️  Warning: No API key configured!")
        print("Set VEHICLE_API_KEY environment variable or update .env file")
        print("Some examples may not work without proper authentication.\n")

    # Run examples
    example_connection_test()
    example_detailed_lookup()  # Show detailed raw response first
    example_license_plate_lookup()
    example_vin_lookup()
    example_raw_data()

    # Interactive mode
    try:
        interactive_lookup()
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()
