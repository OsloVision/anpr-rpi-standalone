#!/usr/bin/env python3
"""
Test script for Norwegian vehicle registry lookup using the official API.
"""

import argparse
import json
import sys
import os
from norwegian_vehicle_api import NorwegianVehicleAPI, VehicleAPIError
from loan_db_utils import LoanStatusDB


def lookup_and_store(numberplate, db_path="loan_status.db", verbose=False):
    """
    Look up a license plate in the Norwegian registry and store the result.

    Args:
        numberplate (str): License plate to look up
        db_path (str): Database path
        verbose (bool): Whether to print detailed information

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize the Norwegian Vehicle API
        api = NorwegianVehicleAPI()

        print(f"Looking up license plate: {numberplate}")

        # Perform the lookup
        vehicle_info = api.lookup_by_license_plate(numberplate)

        # Get the raw data as well for storing in the database
        raw_data = api.get_raw_data(license_plate=numberplate)

        # Print basic information
        if vehicle_info.feilmelding:
            print(f"❌ Error or not found: {vehicle_info.feilmelding}")
            status = "no"
        elif vehicle_info.kuid or vehicle_info.kjennemerke:
            print("✅ Vehicle found!")
            if vehicle_info.kjennemerke:
                print(f"   License plate: {vehicle_info.kjennemerke}")
            if vehicle_info.merke:
                print(f"   Brand: {vehicle_info.merke}")
            if vehicle_info.modell:
                print(f"   Model: {vehicle_info.modell}")
            if vehicle_info.arsmodell:
                print(f"   Year: {vehicle_info.arsmodell}")
            if vehicle_info.registrert_forstgang:
                print(f"   First registered: {vehicle_info.registrert_forstgang}")
            status = "yes"
        else:
            print("❓ No clear data found")
            status = "no"

        if verbose:
            print("\nRaw API response:")
            print(json.dumps(raw_data, indent=2, ensure_ascii=False))

        # Store in database
        print(f"\nStoring result in database: {db_path}")
        db = LoanStatusDB(db_path)

        # Convert raw_data to JSON string
        info_json = json.dumps(raw_data) if raw_data else None

        # Check if record exists
        existing_record = db.get_loan_status(numberplate)

        if existing_record:
            # Update existing record
            success = db.update_loan_status(numberplate, status, info_json)
            if success:
                print(f"📝 Updated database: {numberplate} -> {status}")
        else:
            # Insert new record
            success = db.add_loan_status(numberplate, status, info_json)
            if success:
                print(f"💾 Added to database: {numberplate} -> {status}")

        return True

    except VehicleAPIError as e:
        print(f"❌ Vehicle API error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test Norwegian vehicle registry lookup using official API"
    )
    parser.add_argument("numberplate", help="License plate number to look up")
    parser.add_argument(
        "--db-path",
        default="loan_status.db",
        help="Path to the SQLite database file (default: loan_status.db)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed API response"
    )

    args = parser.parse_args()

    # Check if we have required environment variables for the API
    api_key = os.getenv("VEHICLE_API_KEY")
    if not api_key:
        print("❌ Error: VEHICLE_API_KEY environment variable not found.")
        print("Please set your Norwegian Vehicle API key as environment variable.")
        sys.exit(1)

    success = lookup_and_store(args.numberplate, args.db_path, args.verbose)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
