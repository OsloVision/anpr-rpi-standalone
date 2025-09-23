#!/usr/bin/env python3
"""
Test script for checking both Norwegian registry services.
"""

import argparse
import json
import sys
from license_plate_reader import check_both_norwegian_services, upsert_loan_status


def test_dual_lookup(numberplate, db_path="loan_status.db", verbose=False):
    """
    Test both Norwegian registry services and store the result.

    Args:
        numberplate (str): License plate to look up
        db_path (str): Database path
        verbose (bool): Whether to print detailed information

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Testing dual lookup for license plate: {numberplate}")
        print("=" * 60)

        # Perform the dual lookup
        status, info = check_both_norwegian_services(numberplate)

        print(f"\n🎯 Final Result: {status}")

        # Show detailed comparison
        scraping = info["scraping_service"]
        api = info["official_api"]
        summary = info["summary"]

        print("\n📋 Detailed Results:")
        print(f"   Scraping Service ({scraping['service']}): {scraping['status']}")
        print(f"   Official API ({api['service']}): {api['status']}")
        print(f"   Agreement: {'✅ Yes' if summary['both_agree'] else '❌ No'}")
        print(f"   Services Available: {summary['services_available']}/2")

        if verbose:
            print("\n🔍 Full Information:")
            print(json.dumps(info, indent=2, ensure_ascii=False))

        # Store in database
        print(f"\n💾 Storing result in database: {db_path}")
        success = upsert_loan_status(numberplate, status, info, db_path)

        if success:
            print(f"✅ Successfully stored: {numberplate} -> {status}")
        else:
            print("❌ Failed to store result")

        return success

    except Exception as e:
        print(f"❌ Error during dual lookup: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test both Norwegian registry services and compare results"
    )
    parser.add_argument("numberplate", help="License plate number to look up")
    parser.add_argument(
        "--db-path",
        default="loan_status.db",
        help="Path to the SQLite database file (default: loan_status.db)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed API responses"
    )

    args = parser.parse_args()

    success = test_dual_lookup(args.numberplate, args.db_path, args.verbose)

    if not success:
        sys.exit(1)

    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()
