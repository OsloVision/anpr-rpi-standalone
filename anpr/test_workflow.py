#!/usr/bin/env python3
"""
Simple test to simulate license plate reading with database integration.
"""

from license_plate_reader import upsert_loan_status, check_norwegian_registry


def test_full_workflow():
    """Test the full workflow simulation."""
    print("Testing full license plate workflow...")
    print("=" * 60)

    # Simulate some test license plates
    test_plates = ["EF49617", "ABC123", "XYZ999"]

    for plate in test_plates:
        print(f"\n🔍 Processing license plate: {plate}")

        # Simulate registry check (normally this would be real API call)
        print(f"   Checking Norwegian registry for {plate}...")
        # For testing, let's make EF49617 "yes" and others "no"
        if plate == "EF49617":
            registry_result = "yes"
        else:
            registry_result = "no"

        print(f"   Registry result: {registry_result}")

        # Upsert to database
        success = upsert_loan_status(plate, registry_result)
        print(f"   Database upsert: {'✅ Success' if success else '❌ Failed'}")

    print(f"\n" + "=" * 60)
    print("Checking database contents...")

    from loan_db_utils import LoanStatusDB

    db = LoanStatusDB()
    records = db.list_all_records()

    print(f"Total records: {len(records)}")
    for record in records:
        print(f"  {record[1]} -> {record[2]} (created: {record[3]})")


if __name__ == "__main__":
    test_full_workflow()
