#!/usr/bin/env python3
"""
Test script to verify the database integration works correctly.
"""

from license_plate_reader import upsert_loan_status
from loan_db_utils import LoanStatusDB


def test_upsert_functionality():
    """Test the upsert functionality."""
    print("Testing upsert functionality...")
    print("=" * 50)

    # Test data
    test_numberplate = "TEST999"
    db_path = "test_loan_status.db"

    # Create test database
    db = LoanStatusDB(db_path)

    print("1. Testing initial insert...")
    success = upsert_loan_status(test_numberplate, "yes", db_path)
    print(f"Insert result: {success}")

    # Check if record exists
    record = db.get_loan_status(test_numberplate)
    if record:
        print(f"✅ Record found: {record}")
    else:
        print("❌ Record not found")

    print("\n2. Testing update of existing record...")
    success = upsert_loan_status(test_numberplate, "no", db_path)
    print(f"Update result: {success}")

    # Check updated record
    record = db.get_loan_status(test_numberplate)
    if record:
        print(f"✅ Updated record: {record}")
        print(f"   Created: {record[3]}")
        print(f"   Updated: {record[4]}")
    else:
        print("❌ Record not found after update")

    print("\n3. Testing with different numberplate...")
    success = upsert_loan_status("ANOTHER123", "yes", db_path)
    print(f"Insert result: {success}")

    # List all records
    print("\n4. All records in test database:")
    records = db.list_all_records()
    for record in records:
        print(f"   {record}")

    print(f"\n✅ Test completed! Test database: {db_path}")


if __name__ == "__main__":
    test_upsert_functionality()
