#!/usr/bin/env python3
"""
Migration script to add 'info' JSON column to existing loan_status table.
"""

import sqlite3
import os
import sys


def migrate_database(db_path="loan_status.db"):
    """
    Add 'info' JSON column to existing loan_status table.

    Args:
        db_path (str): Path to the SQLite database file

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if info column already exists
        cursor.execute("PRAGMA table_info(loan_status)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        if "info" in column_names:
            print(f"✅ Column 'info' already exists in {db_path}")
            return True

        # Add the info column
        print(f"📊 Adding 'info' JSON column to {db_path}...")
        cursor.execute("ALTER TABLE loan_status ADD COLUMN info JSON")
        conn.commit()

        print(f"✅ Successfully added 'info' column to {db_path}")

        # Display updated table schema for verification
        cursor.execute("PRAGMA table_info(loan_status)")
        columns = cursor.fetchall()
        print("\nUpdated table schema:")
        for column in columns:
            print(f"  - {column[1]} ({column[2]}) {'PRIMARY KEY' if column[5] else ''}")

        return True

    except sqlite3.Error as e:
        print(f"❌ Error migrating database: {e}")
        return False
    finally:
        if conn:
            conn.close()


def main():
    """Main function to run the migration."""
    db_path = "loan_status.db"

    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    print("Database Migration: Adding 'info' JSON Column")
    print("=" * 50)
    print(f"Target database: {db_path}")
    print()

    success = migrate_database(db_path)

    if success:
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
