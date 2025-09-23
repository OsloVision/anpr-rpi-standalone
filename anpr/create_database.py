#!/usr/bin/env python3
"""
Create SQLite database with loan_status table.
"""

import sqlite3
import os


def create_loan_status_database(db_path="loan_status.db"):
    """
    Create SQLite database with loan_status table.

    Args:
        db_path (str): Path to the SQLite database file

    Returns:
        str: Path to the created database
    """
    try:
        # Connect to SQLite database (creates file if it doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create loan_status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loan_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                numberplate TEXT NOT NULL,
                loan TEXT NOT NULL,
                info JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create unique index on numberplate to prevent duplicates
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_numberplate_unique 
            ON loan_status(numberplate)
        """)

        # Create trigger to automatically update updated_at column
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_loan_status_updated_at
            AFTER UPDATE ON loan_status
            FOR EACH ROW
            BEGIN
                UPDATE loan_status SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """)

        # Commit the changes
        conn.commit()

        print(f"✅ Database created successfully: {os.path.abspath(db_path)}")
        print(
            "✅ Table 'loan_status' created with columns: id, numberplate, loan, info, created_at, updated_at"
        )
        print("✅ Unique index created on 'numberplate' column")
        print("✅ Trigger created to automatically update 'updated_at' column")

        # Display table schema for verification
        cursor.execute("PRAGMA table_info(loan_status)")
        columns = cursor.fetchall()
        print("\nTable schema:")
        for column in columns:
            print(f"  - {column[1]} ({column[2]}) {'PRIMARY KEY' if column[5] else ''}")

        # Display indexes
        cursor.execute("PRAGMA index_list(loan_status)")
        indexes = cursor.fetchall()
        print("\nIndexes:")
        for index in indexes:
            print(f"  - {index[1]} (unique: {bool(index[2])})")

        return os.path.abspath(db_path)

    except sqlite3.Error as e:
        print(f"❌ Error creating database: {e}")
        return None
    finally:
        if conn:
            conn.close()


def test_database_functionality(db_path="loan_status.db"):
    """
    Test the database functionality with sample data.

    Args:
        db_path (str): Path to the SQLite database file
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("\n" + "=" * 50)
        print("Testing database functionality...")
        print("=" * 50)

        # Test inserting data
        test_data = [
            ("ABC123", "active"),
            ("XYZ789", "inactive"),
            ("EF49617", "pending"),
        ]

        for numberplate, loan in test_data:
            try:
                cursor.execute(
                    """
                    INSERT INTO loan_status (numberplate, loan) 
                    VALUES (?, ?)
                """,
                    (numberplate, loan),
                )
                print(f"✅ Inserted: {numberplate} - {loan}")
            except sqlite3.IntegrityError:
                print(f"⚠️  Skipped duplicate: {numberplate}")

        conn.commit()

        # Test querying data
        cursor.execute("SELECT * FROM loan_status")
        rows = cursor.fetchall()

        print(f"\nCurrent records in loan_status table ({len(rows)} rows):")
        print("-" * 70)
        for row in rows:
            print(f"ID: {row[0]}, Numberplate: {row[1]}, Loan: {row[2]}")
            print(f"    Created: {row[3]}, Updated: {row[4]}")

        # Test unique constraint
        print("\nTesting unique constraint...")
        try:
            cursor.execute(
                """
                INSERT INTO loan_status (numberplate, loan) 
                VALUES (?, ?)
            """,
                ("ABC123", "duplicate_test"),
            )
            print("❌ Unique constraint failed - duplicate was inserted!")
        except sqlite3.IntegrityError:
            print("✅ Unique constraint working - duplicate rejected")

        return True

    except sqlite3.Error as e:
        print(f"❌ Error testing database: {e}")
        return False
    finally:
        if conn:
            conn.close()


def main():
    """Main function to create and test the database."""
    print("Creating SQLite database with loan_status table...")
    print("=" * 60)

    # Create database
    db_path = create_loan_status_database()

    if db_path:
        # Test the database
        test_database_functionality(db_path)

        print("\n" + "=" * 60)
        print("Database setup complete!")
        print(f"Database location: {db_path}")
        print("\nYou can now use this database in your applications.")
    else:
        print("❌ Failed to create database")


if __name__ == "__main__":
    main()
