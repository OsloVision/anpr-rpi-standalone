#!/usr/bin/env python3
"""
Utility functions for interacting with the loan_status database.
"""

import sqlite3
import sys
from typing import List, Tuple, Optional


class LoanStatusDB:
    """Class to handle loan status database operations."""

    def __init__(self, db_path="loan_status.db"):
        """Initialize database connection."""
        self.db_path = db_path

    def connect(self):
        """Create database connection."""
        return sqlite3.connect(self.db_path)

    def add_loan_status(
        self, numberplate: str, loan: str, info: Optional[str] = None
    ) -> bool:
        """
        Add a new loan status record.

        Args:
            numberplate (str): License plate number
            loan (str): Loan status
            info (str, optional): JSON string with additional vehicle information

        Returns:
            bool: True if successful, False if duplicate or error
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                if info:
                    cursor.execute(
                        """
                        INSERT INTO loan_status (numberplate, loan, info) 
                        VALUES (?, ?, json(?))
                    """,
                        (numberplate, loan, info),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO loan_status (numberplate, loan) 
                        VALUES (?, ?)
                    """,
                        (numberplate, loan),
                    )
                conn.commit()
                print(f"✅ Added: {numberplate} - {loan}")
                return True
        except sqlite3.IntegrityError:
            print(f"⚠️  Numberplate {numberplate} already exists in database")
            return False
        except sqlite3.Error as e:
            print(f"❌ Error adding record: {e}")
            return False

    def get_loan_status(
        self, numberplate: str
    ) -> Optional[Tuple[int, str, str, Optional[str], str, str]]:
        """
        Get loan status for a specific numberplate.

        Args:
            numberplate (str): License plate number

        Returns:
            Tuple[int, str, str, str, str, str] or None: (id, numberplate, loan, info, created_at, updated_at) or None if not found
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, numberplate, loan, info, created_at, updated_at
                    FROM loan_status 
                    WHERE numberplate = ?
                """,
                    (numberplate,),
                )
                result = cursor.fetchone()
                return result
        except sqlite3.Error as e:
            print(f"❌ Error querying database: {e}")
            return None

    def update_loan_status(
        self, numberplate: str, loan: str, info: Optional[str] = None
    ) -> bool:
        """
        Update loan status for existing numberplate.

        Args:
            numberplate (str): License plate number
            loan (str): New loan status
            info (str, optional): JSON string with additional vehicle information

        Returns:
            bool: True if successful, False if not found or error
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                if info:
                    cursor.execute(
                        """
                        UPDATE loan_status 
                        SET loan = ?, info = json(?)
                        WHERE numberplate = ?
                    """,
                        (loan, info, numberplate),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE loan_status 
                        SET loan = ? 
                        WHERE numberplate = ?
                    """,
                        (loan, numberplate),
                    )

                if cursor.rowcount > 0:
                    conn.commit()
                    print(f"✅ Updated: {numberplate} - {loan}")
                    return True
                else:
                    print(f"⚠️  Numberplate {numberplate} not found")
                    return False

        except sqlite3.Error as e:
            print(f"❌ Error updating record: {e}")
            return False

    def delete_loan_status(self, numberplate: str) -> bool:
        """
        Delete loan status record.

        Args:
            numberplate (str): License plate number

        Returns:
            bool: True if successful, False if not found or error
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM loan_status 
                    WHERE numberplate = ?
                """,
                    (numberplate,),
                )

                if cursor.rowcount > 0:
                    conn.commit()
                    print(f"✅ Deleted: {numberplate}")
                    return True
                else:
                    print(f"⚠️  Numberplate {numberplate} not found")
                    return False

        except sqlite3.Error as e:
            print(f"❌ Error deleting record: {e}")
            return False

    def list_all_records(self) -> List[Tuple[int, str, str, Optional[str], str, str]]:
        """
        Get all loan status records.

        Returns:
            List[Tuple[int, str, str, str, str, str]]: List of (id, numberplate, loan, info, created_at, updated_at) tuples
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, numberplate, loan, info, created_at, updated_at FROM loan_status ORDER BY id"
                )
                return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"❌ Error listing records: {e}")
            return []

    def check_numberplate_exists(self, numberplate: str) -> bool:
        """
        Check if a numberplate exists in the database.

        Args:
            numberplate (str): License plate number

        Returns:
            bool: True if exists, False otherwise
        """
        result = self.get_loan_status(numberplate)
        return result is not None


def main():
    """Command line interface for database operations."""
    if len(sys.argv) < 2:
        print("Usage: python loan_db_utils.py <command> [args...]")
        print("\nCommands:")
        print("  create                     - Create database and table")
        print("  add <numberplate> <loan>   - Add new record")
        print("  get <numberplate>          - Get loan status")
        print("  update <numberplate> <loan> - Update loan status")
        print("  delete <numberplate>       - Delete record")
        print("  list                       - List all records")
        print("  exists <numberplate>       - Check if numberplate exists")
        return

    command = sys.argv[1].lower()
    db = LoanStatusDB()

    if command == "create":
        # Import and run the create script
        from create_database import create_loan_status_database

        create_loan_status_database()

    elif command == "add":
        if len(sys.argv) != 4:
            print("Usage: python loan_db_utils.py add <numberplate> <loan>")
            return
        numberplate, loan = sys.argv[2], sys.argv[3]
        db.add_loan_status(numberplate, loan)

    elif command == "get":
        if len(sys.argv) != 3:
            print("Usage: python loan_db_utils.py get <numberplate>")
            return
        numberplate = sys.argv[2]
        result = db.get_loan_status(numberplate)
        if result:
            print(f"ID: {result[0]}, Numberplate: {result[1]}, Loan: {result[2]}")
            if result[3]:  # info field
                print(f"Info: {result[3]}")
            print(f"Created: {result[4]}, Updated: {result[5]}")
        else:
            print(f"Numberplate {numberplate} not found")

    elif command == "update":
        if len(sys.argv) != 4:
            print("Usage: python loan_db_utils.py update <numberplate> <loan>")
            return
        numberplate, loan = sys.argv[2], sys.argv[3]
        db.update_loan_status(numberplate, loan)

    elif command == "delete":
        if len(sys.argv) != 3:
            print("Usage: python loan_db_utils.py delete <numberplate>")
            return
        numberplate = sys.argv[2]
        db.delete_loan_status(numberplate)

    elif command == "list":
        records = db.list_all_records()
        if records:
            print(f"Found {len(records)} records:")
            print("-" * 80)
            for record in records:
                print(f"ID: {record[0]}, Numberplate: {record[1]}, Loan: {record[2]}")
                if record[3]:  # info field
                    print(f"    Info: {record[3]}")
                print(f"    Created: {record[4]}, Updated: {record[5]}")
        else:
            print("No records found")

    elif command == "exists":
        if len(sys.argv) != 3:
            print("Usage: python loan_db_utils.py exists <numberplate>")
            return
        numberplate = sys.argv[2]
        exists = db.check_numberplate_exists(numberplate)
        print(f"Numberplate {numberplate}: {'EXISTS' if exists else 'NOT FOUND'}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
