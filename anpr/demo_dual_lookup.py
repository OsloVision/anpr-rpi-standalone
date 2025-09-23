#!/usr/bin/env python3
"""
Quick test script to demonstrate the dual Norwegian registry lookup functionality.
"""

from license_plate_reader import check_both_norwegian_services, upsert_loan_status


def demo_dual_lookup():
    """Demonstrate the dual lookup functionality with multiple license plates."""

    test_plates = [
        "EF49617",  # Known good plate
        "AB12345",  # Generic test plate
        "XY99999",  # Likely non-existent
    ]

    print("🚗 Norwegian Vehicle Registry Dual Lookup Demo")
    print("=" * 60)
    print("Checking both services:")
    print("  1. Brønnøysund Register Centre (web scraping)")
    print("  2. Norwegian Public Roads Administration (official API)")
    print()

    for plate in test_plates:
        print(f"Testing: {plate}")
        print("-" * 40)

        try:
            status, info = check_both_norwegian_services(plate)

            # Show summary
            summary = info["summary"]
            scraping = info["scraping_service"]
            api = info["official_api"]

            print(f"Final Status: {status}")
            print(
                f"Services Agreement: {'✅ Yes' if summary['both_agree'] else '❌ No'}"
            )
            print(f"Scraping Service: {scraping['status']} ({scraping['service']})")
            print(f"Official API: {api['status']} ({api['service']})")
            print(f"Available Services: {summary['services_available']}/2")

            # Store in database
            upsert_loan_status(plate, status, info, "demo_dual_lookup.db")

        except Exception as e:
            print(f"❌ Error testing {plate}: {e}")

        print()


if __name__ == "__main__":
    demo_dual_lookup()
    print("✅ Demo completed! Check demo_dual_lookup.db for stored results.")
