#!/usr/bin/env python3
"""
License Plate Reader using OpenAI GPT-4o-mini
Reads license plate text from an image using OpenAI's vision capabilities.
"""

import os
import sys
import base64
import argparse
import re
import requests
import json
from pathlib import Path
from openai import OpenAI
from .loan_db_utils import LoanStatusDB
from .norwegian_vehicle_api import NorwegianVehicleAPI, VehicleAPIError


def encode_image(image_path):
    """Encode image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}", file=sys.stderr)
        sys.exit(1)


def read_license_plate(image_path, api_key=None):
    """
    Read license plate from image using OpenAI GPT-4o-mini.

    Args:
        image_path (str): Path to the image file
        api_key (str): OpenAI API key (optional, can use environment variable)

    Returns:
        str: License plate text
    """
    # Initialize OpenAI client
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        # Will use OPENAI_API_KEY environment variable
        client = OpenAI()

    # Encode the image
    base64_image = encode_image(image_path)

    # Determine image format
    image_ext = Path(image_path).suffix.lower()
    if image_ext in [".jpg", ".jpeg"]:
        image_format = "jpeg"
    elif image_ext == ".png":
        image_format = "png"
    elif image_ext == ".webp":
        image_format = "webp"
    else:
        image_format = "jpeg"  # Default fallback

    try:
        # Send request to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read the license plate in this image. Output should just be the text on the license plate, nothing else.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=50,
            temperature=0,
        )

        # Extract and return the license plate text
        license_plate_text = response.choices[0].message.content.strip()
        # remove whitespace
        license_plate_text = re.sub(r"\s+", "", license_plate_text)
        return license_plate_text

    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)
        sys.exit(1)


def lookup_norwegian_vehicle_registry(numberplate):
    """
    Look up vehicle information from the Norwegian vehicle registry API.

    Args:
        numberplate (str): The license plate number to check

    Returns:
        tuple: (status, raw_data_dict) where status is "yes"/"no"/"error"
               and raw_data_dict contains the full API response
    """
    try:
        # Initialize the Norwegian Vehicle API
        api = NorwegianVehicleAPI()

        # Perform the lookup
        vehicle_info = api.lookup_by_license_plate(numberplate)

        # Get the raw data as well for storing in the database
        raw_data = api.get_raw_data(license_plate=numberplate)

        # Determine status based on whether we got vehicle data
        if vehicle_info.feilmelding:
            # There was an error or no data found
            status = "no"
        elif vehicle_info.kuid or vehicle_info.kjennemerke:
            # We found valid vehicle data
            status = "yes"
        else:
            # No clear data found
            status = "no"

        return status, raw_data

    except VehicleAPIError as e:
        print(f"Vehicle API error for {numberplate}: {e}", file=sys.stderr)
        return "error", {"error": str(e), "error_type": "VehicleAPIError"}
    except Exception as e:
        print(f"Unexpected error looking up {numberplate}: {e}", file=sys.stderr)
        return "error", {"error": str(e), "error_type": "UnexpectedError"}


def check_norwegian_registry(numberplate):
    """
    Check if a license plate is registered in the Norwegian vehicle registry.

    Args:
        numberplate (str): The license plate number to check

    Returns:
        str: "yes" if registered (x > 0), "no" if not registered or error
    """
    try:
        # Format the URL with the license plate number
        url = f"https://rettsstiftelser.brreg.no/nb/oppslag/motorvogn/{numberplate}"

        # Make GET request with proper headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        print("hitting url:", url)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # print(response.text)

        # Search for the pattern in the response text
        pattern = r"Det er (\d+) oppføring på registreringsnummer"
        match = re.search(pattern, response.text)

        if match:
            print("Pattern found in response.")
            count = int(match.group(1))
            return "yes" if count > 0 else "no"
        else:
            print("Pattern not found in response.")
            # Pattern not found, assume not registered
            return "no"

    except requests.exceptions.RequestException as e:
        print(f"Error checking Norwegian registry: {e}", file=sys.stderr)
        return "no"
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return "no"


def check_both_norwegian_services(numberplate):
    """
    Check both Norwegian registry services and combine the results.

    Args:
        numberplate (str): The license plate number to check

    Returns:
        tuple: (combined_status, combined_info) where:
            - combined_status: "yes" if either service found data, "no" if both failed, "partial" if mixed results
            - combined_info: dict containing results from both services
    """
    import datetime

    combined_info = {
        "numberplate": numberplate,
        "scraping_service": {},
        "official_api": {},
        "timestamp": datetime.datetime.now().isoformat(),
        "services_checked": ["brreg_scraping", "vegvesen_api"],
    }

    # Check scraping service (Brønnøysund Register Centre)
    print("🔍 Checking Brønnøysund Register Centre (scraping)...")
    try:
        scraping_result = check_norwegian_registry(numberplate)
        combined_info["scraping_service"] = {
            "status": scraping_result,
            "source": "brreg.no_scraping",
            "service": "Brønnøysund Register Centre",
            "method": "web_scraping",
            "error": None,
        }
        print(f"   Scraping result: {scraping_result}")
    except Exception as e:
        print(f"   Scraping error: {e}")
        combined_info["scraping_service"] = {
            "status": "error",
            "source": "brreg.no_scraping",
            "service": "Brønnøysund Register Centre",
            "method": "web_scraping",
            "error": str(e),
        }

    # Check official API (Norwegian Public Roads Administration)
    print("🔍 Checking Norwegian Public Roads Administration (official API)...")
    try:
        api_status, api_data = lookup_norwegian_vehicle_registry(numberplate)

        # Convert any date objects to strings for JSON serialization
        if api_data and isinstance(api_data, dict):
            api_data = json.loads(json.dumps(api_data, default=str))

        combined_info["official_api"] = {
            "status": api_status,
            "source": "vegvesen_api",
            "service": "Norwegian Public Roads Administration",
            "method": "official_api",
            "raw_data": api_data,
            "error": None,
        }
        print(f"   API result: {api_status}")
    except Exception as e:
        print(f"   API error: {e}")
        combined_info["official_api"] = {
            "status": "error",
            "source": "vegvesen_api",
            "service": "Norwegian Public Roads Administration",
            "method": "official_api",
            "raw_data": None,
            "error": str(e),
        }

    # Determine combined status
    scraping_status = combined_info["scraping_service"]["status"]
    api_status = combined_info["official_api"]["status"]

    if scraping_status == "yes" and api_status == "yes":
        combined_status = "yes"
    elif scraping_status == "yes" or api_status == "yes":
        combined_status = "partial"  # Found in one service but not the other
    elif scraping_status == "error" or api_status == "error":
        combined_status = "error"
    else:
        combined_status = "no"

    # Add summary
    combined_info["summary"] = {
        "final_status": combined_status,
        "scraping_found": scraping_status == "yes",
        "api_found": api_status == "yes",
        "both_agree": scraping_status == api_status,
        "services_available": sum(
            1 for s in [scraping_status, api_status] if s not in ["error", "no"]
        ),
    }

    return combined_status, combined_info


def check_plate_only(numberplate):
    """
    Standalone function to check if a license plate is registered in Norwegian registry.

    Args:
        numberplate (str): The license plate number to check

    Returns:
        str: "yes" if registered (x > 0), "no" if not registered or error
    """
    return check_norwegian_registry(numberplate)


def upsert_loan_status(
    numberplate, loan_status, info_data=None, db_path="loan_status.db"
):
    """
    Insert or update loan status in the database.

    Args:
        numberplate (str): License plate number
        loan_status (str): Loan status ("yes" for registered, "no" for not registered)
        info_data (dict, optional): Additional vehicle information as dictionary
        db_path (str): Path to the SQLite database file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import sqlite3
        import os

        # Check if database file exists, if not create it with proper schema
        if not os.path.exists(db_path):
            print(f"📁 Creating database: {db_path}")
            # Create the database with the schema
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table
            cursor.execute("""
                CREATE TABLE loan_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    numberplate TEXT NOT NULL,
                    loan TEXT NOT NULL,
                    info JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create unique index
            cursor.execute("""
                CREATE UNIQUE INDEX idx_numberplate_unique 
                ON loan_status(numberplate)
            """)

            # Create trigger
            cursor.execute("""
                CREATE TRIGGER update_loan_status_updated_at
                AFTER UPDATE ON loan_status
                FOR EACH ROW
                BEGIN
                    UPDATE loan_status SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """)

            conn.commit()
            conn.close()
            print(f"✅ Database created: {db_path}")

        db = LoanStatusDB(db_path)

        # Convert info_data to JSON string if provided
        info_json = None
        if info_data:
            info_json = json.dumps(info_data)

        # Check if record exists
        existing_record = db.get_loan_status(numberplate)

        if existing_record:
            # Update existing record
            success = db.update_loan_status(numberplate, loan_status, info_json)
            if success:
                print(f"📝 Updated database: {numberplate} -> {loan_status}")
            return success
        else:
            # Insert new record
            success = db.add_loan_status(numberplate, loan_status, info_json)
            if success:
                print(f"💾 Added to database: {numberplate} -> {loan_status}")
            return success

    except Exception as e:
        print(f"❌ Database error: {e}", file=sys.stderr)
        return False


def main():
    """Main function to handle command line arguments and execute license plate reading."""
    parser = argparse.ArgumentParser(
        description="Read license plate text from an image using OpenAI GPT-4o-mini"
    )
    parser.add_argument(
        "image_path", help="Path to the image file containing the license plate"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (optional, can use OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--check-registry",
        action="store_true",
        help="Check Norwegian vehicle registry using the official API",
    )
    parser.add_argument(
        "--check-both",
        action="store_true",
        help="Check both Norwegian registry services (scraping + official API) and compare results",
    )
    parser.add_argument(
        "--use-scraping",
        action="store_true",
        help="Use web scraping method instead of official API for registry check (fallback method)",
    )
    parser.add_argument(
        "--db-path",
        default="loan_status.db",
        help="Path to the SQLite database file (default: loan_status.db)",
    )

    args = parser.parse_args()

    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Check if API key is available
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or use --api-key argument.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read the license plate
    try:
        license_plate_text = read_license_plate(args.image_path, api_key)
        print(license_plate_text)

        # Check Norwegian registry if requested
        if args.check_both:
            # Check both services and compare results
            registry_result, registry_info = check_both_norwegian_services(
                license_plate_text
            )
            print(f"Combined registry result: {registry_result}")

            # Print summary
            summary = registry_info["summary"]
            print("📊 Summary:")
            print(
                f"   Scraping service found: {'✅' if summary['scraping_found'] else '❌'}"
            )
            print(f"   Official API found: {'✅' if summary['api_found'] else '❌'}")
            print(f"   Services agree: {'✅' if summary['both_agree'] else '❌'}")

            # Upsert into database
            upsert_loan_status(
                license_plate_text, registry_result, registry_info, args.db_path
            )

        elif args.check_registry:
            if args.use_scraping:
                # Use the old web scraping method
                registry_result = check_norwegian_registry(license_plate_text)
                registry_info = {
                    "method": "scraping_only",
                    "service": "Brønnøysund Register Centre",
                    "result": registry_result,
                }
            else:
                # Use the official API (default)
                registry_result, api_data = lookup_norwegian_vehicle_registry(
                    license_plate_text
                )
                registry_info = {
                    "method": "api_only",
                    "service": "Norwegian Public Roads Administration",
                    "result": registry_result,
                    "raw_data": api_data,
                }

            print(f"Norwegian registry: {registry_result}")

            # Upsert into database
            upsert_loan_status(
                license_plate_text, registry_result, registry_info, args.db_path
            )

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
