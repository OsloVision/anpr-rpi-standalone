#!/usr/bin/env python3
"""
Test file for the Norwegian registry regex pattern.
Tests the regex on line 117 of license_plate_reader.py
"""

import re
import unittest


class TestNorwegianRegistryRegex(unittest.TestCase):
    """Test cases for the Norwegian vehicle registry regex pattern."""

    def setUp(self):
        """Set up the regex pattern used in the main application."""
        # This is the pattern from line 117 in license_plate_reader.py
        self.pattern = r"Det er (\d+) oppføring på registreringsnummer"

    def test_regex_matches_html_content(self):
        """Test that the regex successfully matches the provided HTML content."""
        # The HTML content provided by the user
        html_content = '<p class="MotorvognResultSection-rettsstiftelseCount-2_7591247065">Det er 1 oppføring på registreringsnummer EF 49617.</p>'

        # Test that the pattern matches
        match = re.search(self.pattern, html_content)
        self.assertIsNotNone(match, "Regex should match the HTML content")

        # Test that the captured group contains the expected count
        self.assertEqual(match.group(1), "1", "Should capture the count '1'")

        # Test that the full match contains the expected text
        expected_match = "Det er 1 oppføring på registreringsnummer"
        self.assertEqual(
            match.group(0), expected_match, f"Full match should be '{expected_match}'"
        )

    def test_regex_with_different_counts(self):
        """Test the regex with different registration counts."""
        test_cases = [
            ("Det er 0 oppføring på registreringsnummer ABC123", "0"),
            ("Det er 1 oppføring på registreringsnummer EF 49617", "1"),
            ("Det er 2 oppføring på registreringsnummer XY123", "2"),
            ("Det er 15 oppføring på registreringsnummer TEST99", "15"),
            ("Det er 999 oppføring på registreringsnummer MULTI1", "999"),
        ]

        for text, expected_count in test_cases:
            with self.subTest(text=text, expected_count=expected_count):
                match = re.search(self.pattern, text)
                self.assertIsNotNone(match, f"Should match text: {text}")
                self.assertEqual(
                    match.group(1),
                    expected_count,
                    f"Should capture count '{expected_count}'",
                )

    def test_regex_with_html_context(self):
        """Test the regex works within various HTML contexts."""
        html_test_cases = [
            '<p class="some-class">Det er 1 oppføring på registreringsnummer EF 49617.</p>',
            "<div>Det er 5 oppføring på registreringsnummer AB123</div>",
            "<span>Before text Det er 3 oppføring på registreringsnummer CD456 after text</span>",
            "No HTML tags Det er 7 oppføring på registreringsnummer XY789 plain text",
        ]

        expected_counts = ["1", "5", "3", "7"]

        for html_content, expected_count in zip(html_test_cases, expected_counts):
            with self.subTest(html_content=html_content):
                match = re.search(self.pattern, html_content)
                self.assertIsNotNone(match, f"Should match HTML: {html_content}")
                self.assertEqual(
                    match.group(1),
                    expected_count,
                    f"Should capture count '{expected_count}'",
                )

    def test_regex_no_match_cases(self):
        """Test cases where the regex should not match."""
        no_match_cases = [
            "Det er oppføring på registreringsnummer ABC123",  # Missing count
            "Det er X oppføring på registreringsnummer ABC123",  # Non-numeric count
            "Det var 1 oppføring på registreringsnummer ABC123",  # Wrong verb
            "Det er 1 oppførings på registreringsnummer ABC123",  # Wrong form
            "Det er 1 oppføring for registreringsnummer ABC123",  # Wrong preposition
            "There are 1 registration for number ABC123",  # English
            "",  # Empty string
            "Random text without the pattern",
        ]

        for text in no_match_cases:
            with self.subTest(text=text):
                match = re.search(self.pattern, text)
                self.assertIsNone(match, f"Should not match text: {text}")

    def test_original_html_example(self):
        """Test the specific HTML example provided by the user."""
        original_html = '<p class="MotorvognResultSection-rettsstiftelseCount-2_7591247065">Det er 1 oppføring på registreringsnummer EF 49617.</p>'

        match = re.search(self.pattern, original_html)

        # Verify the match is successful
        self.assertIsNotNone(match, "Should match the original HTML example")

        # Verify the captured count
        self.assertEqual(
            match.group(1), "1", "Should capture count '1' from the original example"
        )

        # Verify this would result in "yes" for registration check
        count = int(match.group(1))
        result = "yes" if count > 0 else "no"
        self.assertEqual(result, "yes", "Should return 'yes' for count > 0")


def run_individual_test():
    """Run a single test to verify the regex against the provided HTML."""
    pattern = r"Det er (\d+) oppføring på registreringsnummer"
    html_content = '<p class="MotorvognResultSection-rettsstiftelseCount-2_7591247065">Det er 1 oppføring på registreringsnummer EF 49617.</p>'

    print("Testing regex pattern:")
    print(f"Pattern: {pattern}")
    print(f"HTML content: {html_content}")
    print()

    match = re.search(pattern, html_content)

    if match:
        print("✅ SUCCESS: Regex matched!")
        print(f"Full match: '{match.group(0)}'")
        print(f"Captured count: '{match.group(1)}'")
        count = int(match.group(1))
        result = "yes" if count > 0 else "no"
        print(f"Registry result would be: {result}")
    else:
        print("❌ FAILURE: Regex did not match!")


if __name__ == "__main__":
    print("=" * 60)
    print("Norwegian Registry Regex Test")
    print("=" * 60)
    print()

    # Run individual test first
    run_individual_test()

    print()
    print("=" * 60)
    print("Running full test suite...")
    print("=" * 60)

    # Run the full test suite
    unittest.main(verbosity=2)
