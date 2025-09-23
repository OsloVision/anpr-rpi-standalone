"""
Norwegian Vehicle Data API Client
Implements HTTP client for Statens Vegvesen vehicle data API
"""

import requests
import logging
from typing import Optional, Dict, Any
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .vehicle_api_models import KjoretoydataResponse


class VehicleAPIError(Exception):
    """Custom exception for Vehicle API errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class NorwegianVehicleAPIClient:
    """
    Client for Norwegian Vehicle Data API (Statens Vegvesen)
    """

    def __init__(
        self,
        base_url: str = "https://akfell-datautlevering.atlas.vegvesen.no",
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Vehicle API client

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
            api_key: API key for authentication (if required)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

        # Setup session with retry strategy
        self.session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff_factor,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update(
            {
                "User-Agent": "Norwegian-Vehicle-API-Client/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        # Add API key if provided - Norwegian Vehicle API uses SVV-Authorization header
        if self.api_key:
            self.session.headers.update({"SVV-Authorization": f"Apikey {self.api_key}"})

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _make_request(
        self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            VehicleAPIError: If the request fails
        """
        url = urljoin(self.base_url, endpoint)

        try:
            self.logger.debug(f"Making {method} request to {url} with params: {params}")
            self.logger.debug(f"Headers: {dict(self.session.headers)}")

            response = self.session.request(
                method=method, url=url, params=params, timeout=self.timeout
            )

            # Log response details
            self.logger.debug(f"Response status: {response.status_code}")
            self.logger.debug(f"Response headers: {dict(response.headers)}")

            # Handle different status codes
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as e:
                    raise VehicleAPIError(
                        f"Invalid JSON response: {str(e)}",
                        status_code=response.status_code,
                        details=response.text[:500],
                    )

            elif response.status_code == 400:
                error_msg = "Bad request - invalid parameters"
                try:
                    error_data = response.json()
                    if "feilmelding" in error_data:
                        error_msg = error_data["feilmelding"]
                except ValueError:
                    pass
                raise VehicleAPIError(error_msg, status_code=400)

            elif response.status_code == 401:
                raise VehicleAPIError(
                    "Unauthorized - Invalid or missing API key",
                    status_code=401,
                    details="Check your VEHICLE_API_KEY environment variable",
                )

            elif response.status_code == 403:
                error_msg = "Forbidden - Access denied"
                try:
                    error_data = response.json()
                    if "feilmelding" in error_data:
                        error_msg = error_data["feilmelding"]
                    elif "message" in error_data:
                        error_msg = error_data["message"]
                except ValueError:
                    pass
                raise VehicleAPIError(
                    error_msg,
                    status_code=403,
                    details="Your API key may not have permission to access this resource, or the service requires special authorization",
                )

            elif response.status_code == 422:
                # Quota exceeded
                retry_after = response.headers.get("Retry-After", "unknown")
                raise VehicleAPIError(
                    f"Quota exceeded. Retry after: {retry_after}",
                    status_code=422,
                    details="Try again after midnight (Norwegian time)",
                )

            elif response.status_code == 429:
                # Rate limited
                retry_after = response.headers.get("Retry-After", "60")
                raise VehicleAPIError(
                    f"Rate limited. Retry after {retry_after} seconds", status_code=429
                )

            elif response.status_code >= 500:
                raise VehicleAPIError(
                    f"Server error: {response.status_code}",
                    status_code=response.status_code,
                    details=response.text[:500],
                )

            else:
                raise VehicleAPIError(
                    f"Unexpected status code: {response.status_code}",
                    status_code=response.status_code,
                    details=response.text[:500],
                )

        except requests.exceptions.Timeout:
            raise VehicleAPIError(f"Request timeout after {self.timeout} seconds")

        except requests.exceptions.ConnectionError as e:
            raise VehicleAPIError(f"Connection error: {str(e)}")

        except requests.exceptions.RequestException as e:
            raise VehicleAPIError(f"Request failed: {str(e)}")

    def get_vehicle_data(
        self, understellsnummer: Optional[str] = None, kjennemerke: Optional[str] = None
    ) -> KjoretoydataResponse:
        """
        Get vehicle data by chassis number or license plate

        Args:
            understellsnummer: Vehicle chassis number (VIN)
            kjennemerke: License plate number

        Returns:
            KjoretoydataResponse with vehicle data

        Raises:
            VehicleAPIError: If the request fails or parameters are invalid
        """
        if not understellsnummer and not kjennemerke:
            raise VehicleAPIError(
                "Either understellsnummer or kjennemerke must be provided"
            )

        if understellsnummer and kjennemerke:
            raise VehicleAPIError(
                "Only one of understellsnummer or kjennemerke should be provided"
            )

        params = {}
        if understellsnummer:
            params["understellsnummer"] = understellsnummer.strip()
        if kjennemerke:
            params["kjennemerke"] = kjennemerke.strip().upper()

        endpoint = "/enkeltoppslag/kjoretoydata"

        try:
            response_data = self._make_request("GET", endpoint, params)
            return KjoretoydataResponse(**response_data)

        except Exception as e:
            if isinstance(e, VehicleAPIError):
                raise
            raise VehicleAPIError(f"Failed to parse response: {str(e)}")

    def debug_config(self) -> Dict[str, Any]:
        """
        Debug method to show current configuration

        Returns:
            Dictionary with configuration info (API key masked for security)
        """
        config = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
            "api_key_preview": f"{self.api_key[:8]}...{self.api_key[-4:]}"
            if self.api_key
            else None,
            "headers": dict(self.session.headers),
        }

        # Mask the SVV-Authorization header if present
        if "SVV-Authorization" in config["headers"]:
            auth_header = config["headers"]["SVV-Authorization"]
            if len(auth_header) > 20:
                config["headers"]["SVV-Authorization"] = (
                    f"{auth_header[:15]}...{auth_header[-4:]}"
                )

        return config

    def health_check(self) -> bool:
        """
        Check if the API is accessible

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try a simple request to check connectivity
            response = self.session.get(self.base_url, timeout=self.timeout)
            return response.status_code < 500
        except Exception:
            return False

    def close(self):
        """Close the HTTP session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
