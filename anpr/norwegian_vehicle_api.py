"""
Norwegian Vehicle Data API Wrapper
High-level interface for looking up vehicle information
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .vehicle_api_client import NorwegianVehicleAPIClient, VehicleAPIError
from .vehicle_api_models import KjoretoydataResponse, EnkeltOppslagKjoretoydata


@dataclass
class VehicleInfo:
    """Simplified vehicle information for easy access"""

    kuid: Optional[str] = None
    understellsnummer: Optional[str] = None
    kjennemerke: Optional[str] = None
    merke: Optional[str] = None
    modell: Optional[str] = None
    arsmodell: Optional[str] = None
    egenvekt: Optional[int] = None
    totalvekt: Optional[int] = None
    drivstoff: Optional[str] = None
    eier_navn: Optional[str] = None
    eier_adresse: Optional[str] = None
    registrert_forstgang: Optional[str] = None
    kontrollfrist: Optional[str] = None
    feilmelding: Optional[str] = None


class NorwegianVehicleAPI:
    """
    High-level API wrapper for Norwegian vehicle data lookups
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        enable_logging: bool = False,
    ):
        """
        Initialize the Vehicle API wrapper

        Args:
            api_key: API key for authentication (can also be set via VEHICLE_API_KEY env var)
            base_url: Base URL for the API (defaults to official URL)
            timeout: Request timeout in seconds
            enable_logging: Enable debug logging
        """
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.DEBUG)

        # Get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("VEHICLE_API_KEY")

        # Use default URL if not provided
        if not base_url:
            base_url = "https://akfell-datautlevering.atlas.vegvesen.no"

        # Initialize the HTTP client
        self.client = NorwegianVehicleAPIClient(
            base_url=base_url, timeout=timeout, api_key=api_key
        )

        self.logger = logging.getLogger(__name__)

    def lookup_by_license_plate(self, license_plate: str) -> VehicleInfo:
        """
        Look up vehicle information by license plate

        Args:
            license_plate: License plate number (e.g., "AB12345")

        Returns:
            VehicleInfo object with vehicle details

        Raises:
            VehicleAPIError: If the lookup fails
        """
        if not license_plate or not license_plate.strip():
            raise VehicleAPIError("License plate cannot be empty")

        try:
            response = self.client.get_vehicle_data(kjennemerke=license_plate.strip())
            return self._parse_response(response)

        except Exception as e:
            self.logger.error(
                f"Failed to lookup vehicle by license plate {license_plate}: {str(e)}"
            )
            raise

    def lookup_by_vin(self, vin: str) -> VehicleInfo:
        """
        Look up vehicle information by VIN (chassis number)

        Args:
            vin: Vehicle Identification Number

        Returns:
            VehicleInfo object with vehicle details

        Raises:
            VehicleAPIError: If the lookup fails
        """
        if not vin or not vin.strip():
            raise VehicleAPIError("VIN cannot be empty")

        try:
            response = self.client.get_vehicle_data(understellsnummer=vin.strip())
            return self._parse_response(response)

        except Exception as e:
            self.logger.error(f"Failed to lookup vehicle by VIN {vin}: {str(e)}")
            raise

    def _parse_response(self, response: KjoretoydataResponse) -> VehicleInfo:
        """
        Parse API response into simplified VehicleInfo object

        Args:
            response: Raw API response

        Returns:
            VehicleInfo object
        """
        vehicle_info = VehicleInfo()

        # Handle error messages
        if response.feilmelding:
            vehicle_info.feilmelding = response.feilmelding
            return vehicle_info

        # Extract vehicle data if available
        if response.kjoretoydataListe and len(response.kjoretoydataListe) > 0:
            vehicle_data = response.kjoretoydataListe[0]
            self._extract_vehicle_details(vehicle_data, vehicle_info)
        else:
            vehicle_info.feilmelding = "No vehicle data found"

        return vehicle_info

    def _extract_vehicle_details(
        self, vehicle_data: EnkeltOppslagKjoretoydata, vehicle_info: VehicleInfo
    ):
        """
        Extract details from vehicle data into VehicleInfo object

        Args:
            vehicle_data: Raw vehicle data from API
            vehicle_info: VehicleInfo object to populate
        """
        # Basic identity
        if vehicle_data.kjoretoyId:
            vehicle_info.kuid = vehicle_data.kjoretoyId.kuid
            vehicle_info.understellsnummer = vehicle_data.kjoretoyId.understellsnummer
            vehicle_info.kjennemerke = vehicle_data.kjoretoyId.kjennemerke

        # License plate information
        if vehicle_data.kjennemerke and len(vehicle_data.kjennemerke) > 0:
            # Get the most recent license plate
            current_plate = vehicle_data.kjennemerke[0]
            if current_plate.kjennemerke:
                vehicle_info.kjennemerke = current_plate.kjennemerke

        # Technical data
        if vehicle_data.godkjenning and vehicle_data.godkjenning.tekniskGodkjenning:
            # Note: Technical data extraction can be added here as needed
            pass

        # Vehicle brand and model
        # Note: This would require parsing the technical data
        # For now, setting placeholder extraction

        # Registration information
        if vehicle_data.forstegangsregistrering:
            if vehicle_data.forstegangsregistrering.registrertForstegangNorgeDato:
                vehicle_info.registrert_forstgang = str(
                    vehicle_data.forstegangsregistrering.registrertForstegangNorgeDato
                )

        # Inspection information
        if vehicle_data.periodiskKjoretoyKontroll:
            if vehicle_data.periodiskKjoretoyKontroll.kontrollfrist:
                vehicle_info.kontrollfrist = str(
                    vehicle_data.periodiskKjoretoyKontroll.kontrollfrist
                )

    def debug_config(self) -> Dict[str, Any]:
        """
        Show current API configuration for debugging

        Returns:
            Configuration information
        """
        return self.client.debug_config()

    def test_connection(self) -> bool:
        """
        Test if the API is accessible

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            return self.client.health_check()
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_raw_data(
        self, license_plate: Optional[str] = None, vin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get raw API response without parsing

        Args:
            license_plate: License plate number
            vin: Vehicle Identification Number

        Returns:
            Raw API response as dictionary
        """
        if license_plate:
            response = self.client.get_vehicle_data(kjennemerke=license_plate)
        elif vin:
            response = self.client.get_vehicle_data(understellsnummer=vin)
        else:
            raise VehicleAPIError("Either license_plate or vin must be provided")

        return response.dict()

    def close(self):
        """Close the underlying HTTP client"""
        self.client.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience functions for quick lookups
def lookup_vehicle_by_plate(
    license_plate: str, api_key: Optional[str] = None
) -> VehicleInfo:
    """
    Quick lookup by license plate

    Args:
        license_plate: License plate number
        api_key: Optional API key

    Returns:
        VehicleInfo object
    """
    with NorwegianVehicleAPI(api_key=api_key) as api:
        return api.lookup_by_license_plate(license_plate)


def lookup_vehicle_by_vin(vin: str, api_key: Optional[str] = None) -> VehicleInfo:
    """
    Quick lookup by VIN

    Args:
        vin: Vehicle Identification Number
        api_key: Optional API key

    Returns:
        VehicleInfo object
    """
    with NorwegianVehicleAPI(api_key=api_key) as api:
        return api.lookup_by_vin(vin)
