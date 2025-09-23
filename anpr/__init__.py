"""ANPR (Automatic Number Plate Recognition) Core Module"""

from .license_plate_reader import read_license_plate, check_both_norwegian_services, upsert_loan_status
from .loan_db_utils import LoanStatusDB
from .norwegian_vehicle_api import NorwegianVehicleAPI

__version__ = "1.0.0"
__all__ = [
    "read_license_plate",
    "check_both_norwegian_services", 
    "upsert_loan_status",
    "LoanStatusDB",
    "NorwegianVehicleAPI"
]