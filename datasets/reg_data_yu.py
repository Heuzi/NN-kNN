"""Legacy compatibility wrapper for the shared regression dataset loaders.

This module used to maintain a separate copy of a few regression dataset
loaders. The canonical implementations now live in `datasets.reg_data`.
"""

from datasets.reg_data import (
    Abalone,
    Bike_Sharing,
    Body_Fat,
    California_Housing,
    DATATYPES,
    Diabetes,
    Reg_data,
    Wine,
    standardize_tensor,
)

__all__ = [
    "Abalone",
    "Bike_Sharing",
    "Body_Fat",
    "California_Housing",
    "DATATYPES",
    "Diabetes",
    "Reg_data",
    "Wine",
    "standardize_tensor",
]
