"""Unit conversions for fixed 15-minute market time units."""

from collections.abc import Mapping

MTU_DURATION_HOURS = 0.25


def mwh_to_mw(value):
    """Convert a 15-minute energy quantity in MWh to average power in MW."""
    return value / MTU_DURATION_HOURS


def mw_to_mwh(value):
    """Convert a 15-minute average power quantity in MW to energy in MWh."""
    if value is None:
        return None

    if isinstance(value, Mapping):
        return {
            key: mw_to_mwh(item)
            for key, item in value.items()
        }

    return value * MTU_DURATION_HOURS