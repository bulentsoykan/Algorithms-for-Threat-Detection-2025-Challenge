from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import pandas as pd


class AISColumns(str, Enum):
    tid = "track_id"
    time = "time"
    lat = "lat"
    lon = "lon"
    sog = "speed"
    cog = "course"
    pid = "point_id"


@dataclass
class Region:
    name: str
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]


RegionOptions: Dict[str, Region] = dict(
    PR=Region(name="Puerto Rico", lat_range=(15, 21), lon_range=(-70.5, -62.5)),
    EC=Region(name="Eastern US Seaboard", lat_range=(22, 45.5), lon_range=(-85, -62)),
    CT=Region(name="Central US", lat_range=(22, 45.5), lon_range=(-98, -85)),
    GC=Region(name="Gulf Coast", lat_range=(22, 31), lon_range=(-98, -81.5)),
    WC=Region(name="Western US Seaboard", lat_range=(25, 55.5), lon_range=(-130, -110)),
    HI=Region(name="Hawaii", lat_range=(14, 27), lon_range=(-166, -152)),
    SF=Region(name="South Florida", lat_range=(22, 29), lon_range=(-85, -75)),
    SC=Region(name="Southern US Seaboard", lat_range=(22, 31), lon_range=(-98, -75)),
)


def restrict_range(
    df: pd.DataFrame,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Return dataframe that only has tracks within the specified spatial area.

    Args:
        df: DataFrame of points.
        lat_range: Inclusive ounds for latitude (min, max) bounds (Optional).
        lon_range: Inclusive ounds for longitude (min, max) bounds (Optional).

    Returns:
        DataFrame containing only posits within the bounds specified.

    Example:
    ```py
        restrict_range(df, region="PR")
        # Returns DataFrame containing only posits near Puerto Rico.

    ```

    """
    assert_valid_ranges(lat_range=lat_range, lon_range=lon_range)
    if lon_range is not None:
        df = df[df[AISColumns.lon].between(*lon_range)]

    if lat_range is not None:
        df = df[df[AISColumns.lat].between(*lat_range)]
    return df


def assert_valid_ranges(
    lat_range: Optional[Tuple[float, float]], lon_range: Optional[Tuple[float, float]]
) -> None:
    """Assert ranges are valid longitude/latitude ranges.

    Args:
        lat_range: Inclusive ounds for latitude (min, max) bounds (Optional).
        lon_range: Inclusive ounds for longitude (min, max) bounds (Optional).

    Raises:
        ValueError: if min value is greater than maximum value for either range.
    """

    if lon_range is not None and lon_range[0] >= lon_range[1]:
        raise ValueError(f"Invalid longitude range: {lon_range[0]} >= {lon_range[1]}")

    if lat_range is not None and lat_range[0] >= lat_range[1]:
        raise ValueError(f"Invalid latitude range: {lat_range[0]} >= {lat_range[1]}")


def get_region(key: str) -> Region:
    """Get region by key, and raise helpful error message otherwise.

    Args:
        key: Region key.

    Returns:
        Region.

    Raises:
        ValueError: if key is not in RegionOptions.
    """
    try:
        return RegionOptions[key]
    except KeyError:
        raise ValueError(f"{key} is not a valid options.\n{show_valid_regions()}")


def show_valid_regions(include_bounds: bool = True) -> str:
    """Format RegionOptions as a human readble string.

    Args:
        include_bounds: Whether to include or exclude the geodetic range bounds.

    Returns
        Human readable summary of region options available.
    """
    options = (
        "\n\n".join(
            f"[{k}] {r.name}: lat={r.lat_range} lon={r.lon_range}" for k, r in RegionOptions.items()
        )
        if include_bounds
        else "\n\n".join(f"[{k}] {r.name}" for k, r in RegionOptions.items())
    )
    return f"Region options:\n\n{options}"
