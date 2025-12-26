"""Point and Track classes for object tracking.

This module provides the core data structures for representing track data,
including individual points with temporal and spatial information, and tracks
as sequences of points.

Classes:
    Point: Base class for point data.
    Track: Sequence of points forming a track.

Attributes:
    LabelledPoint: Point with a track_id and track_idx.
    LabelledTrack: Track consisting of LabelledPoints.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import pandas as pd

from .constants import EARTH_RAD_M, KNOTS_TO_MS

__all__ = ["Point", "LabelledPoint", "Track", "LabelledTrack"]

T = TypeVar("T")


@dataclass
class Point(Generic[T]):
    """Single point in a track with temporal and spatial information.

    Attributes:
        point_id: Unique identifier for the point.
        time: Timestamp of the point.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        course: Heading/course in degrees clockwise from North.
        speed: Speed in knots.
        track_id: Optional track identifier.
        track_idx: Optional index within track.
    """

    point_id: int
    time: datetime.datetime
    lat: float
    lon: float
    course: float
    speed: float
    track_id: T = None  # type: ignore[assignment]
    track_idx: T = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.time, pd.Timestamp):
            self.time = self.time.to_pydatetime()

        if self.track_id is not None and math.isnan(self.track_id):  # type: ignore[arg-type]
            self.track_id = None  # type: ignore[assignment]

        if self.track_idx is not None and math.isnan(self.track_idx):  # type: ignore[arg-type]
            self.track_idx = None  # type: ignore[assignment]

        self.seconds = (self.time - datetime.datetime(2024, 1, 1, 0, 0, 0)).total_seconds()
        self.speed_ms = KNOTS_TO_MS * self.speed
        self.hours = self.seconds / 3600

    def __lt__(self, other: Point[Any]) -> bool:
        if isinstance(other, Point):
            return self.seconds < other.seconds
        else:
            return NotImplemented

    def rdist_m(self, other: Point[Any]) -> float:
        """Calculate the rhumb line (constant heading) distance between the object and another
        object (in m).

        Source: https://www.movable-type.co.uk/scripts/latlong.html"""
        r = float(EARTH_RAD_M)
        atan1 = math.atan(1.0)
        p1_lat_rad = self.lat * (atan1 / 45.0)
        p1_lon_rad = self.lon * (atan1 / 45.0)
        p2_lat_rad = other.lat * (atan1 / 45.0)
        p2_lon_rad = other.lon * (atan1 / 45.0)
        dphi = math.log(math.tan(p2_lat_rad / 2 + atan1) / math.tan(p1_lat_rad / 2 + atan1))
        dlat = p2_lat_rad - p1_lat_rad
        dlon = p2_lon_rad - p1_lon_rad

        if abs(dlon) > 4 * atan1:
            dlon = -(8 * atan1 - dlon) if dlon > 0 else 8 * atan1 + dlon

        q = math.cos(p1_lat_rad) if abs(dphi) <= 10 ** (-12) else dlat / dphi
        return math.sqrt(dlat**2 + q**2 * dlon**2) * r

    def almost_equal(self, other: Any) -> bool:
        if isinstance(other, Point):
            return (
                self.point_id == other.point_id
                and self.time == other.time
                and math.isclose(self.lat, other.lat)
                and math.isclose(self.lon, other.lon)
                and math.isclose(self.course, other.course)
                and math.isclose(self.speed, other.speed)
                and self.track_id == other.track_id
                and self.track_idx == other.track_idx
            )
        return False

    def __eq__(self, other: Any) -> bool:
        return self.almost_equal(other)


LabelledPoint = Point[int]


class Track(list[Point[T]]):
    """Sequence of points forming a track.

    Inherits from list and adds methods for track analysis and metrics calculation.

    Attributes:
        track_id: Identifier of the track (from first point).
        duration: Total duration of track in seconds.
        distance: Total distance covered by track in meters.
    """

    @property
    def track_id(self: LabelledTrack) -> int:
        if not self:
            raise IndexError("Track ID of empty Track is not defined.")
        assert self[0].track_id is not None
        return self[0].track_id

    @property
    def duration(self) -> float:
        """This function returns the total track time (in s)"""
        return self[-1].seconds - self[0].seconds

    @property
    def distance(self) -> float:
        """This function returns the total track (rhumb) distance (in m)"""
        if len(self) <= 1:
            return 0

        return sum(last.rdist_m(this) for last, this in zip(self[:-1], self[1:]))

    @property
    def _point_ids(self) -> set[int]:
        """Lazily return point ids as a set.

        This benefits the `atd2025.metrics.completeness.

        Note:
            This caches the set on point_ids upon first call to the property. Because Tracks are
            mutable, modifying the Track after calling this will method will result using a stale
            value. Therefore, do not use this property until the list of points or their point IDs
            will not be modified further.
        """
        try:
            pt_ids: set[int] = self._point_ids_cache  # type: ignore[has-type]
        except AttributeError:
            pt_ids = {pt.point_id for pt in self}
            self._point_ids_cache = pt_ids
        return pt_ids


LabelledTrack = Track[int]
