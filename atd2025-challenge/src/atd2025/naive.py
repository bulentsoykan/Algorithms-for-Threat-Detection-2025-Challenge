"""Naive tracking assignment algorithms.

This module provides simple algorithms for assigning points to tracks,
useful for testing and baseline comparisons.

Functions:
    assign_to_one_track: Assign all points to a single track.
    assign_to_unique_track: Assign each point to its own track.
    assign_randomly: Randomly assign points to tracks.
"""

from __future__ import annotations

import random
from typing import Any

from tqdm import tqdm

from .tracks import LabelledPoint, Point, Track

__all__ = [
    "assign_to_one_track",
    "assign_to_unique_track",
    "assign_randomly",
    "assign_most_similar_speed_in_foc",
]


def assign_to_one_track(points: list[Point[Any]]) -> list[LabelledPoint]:
    """Assign all points to the same track.

    Args:
        points: List of points to assign.

    Returns:
        List of points all assigned to the same track.
    """
    track_idx = 0
    for pt in points:
        pt.track_id = 0
        pt.track_idx = track_idx
        track_idx += 1
    return points


def assign_to_unique_track(points: list[Point[Any]]) -> list[LabelledPoint]:
    """Assign each point to its own unique track.

    Args:
        points: List of points to assign.

    Returns:
        List of points where each point is assigned a unique track_id.
    """
    track_id = 0
    for pt in points:
        pt.track_id = track_id
        pt.track_idx = 0
        track_id += 1
    return points


def assign_randomly(
    points: list[Point[Any]], probability_new_track: float = 0.05
) -> list[LabelledPoint]:
    """Randomly assign points to tracks.

    Args:
        points: List of points to assign.
        probability_new_track: Probability of starting a new track at each point.

    Returns:
        List of points with randomly assigned track_ids.
    """
    track_id = 0
    last_track_id: list[int] = []
    for i, pt in enumerate(points):
        if (random.random() < probability_new_track) or i == 0:
            pt.track_id = track_id
            pt.track_idx = 0
            last_track_id.append(0)
            track_id += 1
        else:
            pt.track_id = random.randint(0, track_id - 1)
            last_track_id[pt.track_id] += 1
            pt.track_idx = last_track_id[pt.track_id]
    return points


def assign_most_similar_speed_in_foc(
    points: list[Point[Any]], v_max: float = 40
) -> list[LabelledPoint]:
    """Assign point within furthest-on-circle that has most similar velocity.

    Note:
        *Furthest-no-circle* (FOC) is the circular region around a vessel's current position
        that the vessel could feasibly reach if it travelled at full speed in an arbitrary
        direction for some fixed period of time.

    Args:
        points: List of points to assign.
        v_max: The maximum speed the vessel can travel.

    Returns:
        List of labelled points.
    """
    points.sort()
    num_ntl_tracks = 0
    ntl_tracks: list[Track[Any]] = []
    for pt in tqdm(points, "points", total=len(points)):
        if num_ntl_tracks == 0:
            num_ntl_tracks += 1
            pt.track_id = num_ntl_tracks
            pt.track_idx = 1
            ntl_tracks.append(Track([pt]))
            continue

        best_track: Track[Any] | None = None
        best_v = float("inf")

        for ntl_track in ntl_tracks:
            last_pt = ntl_track[-1]

            # Calculate the required velocity to connect these points
            delta_time = pt.seconds - last_pt.seconds
            if delta_time > 0:
                distance = pt.rdist_m(last_pt)
                required_velocity = distance / delta_time
                if required_velocity <= v_max:
                    delta_v = abs(required_velocity - last_pt.speed_ms)

                    if delta_v < best_v:
                        best_v = delta_v
                        best_track = ntl_track
        if best_track:
            pt.track_id = best_track[0].track_id
            pt.track_idx = len(best_track) + 1
            best_track.append(pt)
        else:
            num_ntl_tracks += 1
            pt.track_id = num_ntl_tracks
            pt.track_idx = 1
            ntl_tracks.append(Track([pt]))

    return [pt for ntl_track in ntl_tracks for pt in ntl_track]
