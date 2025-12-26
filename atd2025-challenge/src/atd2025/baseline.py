"""
This is the baseline algorithm given for the 2019 ATD Challenge. It was designed for
a much smaller dataset than we're using in the 2025 challenge, so it is slow.
"""

from __future__ import annotations

import math
from typing import Any

from tqdm import tqdm

from .tracks import LabelledPoint, LabelledTrack, Point, Track

__all__ = ["baseline"]

VERY_POSITIVE = float("inf")
VERY_NEGATIVE = float("-inf")


def m_str(t1: float, t2: float, v1: float, v2: float, dist: float) -> float:
    """Calculate the slope m such that:
    1. The segment between (t1, v1) and (tm, vm) has slope m
    2. The segment between (tm, vm) and (t2, v2) has slope -m
    3. The distance travelled across the internval (t1, t2) is 'dist'.

    See scripts/derive_m_star.py for a derivation.
    """
    # Check for special cases
    delta_t = t2 - t1
    if t1 == t2:
        return 0 if math.isclose(dist, 0) else VERY_NEGATIVE  # No solution when t1 equals t2
    if v1 == v2:
        return (dist - delta_t * v1) / (delta_t / 2) ** 2

    # Define components of the quadratic formula derived in scripts/derive_ms_star.
    a = delta_t**2
    b = 2 * dist - delta_t * (v1 + v2)
    c = 2 * dist**2 - 2 * dist * delta_t * (v1 + v2) + delta_t**2 * (v1**2 + v2**2)

    # Check if the term within the radical is significantly negative
    if c < -0.001:
        return VERY_NEGATIVE  # No real solutions

    # If it's only slightly negative, ignore the radical
    # Note: This appears to be an approach for dealing with minor floating point issues.
    # There are probably better ways of doing it.
    if c < 0:
        # Assume c is actually 0 (modulo floating point issues).
        # One solution
        m_star1 = b / a
        m_star2 = m_star1
    else:
        # Two real solutions
        m_star1 = (b + math.sqrt(2 * c)) / a
        m_star2 = (b - math.sqrt(2 * c)) / a

    # Note: This is the "true" way of doing it, but not robust to floating point
    # Errors. The difference is probably inconsequential even for very large datasets.
    # if c < 0:
    #     return VERY_NEGATIVE  # No real solutions
    # m_star1 = (b + math.sqrt(2 * c)) / a
    # m_star2 = (b - math.sqrt(2 * c)) / a

    # If either m1* or m2* perfectly matches slope, return it.
    slope = (v2 - v1) / delta_t
    if math.isclose(m_star1, slope):
        return m_star1
    if math.isclose(m_star2, slope):
        return m_star2

    # Otherwise, get possible t*
    tstar1 = t1 if math.isclose(m_star1, -slope) else t_str(m_star1, t1, t2, v1, v2)
    tstar2 = t1 if math.isclose(m_star2, -slope) else t_str(m_star2, t1, t2, v1, v2)

    # Choose best solution
    if (t1 <= tstar1 <= t2) and (t1 <= tstar2 <= t2):
        v_err1 = abs((m_star1 - slope) * (tstar1 - t1))
        v_err2 = abs((m_star2 - slope) * (tstar2 - t1))
        return m_star1 if v_err1 < v_err2 else m_star2
    elif t1 <= tstar1 <= t2:
        return m_star1
    elif t1 <= tstar2 <= t2:
        return m_star2
    else:
        return VERY_NEGATIVE  # No solutions


def t_str(m_star: float, t1: float, t2: float, v1: float, v2: float) -> float:
    if m_star == VERY_NEGATIVE:
        return VERY_NEGATIVE
    return (
        t2 - ((v1 - v2) + m_star * (t2 - t1)) / (2 * m_star) if not math.isclose(m_star, 0) else t2
    )


def v_str(m_star: float, t_star: float, t1: float, v1: float) -> float:
    return VERY_NEGATIVE if t_star == VERY_NEGATIVE else v1 + m_star * (t_star - t1)


def v_str_err(m_star: float, t_star: float, t1: float, t2: float, v1: float, v2: float) -> float:
    if t1 == t2:
        return 0.0
    if t_star == VERY_NEGATIVE:
        return VERY_NEGATIVE
    if t_star == t1:
        return VERY_NEGATIVE

    slope = (v2 - v1) / (t2 - t1)
    return abs((m_star - slope) * (t_star - t1))


def interp(x: float, x_min: float, x_max: float, map_min: float, map_max: float) -> float:
    if x_max == x_min:
        return map_min
    if x <= x_min:
        return map_min
    if x >= x_max:
        return map_max
    return map_min + (map_max - map_min) * (x - x_min) / (x_max - x_min)


def baseline(
    points: list[Point[Any]],
    alpha_v: float = 1,
    alpha_dh: float = 1,
    alpha_d2h: float = 1,
    max_v_ms: float = 40,
) -> list[LabelledPoint]:
    """A baseline algorithm for associating points to tracks."""
    points.sort()

    num_ntl_tracks = 0
    ntl_tracks: list[LabelledTrack] = []

    for pt in tqdm(points, "points", total=len(points)):
        if num_ntl_tracks == 0:
            # Assign the first point to the first track.
            num_ntl_tracks += 1
            pt.track_id = 1
            pt.track_idx = 1
            ntl_tracks.append(Track([pt]))
            continue

        best_score = VERY_POSITIVE
        best_track = Track([pt])
        for ntl_track in ntl_tracks:
            last_pt = ntl_track[-1]

            if pt.seconds == last_pt.seconds:  # dt = 0, don't calculate t_dh either
                t_dh = 0.0
            else:
                t_dh = abs(pt.course - last_pt.course)
                t_dh = min(t_dh, 360 - t_dh)
                t_dh = t_dh / (pt.seconds - last_pt.seconds)

            if len(ntl_track) == 1:  # no calculation of d2h
                t_d2h = 0.0
            else:
                second_last_pt = ntl_track[len(ntl_track) - 2]
                if last_pt.seconds == second_last_pt.seconds:
                    t_d2h = 0.0
                else:
                    t_dh2 = abs(last_pt.course - second_last_pt.course)
                    t_dh2 = min(t_dh2, 360 - t_dh2)
                    t_dh2 = t_dh2 / (last_pt.seconds - second_last_pt.seconds)
                    t_d2h = abs((t_dh2 - t_dh) / (last_pt.seconds - second_last_pt.seconds))

            t1 = last_pt.seconds
            t2 = pt.seconds

            v1 = last_pt.speed_ms
            v2 = pt.speed_ms

            dist = pt.rdist_m(last_pt)

            m_star = m_str(t1, t2, v1, v2, dist)
            t_star = t_str(m_star, t1, t2, v1, v2)
            v_star = v_str(m_star, t_star, t1, v1)

            if v_star != VERY_NEGATIVE and (abs(v_star) <= max_v_ms):
                v_star_err = v_str_err(m_star, t_star, t1, t2, v1, v2)
                t_score = alpha_v * v_star_err + alpha_dh * t_dh + alpha_d2h * t_d2h

                if t_score < best_score:
                    best_score = t_score
                    best_track = ntl_track

        if best_score == VERY_POSITIVE:
            num_ntl_tracks += 1
            pt.track_id = num_ntl_tracks
            pt.track_idx = 1
            best_track[0].track_id = num_ntl_tracks
            best_track[0].track_idx = 1
            ntl_tracks.append(best_track)
        else:
            for ntl_track in ntl_tracks:
                if ntl_track == best_track:
                    pt.track_id = best_track[0].track_id
                    pt.track_idx = len(best_track) + 1
                    ntl_track.append(pt)

    labeled_points = [pt for ntl_track in ntl_tracks for pt in ntl_track]
    labeled_points.sort(key=lambda p: p.point_id)

    return labeled_points
