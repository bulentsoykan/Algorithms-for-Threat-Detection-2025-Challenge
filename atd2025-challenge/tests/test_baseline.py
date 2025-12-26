import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from atd2025.baseline import VERY_NEGATIVE, m_str, t_str

positive_float = st.floats(
    min_value=1, max_value=9999, allow_infinity=False, allow_nan=False, allow_subnormal=False
)


def is_between(t: float, t1: float, t2: float) -> bool:
    return t1 <= t <= t2


def m_str_old(t1: float, t2: float, v1: float, v2: float, dist: float) -> float:
    c = 2 * dist**2 + 2 * dist * (t1 - t2) * (v1 + v2) + (t1 - t2) ** 2 * (v1**2 + v2**2)
    if math.isclose(dist, 0) and math.isclose(t1, t2):
        return 0
    elif t1 == t2:
        return VERY_NEGATIVE
    elif v1 == v2:
        return (dist - (t2 - t1) * v1) / ((t2 - t1) / 2) ** 2
    elif c < -0.001:
        return VERY_NEGATIVE
    else:
        denom = (t1 - t2) ** 2
        base = 2 * dist + (t1 - t2) * (v1 + v2)

        if c < 0:
            m_star_1 = base / denom
            m_star_2 = m_star_1
        else:
            # Multiplying inside the radical is more numerically stable
            # than the original math.sqrt(2) * math.sqrt(c) approach.
            m_star_1 = (base + math.sqrt(2 * c)) / denom
            m_star_2 = (base - math.sqrt(2 * c)) / denom
            # m_star_1 = (base + math.sqrt(2) * math.sqrt(c)) / denom
            # m_star_2 = (base - math.sqrt(2) * math.sqrt(c)) / denom

        slope = (v2 - v1) / (t2 - t1)
        if math.isclose(m_star_1, slope):
            return m_star_1
        elif math.isclose(m_star_1, -slope):
            t_star_1 = t1
            v_err_1 = 0.0
        else:
            t_star_1 = t_str(m_star_1, t1, t2, v1, v2)
            v_err_1 = abs(m_star_1 - (v2 - v1) / (t2 - t1)) * (t_star_1 - t1)

        if math.isclose(m_star_2, slope):
            return m_star_2
        elif math.isclose(m_star_2, -slope):
            t_star_2 = t1
            v_err_2 = 0.0
        else:
            t_star_2 = t_str(m_star_2, t1, t2, v1, v2)
            v_err_2 = abs(m_star_2 - (v2 - v1) / (t2 - t1)) * (t_star_2 - t1)

        if is_between(t_star_1, t1, t2) and is_between(t_star_2, t1, t2):
            if v_err_1 < v_err_2:
                return m_star_1
            else:
                return m_star_2
        elif is_between(t_star_1, t1, t2):
            return m_star_1
        elif is_between(t_star_2, t1, t2):
            return m_star_2
        else:
            return VERY_NEGATIVE


@settings(max_examples=1_000)
@given(
    t1=positive_float,
    t2=positive_float,
    v1=positive_float,
    v2=positive_float,
    dist=positive_float,
)
def test_mstr_refactor(t1: float, t2: float, v1: float, v2: float, dist: float) -> None:
    try:
        expected = m_str_old(t1, t2, v1, v2, dist)
    except ZeroDivisionError:
        # If we previously had a division by zero, make sure we've fixed it.
        actual = m_str(t1, t2, v1, v2, dist)
        assert actual == VERY_NEGATIVE
        return
    actual = m_str(t1, t2, v1, v2, dist)
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6), (actual, expected)
