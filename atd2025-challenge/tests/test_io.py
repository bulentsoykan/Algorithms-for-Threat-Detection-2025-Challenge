from datetime import datetime
from typing import Any, List

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from atd2025 import io, tracks

int_st = st.integers(min_value=0, max_value=99_999_999)
float_st = st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False)
datetime_st = st.datetimes(
    min_value=datetime(2015, 1, 1), max_value=datetime(2024, 3, 31), allow_imaginary=False
)

point_data = [int_st, datetime_st, float_st, float_st, float_st, float_st]
pred_data = [int_st, datetime_st, float_st, float_st, float_st, float_st, int_st, int_st]


@given(points=st.lists(st.builds(tracks.Point, *point_data), min_size=1))
@settings(max_examples=100)
def test_df_conversion(points: List[tracks.Point[Any]]) -> None:
    df = io.to_pandas(points, all_attrs=False)
    points_from_df = io.to_points(df)
    df_from_points = io.to_pandas(points_from_df, all_attrs=False)

    assert points == points_from_df
    pd.testing.assert_frame_equal(df, df_from_points)


@given(points=st.lists(st.builds(tracks.Point, *point_data), min_size=1))
@settings(max_examples=50)
def test_csv_conversion(
    points: List[tracks.Point[Any]], tmpdir_factory: pytest.TempdirFactory
) -> None:
    csv_file = tmpdir_factory.mktemp("data").join("data.csv")
    io.points_to_csv(csv_file, points)
    points_from_csv = io.read_points(csv_file)

    csv_file_out = tmpdir_factory.mktemp("data").join("data_out.csv")
    io.points_to_csv(csv_file_out, points)
    df_csv = pd.read_csv(csv_file)
    df_csv_out = pd.read_csv(csv_file_out)

    assert points == points_from_csv
    pd.testing.assert_frame_equal(df_csv, df_csv_out)


@given(
    points_pred=st.lists(
        st.builds(tracks.Point, *pred_data),
        min_size=1,
        unique_by=(lambda x: x.point_id, lambda y: y.track_idx),
    )
)
@settings(max_examples=50)
def test_pred_conversion(
    points_pred: List[tracks.LabelledPoint], tmpdir_factory: pytest.TempdirFactory
) -> None:
    points = io.to_points(io.to_pandas(points_pred).drop(columns=["track_id", "track_idx"]))
    pred = io.to_pandas(points_pred)[["point_id", "track_id", "track_idx"]]

    pred_csv = tmpdir_factory.mktemp("data").join("data.csv")
    pred.to_csv(pred_csv, index=False)

    labelled_points = io.read_predictions(pred_csv, points)

    out_csv = tmpdir_factory.mktemp("data").join("out.csv")
    io.predictions_to_csv(out_csv, labelled_points)
    pred_from_csv = pd.read_csv(out_csv).sort_values("point_id").reset_index(drop=True)
    pred_only = (
        io.to_pandas(labelled_points)[["point_id", "track_id"]]
        .sort_values("point_id")
        .reset_index(drop=True)
    )

    assert points_pred.sort() == labelled_points.sort()
    pd.testing.assert_frame_equal(pred_only, pred_from_csv)
