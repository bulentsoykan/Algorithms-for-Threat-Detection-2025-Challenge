import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from atd2025 import data as ap
from atd2025.io import AISColumns

data_st = data_frames(
    columns=[
        column("track_id", dtype=int, elements=st.integers(min_value=0, max_value=3)),
        column(
            "time",
            dtype="datetime64[ns]",
            elements=st.datetimes(
                min_value=datetime(2023, 1, 1, 0, 0, 0), max_value=datetime(2023, 1, 1, 23, 59, 59)
            ),
        ),
        column("lat", dtype=float, elements=st.floats(min_value=22, max_value=31, width=16)),
        column("lon", dtype=float, elements=st.floats(min_value=-98, max_value=-75, width=16)),
        column("speed", dtype=float, elements=st.floats(0, 102.3)),
        column("course", dtype=float, elements=st.floats(0, 360)),
    ],
    index=range_indexes(min_size=30),
)


@pytest.fixture(scope="session")
def day() -> str:
    return "01/01/2023"


@pytest.fixture(scope="session")
def data_path() -> Path:
    return Path("./data")


@pytest.fixture(scope="session")
def out_path() -> Path:
    return Path("./out")


@pytest.fixture(scope="session")
def csv_path(day: str, data_path: Path) -> Path:
    return ap.query_ais(datetime.strptime(day, "%m/%d/%Y"), data_path)


@pytest.fixture(scope="session")
def ais_df(csv_path: Path) -> pd.DataFrame:
    return ap.ais_to_df(csv_path, None)


@pytest.fixture(scope="session")
def messy_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "track_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            "time": pd.date_range(start="1/1/2023", periods=10, freq="6min"),
            "lat": [90, -90, -89.91, 200, -100.2, 23.82, 23.33, 23.31, 23.34, 23.38],
            "lon": [180, -180, -179.6, 230.9, -360.9, -82.2, -82.26, -82.22, -82.23, -82.21],
            "speed": [0.5, 2.1, 0.2, 102.3, -1.2, 1, 1, 0, 0, 0],
            "course": [25, 75, 73.2, 360, -223, 12, 21, 55, 36.8, 63.5],
        }
    )
    df.index.name = "point_id"
    return df


### tests for data collection
def test_date_wrong_pattern() -> None:
    with pytest.raises(ValueError):
        ap.validate_date("1/2024")


def test_date_before_range() -> None:
    with pytest.raises(Exception):
        ap.validate_date("12/31/2014")


def test_date_after_range() -> None:
    with pytest.raises(Exception):
        ap.validate_date("4/1/2024")


@pytest.mark.slow
def test_existing_data_query(day: str, data_path: Path) -> None:
    file = ap.query_ais(datetime.strptime(day, "%m/%d/%Y"), data_path)
    assert os.path.exists(file)


@pytest.mark.slow
def test_new_data_query(data_path: Path) -> None:
    Path.unlink(data_path / "AIS_2023_02_02.csv", missing_ok=True)
    Path.unlink(data_path / "AIS_2023_02_02.zip", missing_ok=True)
    file = ap.query_ais(datetime.strptime("02/02/2023", "%m/%d/%Y"), data_path)
    assert os.path.exists(file)


### tests for regular data
@pytest.mark.slow
def test_ais_to_df_format_col(ais_df: pd.DataFrame) -> None:
    col_names = {
        AISColumns.tid,
        AISColumns.time,
        AISColumns.lat,
        AISColumns.lon,
        AISColumns.sog,
        AISColumns.cog,
    }
    assert set(ais_df) == col_names


@pytest.mark.slow
def test_ais_to_df_format_idx(ais_df: pd.DataFrame) -> None:
    assert ais_df.index.name == AISColumns.pid


@given(df=data_st)
@settings(max_examples=1)
@pytest.mark.slow
def test_df_to_csv(df: pd.DataFrame, csv_path: Path, out_path: Path) -> None:
    base_filename = "output_test.csv"
    truth_filename = "output_test_truth.csv"

    ap.df_to_csv(df.head(), out_path, "output_test.csv")

    assert os.path.exists(out_path / truth_filename)
    assert os.path.exists(out_path / base_filename)


### tests for data with added noise
@given(df=data_st)
@settings(max_examples=20)
def test_apply_error(df: pd.DataFrame) -> None:
    error_df = ap.apply_spatial_error(df)
    error_df = ap.apply_temporal_error(error_df)

    # temporal variation
    true_time = pd.to_datetime(df[AISColumns.time], format="%H:%M:%S")
    error_time = pd.to_datetime(error_df[AISColumns.time], format="%H:%M:%S")
    assert not (abs((true_time - error_time).dt.seconds).values == 0).all()  # type: ignore[union-attr]

    # spatial variation
    assert not (abs(df[AISColumns.lat] - error_df[AISColumns.lat]).values == 0).all()  # type: ignore[union-attr]
    assert not (abs(df[AISColumns.lon] - error_df[AISColumns.lon]).values == 0).all()  # type: ignore[union-attr]


def test_trim_stationary(messy_df: pd.DataFrame) -> None:
    assert len(messy_df) > len(ap.trim_stationary(messy_df))


def test_invalid_lon(messy_df: pd.DataFrame) -> None:
    messy_df = ap.trim_invalid_values(messy_df)
    assert not (messy_df[AISColumns.lon].values < -180).any()  # type: ignore[operator,union-attr]
    assert not (messy_df[AISColumns.lon].values > 180).any()  # type: ignore[operator,union-attr]


def test_invalid_lat(messy_df: pd.DataFrame) -> None:
    messy_df = ap.trim_invalid_values(messy_df)
    assert not (messy_df[AISColumns.lat].values < -90).any()  # type: ignore[operator,union-attr]
    assert not (messy_df[AISColumns.lat].values > 90).any()  # type: ignore[operator,union-attr]


def test_invalid_speed(messy_df: pd.DataFrame) -> None:
    messy_df = ap.trim_invalid_values(messy_df)
    assert not (
        messy_df[AISColumns.sog].values < 0  # type: ignore[operator,union-attr]
    ).any()  # negative values can be corrected by adding 102.4
    # 102.3 == not available
    assert not (messy_df[AISColumns.sog].values >= 102.3).any()  # type: ignore[operator,union-attr]


def test_invalid_course(messy_df: pd.DataFrame) -> None:
    messy_df = ap.trim_invalid_values(messy_df)
    assert not (
        messy_df[AISColumns.cog].values < 0  # type: ignore[operator,union-attr]
    ).any()  # negative values can be corrected by adding 409.6
    # 360.0 == not available
    assert not (messy_df[AISColumns.cog].values >= 360.0).any()  # type: ignore[operator,union-attr]


def test_within_region(messy_df: pd.DataFrame) -> None:
    messy_df = ap.restrict_range(messy_df, region="GC")

    # southeast US region
    assert messy_df[AISColumns.lon].between(-98, -75).any()
    assert messy_df[AISColumns.lat].between(22, 31).any()
