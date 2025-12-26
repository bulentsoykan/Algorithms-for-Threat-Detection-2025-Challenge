import os
from pathlib import Path

import pandas as pd
import pytest

import atd2025.aisvis.show_tracks as st
from atd2025 import read_points_as_df


class DataFiles:
    no_id_posits = "no_track_id_test_data.csv"
    tracks = "with_track_id_test_data.csv"


SHARED_DATADIR = Path(__file__).parent / "data"


@pytest.mark.parametrize("filename", [DataFiles.no_id_posits, DataFiles.tracks])
def test_file_exists(filename: str) -> None:
    assert os.path.exists(SHARED_DATADIR / filename)


@pytest.fixture
def with_id_ais_df() -> pd.DataFrame:
    return read_points_as_df(SHARED_DATADIR / DataFiles.tracks)


@pytest.fixture
def no_id_ais_df() -> pd.DataFrame:
    return read_points_as_df(SHARED_DATADIR / DataFiles.no_id_posits)


@pytest.mark.parametrize(
    "show_lines, show_velocity, answer",
    [(True, True, 172), (True, False, 86), (False, True, 172), (False, False, 86)],
)
def test_plots_with_id(
    with_id_ais_df: pd.DataFrame, show_lines: bool, show_velocity: bool, answer: int
) -> None:
    fig = st.make_line_plot(
        with_id_ais_df,
        show_lines=show_lines,
        show_velocity=show_velocity,
        return_n=False,
    )
    assert len(fig.data) == answer


@pytest.mark.filterwarnings("ignore:Data doesn't include ship tid")
@pytest.mark.parametrize(
    "show_lines, show_velocity, answer",
    [(True, True, 2), (True, False, 1), (False, True, 2), (False, False, 1)],
)
def test_plots_no_id(
    no_id_ais_df: pd.DataFrame, show_lines: bool, show_velocity: bool, answer: int
) -> None:
    fig = st.make_line_plot(
        no_id_ais_df,
        show_lines=show_lines,
        show_velocity=show_velocity,
        return_n=False,
    )
    assert len(fig.data) == answer


def test_plot_return_tuple(with_id_ais_df: pd.DataFrame) -> None:
    fig, n_tracks = st.make_line_plot(
        with_id_ais_df, show_lines=True, show_velocity=False, return_n=True
    )
    assert len(fig.data) == n_tracks
