"""File input/output utilities for track data processing.

This module provides functions for reading and writing point and track data from/to CSV files,
as well as converting between pandas DataFrames and Point objects.

Functions:
    read_points: Read point data from a CSV file.
    read_predictions: Read prediction data and merge with existing points.
    to_points: Convert a pandas DataFrame to a list of Point objects.
    to_pandas: Convert Point objects to a pandas DataFrame.
    points_to_csv: Write points to a CSV file.
    predictions_to_csv: Write track predictions to a CSV file.
"""

from dataclasses import asdict, fields
from os import PathLike
from typing import Any, Callable, Iterable, Union, cast

import pandas as pd

from .tracks import LabelledPoint, Point

__all__ = [
    "read_points",
    "read_predictions",
    "to_points",
    "to_pandas",
    "points_to_csv",
    "predictions_to_csv",
    "read_points_as_df",
]


class AISColumns:
    """Required column names for track/AIS .csv documents."""

    tid = "track_id"
    time = "time"
    lat = "lat"
    lon = "lon"
    sog = "speed"
    cog = "course"
    pid = "point_id"


StrPath = Union[str, PathLike[str]]


def read_points(file: StrPath) -> list[Point[Any]]:
    """Read point data from a CSV file.

    Args:
        file: str or PathLike to CSV file containing point data with 'time' column.

    Returns:
        List of Point objects created from the CSV data.
    """
    return to_points(pd.read_csv(file, parse_dates=[AISColumns.time]))


def read_points_as_df(file: StrPath) -> pd.DataFrame:
    """Read point data from a CSV file.

    Args:
        file: str or PathLike to CSV file containing point data with 'time' column.

    Returns:
        Pandas dataframe.
    """
    return pd.read_csv(file, parse_dates=[AISColumns.time])


def to_points(df: pd.DataFrame) -> list[Point[Any]]:
    """Convert a pandas DataFrame to a list of Point objects.

    Args:
        df: DataFrame containing point data with columns matching Point fields.

    Returns:
        List of Point objects created from DataFrame rows.
    """
    attrs = [field.name for field in fields(Point) if field.name in df.columns]
    return list(Point(*row) for row in df[attrs].values)


def read_predictions(file: StrPath, points: Iterable[Point[Any]]) -> list[LabelledPoint]:
    """Read prediction data and merge with existing points.

    Args:
        file: str or PathLike to CSV file containing predictions.
        points: Iterable of Point objects to merge with predictions.

    Returns:
        List of LabelledPoint objects with merged prediction data.
    """
    pred_df = pd.read_csv(file)
    point_df = to_pandas(points, all_attrs=False).drop(
        columns=[AISColumns.tid, "track_idx"], errors="ignore"
    )
    joined = pd.merge(point_df, pred_df, on=AISColumns.pid, validate="one_to_one", how="outer")
    return to_points(joined)


def to_pandas(points: Iterable[Point[Any]], all_attrs: bool = True) -> pd.DataFrame:
    """Convert Point objects to a pandas DataFrame.

    Args:
        points: Iterable of Point objects to convert.
        all_attrs: If True, include all attributes. If False, only include basic fields.

    Returns:
        DataFrame containing point data.
    """
    f = cast(Callable[[Point[Any]], dict[str, Any]], vars if all_attrs else asdict)
    return pd.DataFrame([f(point) for point in points])


def points_to_csv(outfile: StrPath, points: Iterable[Point[Any]]) -> None:
    """Write points to a CSV file.

    Args:
        outfile: str or PathLike where CSV file will be written.
        points: Iterable of Point objects to write.
    """
    to_pandas(points, all_attrs=False).to_csv(outfile, index=False)


def predictions_to_csv(outfile: StrPath, points: Iterable[LabelledPoint]) -> None:
    """Write track predictions to a CSV file.

    Args:
        outfile: str or PathLike where CSV file will be written.
        points: Iterable of LabelledPoint objects containing predictions.
    """
    to_pandas(points, all_attrs=False)[[AISColumns.pid, AISColumns.tid]].to_csv(
        outfile, index=False
    )
