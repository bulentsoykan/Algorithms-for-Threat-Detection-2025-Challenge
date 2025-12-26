import argparse
import logging
import os
import random
import zipfile
from collections import namedtuple
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from time import time
from typing import List, Optional, Sequence, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from .constants import INVALID_COURSE, INVALID_SPEED

__all__ = [
    "validate_date",
    "query_ais",
    "ais_to_df",
    "trim_invalid_values",
    "restrict_range",
    "valid_range",
    "get_region",
    "show_valid_regions",
    "trim_stationary",
    "apply_spatial_error",
    "apply_temporal_error",
    "subsample_ais_df",
    "df_to_csv",
    "process_data",
]

logger = logging.getLogger()

Region = namedtuple("Region", ["name", "lat_range", "lon_range"])

RegionOptions = dict(
    PR=Region(name="Puerto Rico", lat_range=(15, 21), lon_range=(-70.5, -62.5)),
    EC=Region(name="Eastern US Seaboard", lat_range=(22, 45.5), lon_range=(-85, -62)),
    CT=Region(name="Central US", lat_range=(22, 45.5), lon_range=(-98, -85)),
    GC=Region(name="Gulf Coast", lat_range=(22, 31), lon_range=(-98, -75)),
    WC=Region(name="Western US Seaboard", lat_range=(25, 55.5), lon_range=(-130, -110)),
    HI=Region(name="Hawaii", lat_range=(14, 27), lon_range=(-166, -152)),
    SF=Region(name="South Florida", lat_range=(22, 29), lon_range=(-85, -75)),
)

RegionList = list(RegionOptions.keys())


class CommandLine:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Download AIS data and format into ATD2025 standard format."
        )
        parser.add_argument(
            "-l",
            "--loc",
            help="Specifies the location where your raw AIS data files are stored or should "
            "be stored.",
            required=True,
        )
        parser.add_argument(
            "-d",
            "--date",
            help="Which day to format AIS data in the format MM/DD/YYYY",
            required=True,
        )
        parser.add_argument(
            "-r",
            "--region",
            help="Region to restrict formatted data to.",
            required=False,
            default="",
        )
        parser.add_argument(
            "-o",
            "--out",
            help="Specifies where to save formatted AIS files to",
            required=False,
            default="out/",
        )

        self.args = parser.parse_args()

        if self.args.date:
            self.args.date = validate_date(self.args.date)


def validate_date(date: str) -> datetime:
    """Converts a string date to datetime object and ensures it is within the available date range.

    Args:
        date: Date as a string in the format MM/DD/YYYY. Must be between 01/01/2015 and 03/31/2024.

    Returns:
        A datetime object for the given date.

    Raises:
        ValueError: If the date is outside of the available range.

    """

    date_format = "%m/%d/%Y"
    date = datetime.strptime(date, date_format)
    earliest = datetime(2015, 1, 1)
    latest = datetime(2024, 3, 31)
    if not earliest <= date <= latest:
        raise ValueError(
            f"Provided date must be between {earliest.strftime(date_format)} and "
            f"{latest.strftime(date_format)}"
        )

    return date


def query_ais(date: datetime, out: Path) -> Path:
    """Downloads and extracts AIS data from the US Coast Guard database.

    Downloads AIS data for the given day from the US Coast Guard database and extracts it to the
    specified output location. If data already exists in the output location for the given day,
    then just return its location.

    Additionally creates the output directory if it does not exist.

    Args:
        date: The date for which to retrieve AIS data.
        out: The directory to store and/or retrieve AIS data from.

    Returns:
        A string representing the location of the AIS csv for the given date.

    """

    year = date.strftime("%Y")
    query_format = "%Y_%m_%d"

    # out = Path(out) / year
    out.mkdir(parents=True, exist_ok=True)
    csv_filename = f"AIS_{date.strftime(query_format)}.csv"
    zip_filename = f"AIS_{date.strftime(query_format)}.zip"

    if os.path.isfile(out / csv_filename):
        logger.info("Data already present. Skipping...")
        return out / csv_filename

    if not os.path.isfile(out / zip_filename):
        logger.info("Downloading data...")
        url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/{zip_filename}"
        urlretrieve(url, out / zip_filename)

    with zipfile.ZipFile(out / zip_filename, "r") as zip_ref:
        logger.info("Extracting...")
        zip_ref.extractall(out)

    return out / csv_filename


def ais_to_df(
    csv_location: Path,
    region: Optional[str] = None,
    trim_invalid: bool = True,
    avoid_splitting_tracks: bool = True,
) -> pd.DataFrame:
    """Converts an ais csv into a df with the correct format.

    Reads the csv cointaining ais posits from the given location as a dataframe. Renames the columns
    and restricts the data to a region, if one is provided.

    Args:
        csv_location: File location of the raw AIS csv
        region (optional): A predefined region that you can specify with either an int or 2-letter
            code from the list below. Optional if lat_range and lon_range are supplied.
            Defaults to "".
                [0] PR: Puerto Rico
                [1] EC: Eastern US Seaboard
                [2] CT: Central US
                [3] GC: Gulf Coast
                [4] WC: Western US Seaboard
                [5] HI: Hawaii
                [6] SF: South Florida
        trim_invalid: (optional): If true, removes posits with invalid speed or course values.
        avoid_splitting_tracks (optional): When false, remove all points that extend out of the
            defined region. When true, keep all points of tracks that have at least one point
            within the defined region. Defaults to True.

    Returns:
        A formatted dataframe containing ais posits for a given region (if region provided).

    """
    logging.info("Reading csv...")
    keep_cols = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"]

    df = pd.read_csv(csv_location, usecols=keep_cols)
    df = df.rename(
        columns={
            "MMSI": "track_id",
            "BaseDateTime": "time",
            "LAT": "lat",
            "LON": "lon",
            "SOG": "speed",
            "COG": "course",
        }
    )

    df["time"] = "2024-01-01T" + df["time"].str[11:]
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%S")
    df.index.name = "point_id"

    if trim_invalid:
        df = trim_invalid_values(df)
    if region is not None:
        df = restrict_range(df, region=region, avoid_splitting_tracks=avoid_splitting_tracks)
    return df


def trim_invalid_values(
    df: pd.DataFrame, invalid_course: float = INVALID_COURSE, invalid_speed: float = INVALID_SPEED
) -> pd.DataFrame:
    """Removes rows with invalid values in course, speed, lat, or lon columns.

    Args:
        df: The dataframe to trim the course and speed columns for.
        invalid_course (Optional): The invalid course value to remove. Default is 360.
        invalid_speed (Optional): The invalid speed value to remove. Default is 102.3.

    Returns:
        Dataframe with no rows containing invalid values.
    """

    df = df.copy()

    df = df[(df["lon"] >= -180) & (df["lon"] <= 180)]
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90)]

    df = df[(df["course"] >= 0) & (df["course"] != invalid_course)]
    df = df[(df["speed"] >= 0) & (df["speed"] != invalid_speed)]

    return df


def restrict_range(
    df: pd.DataFrame,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    region: str = "",
    avoid_splitting_tracks: bool = True,
) -> pd.DataFrame:
    """Return dataframe that only has tracks within the specified spatial area.

    Example:
    restrict_range(df, region="PR")
    returns a dataframe containing only posits close to Puerto Rico.

    Args:
        df: The dataframe to trim points from.
        lat_range (optional): [min_latitude, max_latitude] Latitude bounds. Defaults to None.
        lon_range (optional): [min_longitude, max_longitude] Longitude bounds. Defaults to None.
        region (optional): A predefined region that you can specify with either an int or
            2-letter code from the list below. Defaults to "".
                [0] PR: Puerto Rico
                [1] EC: Eastern US Seaboard
                [2] CT: Central US
                [3] GC: Gulf Coast
                [4] WC: Western US Seaboard
                [5] HI: Hawaii
                [6] SF: South Florida
        avoid_splitting_tracks (optional): When false, remove all points that extend out of the
            defined region. When true, keep all points of tracks that have at least one point
            within the defined region. Defaults to True.

    Returns:
        Dataframe with only has posits within the bounds specified.

    """

    df_copy = df.copy()

    lat_range, lon_range = valid_range(lat_range=lat_range, lon_range=lon_range, region=region)

    if lon_range:
        df_copy = df_copy[
            np.logical_and(lon_range[0] <= df_copy["lon"], df_copy["lon"] <= lon_range[1])
        ]

    if lat_range:
        df_copy = df_copy[
            np.logical_and(lat_range[0] < df_copy["lat"], df_copy["lat"] < lat_range[1])
        ]

    if not avoid_splitting_tracks:
        return df_copy
    else:
        indexes = list(df_copy.track_id.unique())
        return df.loc[df["track_id"].isin(indexes)]


def valid_range(
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    region: str = "",
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Validates the supplied latitude and longitude ranges.

    Ensures the latitude and longitude bounds are given in the correct format.
    Or if a region is supplied, return the valid latitude/longitude ranges for the region.

    Args:
        lat_range (optional): The latitude range to be validated. Defaults to None.
        lon_range (optional): The longitude range to be validated. Defaults to None.
        region (optional): A predefined region that you can specify with either an
            int or 2-letter code from the list below. Optional if lat_range and
            lon_range are supplied. Defaults to "".
                [0] PR: Puerto Rico
                [1] EC: Eastern US Seaboard
                [2] CT: Central US
                [3] GC: Gulf Coast
                [4] WC: Western US Seaboard
                [5] HI: Hawaii
                [6] SF: South Florida

    Returns:
        A tuple (lat_range, lon_range) where lat_range and lon_range are the supplied
            latitude and longitude ranges if given and valid, or the latitude and longitude
            ranges of the region if a region code was supplied.

    Raises:
        ValueError: If the first given longitude is larger than the second given longitude
            or the first given latitude is larger than the second given latitude.

    """

    if lon_range and lon_range[0] >= lon_range[1]:
        raise ValueError(f"Invalid longitude range: {lon_range[0]} >= {lon_range[1]}")

    if lat_range and lat_range[0] >= lat_range[1]:
        raise ValueError(f"Invalid latitude range: {lat_range[0]} >= {lat_range[1]}")

    if region:
        lat_range, lon_range = (
            get_region(region).lat_range,
            get_region(region).lon_range,
        )

    assert lat_range is not None
    assert lon_range is not None
    return lat_range, lon_range


def get_region(key: str) -> Region:
    """Converts a region string to its corresponding region object.

    Args:
        key: The two letter string representation of the desired region.

    Returns:
        The Region object for the corresponding key.

    Raises:
        ValueError: If the given key is not a valid option.

    """

    try:
        return RegionOptions[key]
    except Exception:
        try:
            return RegionOptions[RegionList[int(key)]]
        except Exception:
            raise ValueError(f"{key} is not a valid options.\n{show_valid_regions()}")


def show_valid_regions(include_bounds: bool = False) -> str:
    """Returns the available regions.

    Returns the available regions as a string, optionally with the lat/lon bounds for each region.

    Args:
        include_bounds (optional): Whether to include the latitude/longitude bounds in the output.
            Defaults to False.

    Returns:
        A string listing the available regions and optionally their bounds.

    """

    result = "Region options:\n"
    if include_bounds:
        result += "\n".join([f"[{k}] {r}: {RegionOptions[r]}" for k, r in enumerate(RegionList)])
    else:
        result += "\n".join(
            [f"[{k}] {r}: {RegionOptions[r].name}" for k, r in enumerate(RegionList)]
        )
    return result


def trim_stationary(
    df: pd.DataFrame,
    keep_first: bool = True,
    keep_last: bool = True,
    remove_edge: bool = False,
) -> pd.DataFrame:
    """Removes stationary points from vessel tracks.

    Splits an input dataframe by vessel, then removes stationary points from each resulting group.
    Points are considered stationary if they have 0 speed.

    Can optionally keep the first and/or last stationary point among groups of stationary
    points. Can also optionally remove all stationary points on the boundary of the group.

    Expects that the input dataframe is sorted by time.

    Example:
    df = |  time | speed |
         |-------|-------|
         |   0   |   0   |
         |   1   |  0.5  |
         |   2   |   0   |
         |   3   |   0   |
         |   4   |   0   |
         |   5   |   0   |
         |   6   |  0.1  |
         |   7   |  1.2  |
         |   8   |   0   |
         |   9   |   0   |

     keep_first=False, keep_last=False, remove_edge=False:
     |  time | speed |
     |-------|-------|
     |   1   |  0.5  |
     |   6   |  0.1  |
     |   7   |  1.2  |

     keep_first=True, keep_last=False, remove_edge=False:
     |  time | speed |
     |-------|-------|
     |   0   |   0   |
     |   1   |  0.5  |
     |   2   |   0   |
     |   6   |  0.1  |
     |   7   |  1.2  |
     |   8   |   0   |

     keep_first=False, keep_last=True, remove_edge=False:
     |  time | speed |
     |-------|-------|
     |   0   |   0   |
     |   1   |  0.5  |
     |   5   |   0   |
     |   6   |  0.1  |
     |   7   |  1.2  |
     |   9   |   0   |

     keep_first=False, keep_last=False, remove_edge=True:
     |  time | speed |
     |-------|-------|
     |   1   |  0.5  |
     |   2   |   0   |
     |   3   |   0   |
     |   4   |   0   |
     |   5   |   0   |
     |   6   |  0.1  |
     |   7   |  1.2  |

     keep_first=True, keep_last=True, remove_edge=True:
     |  time | speed |
     |-------|-------|
     |   1   |  0.5  |
     |   2   |   0   |
     |   5   |   0   |
     |   6   |  0.1  |
     |   7   |  1.2  |

    Args:
        df: The dataframe group to trim. Must be sorted by time.
        keep_first (optional): Whether to keep the first stationary point from a consecutive group
            of stationary points. Defaults to true.
        keep_last (optional): Whether to keep the last stationary point from a consecutive group of
            stationary points. Defaults to true.
        remove_edge (optional): Whether to trim the beginning and end of the dataframe such that
            the dataframe starts and ends with a non-stationary point. Defaults to true.

    Returns:
        A dataframe with consecutive stationary points trimmed out.

    """

    df = df.sort_values(["time"])

    trimmed_df = df.groupby("track_id", as_index=False).apply(
        partial(_trim_group, keep_first=keep_first, keep_last=keep_last, remove_edge=remove_edge),
        include_groups=False,
    )

    trimmed_df.index = trimmed_df.index.droplevel(0)

    return df.loc[trimmed_df.index]


def _trim_group(
    df: pd.DataFrame,
    keep_first: bool = True,
    keep_last: bool = True,
    remove_edge: bool = True,
) -> pd.DataFrame:
    """Helper function for trim_stationary. Removes stationary points from vessel group.

    Removes stationary points from a dataframe group. Points are considered stationary
    if they have 0 speed. Can optionally keep the first and/or last stationary point among
    groups of stationary points. Can also optionally remove all stationary points on the boundary
    of the group.

    Expects that the input dataframe is sorted by time and only contain points belonging
    to the same vessel.

    See trim_stationary() for examples.


    Args:
        df: The dataframe group to trim. Must only contain points belonging to the same vessel
            and be sorted by time.
        keep_first (optional): Whether to keep the first stationary point from a consecutive
            group of stationary points. Defaults to true.
        keep_last (optional): Whether to keep the last stationary point from a consecutive group of
            stationary points. Defaults to true.
        remove_edge (optional): Whether to trim the beginning and end of the dataframe such that
            the dataframe starts and ends with a non-stationary point. Defaults to true.

    Returns:
        A dataframe with consecutive stationary points trimmed out.

    """

    if remove_edge:
        start = df.speed.ne(0).idxmax()
        end = df[::-1].speed.ne(0).idxmax()

        df[(df.index >= start) & (df.index <= end)]

    keep = df.speed != 0
    if keep_first:
        keep = keep | (df.speed.shift(1) != 0)
    if keep_last:
        keep = keep | (df.speed.shift(-1) != 0)
    return df[keep]


def apply_spatial_error(
    df: pd.DataFrame,
    cov: Sequence[Sequence[float]] = ((0.0000001, 0.0000001), (0.0000001, 0.0000009)),
) -> pd.DataFrame:
    """Applies error spatially to all points in the df.

    Applies error following a gausian distribution to all longtide and latitude points in the df.

    Args:
        df: The dataframe to add error to.
        cov (optional): The covariance for the gausian distribution.
            Defaults to [[0.0000001, 0.0000001], [0.0000001, 0.0000009]].

    Returns:
        A dataframe with error applied to each latitude and longitude point.

    """

    df = df.copy()
    pts = np.random.multivariate_normal([0, 0], cov, len(df))
    df.lon = df.lon + pts[:, 0]
    df.lat = df.lat + pts[:, 1]

    return df


def apply_temporal_error(df: pd.DataFrame, seconds: int = 3) -> pd.DataFrame:
    """Applies error temporally to all points in the df.

    Applies random error between -seconds and seconds to all times in the df.

    Args:
        df: The dataframe to add error to.
        seconds (optional): Range for how many seconds should the time error should vary by.

    Returns:
        A dataframe with temporal error applied to each point.

    """
    df = df.copy()
    choices = [timedelta(seconds=x) for x in range(-seconds, seconds + 1)]
    td_list = [random.choice(choices) for i in range(len(df))]

    df["time"] = df["time"].add(td_list)  # type: ignore[arg-type]
    df.loc[df["time"] < datetime(2024, 1, 1), "time"] = df.loc[
        df["time"] < datetime(2024, 1, 1), "time"
    ].add(timedelta(seconds=seconds))
    df.loc[df["time"] >= datetime(2024, 1, 2), "time"] = df.loc[
        df["time"] >= datetime(2024, 1, 2), "time"
    ].sub(timedelta(seconds=seconds))  # type: ignore[arg-type]

    return df


def subsample_ais_df(df: pd.DataFrame, delta: pd.Timedelta = pd.Timedelta("0.5h")) -> pd.DataFrame:
    """Subsamples the given dataframe to the given sampling rate.

    Groups the input dataframe by unique tracks, then subsamples them to the given sampling rate.
    Points will be less than delta time apart, if possible.

    Args:
        df: The dataframe to subsample.
        delta (optional): The sampling rate of the maximum amount of time between points,
            if possible. Defaults to 30 minutes.

    Returns:
        A dataframe with points within tracks at most 30 minutes apart, if possible.

    """
    return (
        df.groupby("track_id")
        .apply(partial(_subsample_ais_single, delta=delta), include_groups=False)
        .reset_index()
        .set_index("point_id")
    )


def _subsample_ais_single(
    df: pd.DataFrame, delta: pd.Timedelta = pd.Timedelta("0.5h")
) -> pd.DataFrame:
    """Helper function for subsample_ais_df to subsample a single track."""

    df = df.sort_values(by="time", ascending=True)

    temp_df = (
        df.sort_values(by="time", ascending=False)
        .reset_index()
        .set_index("time", drop=False)[["point_id", "time"]]
    )

    temp_df["window"] = temp_df.time.rolling(delta).count()
    temp_df = temp_df.drop("time", axis=1).sort_values(by="time", ignore_index=True)
    temp_df.iat[len(temp_df) - 1, len(temp_df.columns) - 1] = np.nan

    return df.iloc[_get_subsample_indexes(temp_df, 0)]


def _get_subsample_indexes(df: pd.DataFrame, idx: int, keep_before_delta: int = 1) -> List[int]:
    """Helper function for _subsample_ais_single to determine the range of indexes to subsample."""

    next_idx = df.iat[idx, len(df.columns) - 1]
    if np.isnan(next_idx):
        return [idx]
    if next_idx >= len(df):
        return [idx - 1]

    idx_list = _get_subsample_indexes(
        df, max(idx + 1, idx + int(next_idx) - keep_before_delta), keep_before_delta
    )
    idx_list.insert(0, idx)

    return idx_list


def df_to_csv(df: pd.DataFrame, out_location: Path, filename: str) -> None:
    """Writes an AIS dataframe to a csv and truth csv pair.

    Writes an AIS dataframe to a truth and data pair of csvs.
    Due to the applied errors, output csvs will not always be in the same order.

    Args:
        df: The dataframe to convert to csv.
        out_location: The directory to write the csv.
        filename: The name of the csv file to write. A corresponding truth file will
            also be generated.

    Returns:
        A dataframe with points within tracks at most 30 minutes apart, if possible.

    """
    logging.info("Exporting...")

    df = df.copy()

    out_location.mkdir(parents=True, exist_ok=True)
    out_date_format = "%Y-%m-%dT%H:%M:%S"

    if "." not in filename:
        filename += ".csv"

    truth_filename = "_truth.".join(filename.split("."))

    df = df.sort_values(["time"]).reset_index(names="point_id_original")
    df.index.name = "point_id"

    df.to_csv(out_location / truth_filename, date_format=out_date_format)
    df = df.drop(["point_id_original", "track_id"], axis=1)

    df.to_csv(out_location / filename, date_format=out_date_format)


def process_data(
    data_location: Path,
    date: str,
    region: str = "GC",
) -> pd.DataFrame:
    """Downloads AIS data and converts into a formatted dataframe.

    Downloads data for the specified date from the AIS website if not found locally,
    then processes it and returns the data as a dataframe.

    Args:
        data_location: Location where your raw AIS data files should be stored
        date: Date to download and format data for in the format MM/DD/YYYY.
        region (optional): Region to restrict the formatted data to. Default is 'GC'

    Returns:
        A formatted dataframe with the AIS data for the specified date.

    """

    date = validate_date(date)
    csv_location = query_ais(date, data_location)

    df = ais_to_df(csv_location, region)
    df = subsample_ais_df(df)
    df = trim_stationary(df)
    df = apply_spatial_error(df)
    df = apply_temporal_error(df)

    return df


def main() -> None:
    t0 = time()
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.WARNING) # Uncomment to disable logs
    app = CommandLine()
    args = app.args

    data_location = Path(args.loc)
    csv_location = query_ais(args.date, data_location)
    out_location = Path(args.out)
    region = args.region

    df = process_data(data_location, args.date, region)

    if region:
        region = "_" + region

    filename = csv_location.stem + region + csv_location.suffix
    df_to_csv(df, out_location, filename)
    logging.info("Done.")
    print(time() - t0)
    exit()
