import warnings
from os import PathLike
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
import plotly.graph_objs as go  # type: ignore[import-untyped]
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from ..io import read_points_as_df
from . import plot_tools as pts
from .base import AISColumns, assert_valid_ranges, get_region, restrict_range, show_valid_regions

BASE_VECTOR_LENGTH = 0.002
MOD_TIME_COL = "np_timedelta"
DEFAULT_RANGE = (-300, 300)

ArrayLike = Union[np.ndarray, pd.Series]  # type: ignore[type-arg]
StrPath = Union[str, PathLike[str]]


@overload
def make_line_plot(
    df: pd.DataFrame,
    return_n: Literal[False] = False,
    show_lines: bool = True,
    show_velocity: bool = False,
    color: str = "",
    show_text: bool = False,
) -> go.Figure: ...
@overload
def make_line_plot(
    df: pd.DataFrame,
    return_n: Literal[True] = True,
    show_lines: bool = True,
    show_velocity: bool = False,
    color: str = "",
    show_text: bool = False,
) -> Tuple[go.Figure, int]: ...
def make_line_plot(
    df: pd.DataFrame,
    return_n: bool = False,
    show_lines: bool = True,
    show_velocity: bool = False,
    color: str = "",
    show_text: bool = False,
) -> Union[go.Figure, Tuple[go.Figure, int]]:
    """Map with ship posits (associated with tracks, if known).

    Args:
        df: Dataframe with AIS tracks.
        return_n: If True, also return n, where n is the number of ship tracks (default is False).
        show_lines: If True, draw a line segment starting at each marker in the direction of
            Course Over Ground (COG). Length is proportional to speed (default is False).
        show_velocity: If True, posits will include line segment representing velocity. (points in
            the direction of velocity, and length scaled with speed) (default is False).
    color: Options for coloring the markers (default is None). These are the options:
        color="time": color markers by time since the first observation.
        color="uniform": color all markers blue.
        Any other value: color markers by ship tid.
    show_text: If True, timestamps will appear below each posit (default is False).

    Returns:
        fig: Map showing posits and tracks. To display in a browser, use this command:
            fig.show("browser")
        n_tracks: Number of ship tracks in the plot.
    """
    min_time = df[AISColumns.time].min()
    df[MOD_TIME_COL] = (df[AISColumns.time] - min_time) / np.timedelta64(1, "h")

    # ADD an ID column?
    if AISColumns.tid not in df:
        if show_lines:
            show_lines = False
            warnings.warn("Data doesn't include ship tid, so plot only shows posits, no tracks.")
        df[AISColumns.tid] = "-1"

    mode = _plot_mode(show_lines, show_text)
    ships = np.unique(df[AISColumns.tid])
    fig = go.Figure()
    for ship in tqdm(ships):
        ship_track = df[df[AISColumns.tid] == ship]
        fig.add_trace(
            go.Scattergeo(
                mode=mode,
                lon=ship_track[AISColumns.lon],
                lat=ship_track[AISColumns.lat],
                customdata=ship_track[AISColumns.time],
                name=str(ship),
                **_posit_text(show_text, ship_track[MOD_TIME_COL]),
                hovertemplate=str(ship) + "<br> Lat: %{lat} <br> Lon: %{lon} <br> "
                "Time: %{"
                "customdata}",
                marker=_posit_marker(color, ship_track[MOD_TIME_COL]),
                line_width=3,
            )
        )
        if show_velocity:
            _show_velocity(ship_track, fig)

    if return_n:
        return _make_plot_pretty(fig), len(ships)
    else:
        return _make_plot_pretty(fig)


def _posit_text(show_text: bool, tk: np.ndarray) -> dict:  # type: ignore[type-arg]
    if show_text:
        return dict(
            text=[f"{h:.2f} hours" for h in tk],
            textfont=dict(size=12),
            textposition="bottom center",
        )
    else:
        return dict(text=None)


def _posit_marker(color: str, tk: np.ndarray) -> dict:  # type: ignore[type-arg]
    if color.lower() == "time":
        return dict(size=10, color=tk)
    elif color.lower() == "uniform":
        return dict(size=10, color="blue")
    else:
        return dict(size=10)


def _plot_mode(show_lines: bool, show_text: bool) -> str:
    """Scatter plot mode for plotly."""
    result = "markers"
    if show_lines:
        result += "+lines"

    if show_text:
        result += "+text"
    return result


@overload
def degrees_to_radians(degrees: float) -> float: ...
@overload
def degrees_to_radians(degrees: np.ndarray) -> np.ndarray: ...  # type: ignore[type-arg]
@overload
def degrees_to_radians(degrees: pd.Series) -> pd.Series: ...  # type: ignore[type-arg]
def degrees_to_radians(degrees: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
    return (degrees / 360) * 2 * np.pi


def cog_to_vector(cog: Union[float, ArrayLike]) -> np.ndarray:  # type: ignore[type-arg]
    """Convert course over ground to a vector.

    Args:
        cog: Course over ground in degrees.

    Returns:
        Vector(s) representating course over ground direction.
    """
    cog_in_radians = degrees_to_radians(cog)
    return np.array([np.cos(cog_in_radians), np.sin(cog_in_radians)])


def speed_cog_to_vector(speed: ArrayLike, cog: ArrayLike) -> np.ndarray:  # type: ignore[type-arg]
    """Calculate approximate vectors v0 v1 for use in cubic parametization.


    This approach *probably* results in local some linearization of the problem,
    and it gets worse the further apart points are.


    Args:
        speed: Speed in arbitrary units (float or size N array).
        cog: Course over ground degrees clockwise of north (float or size N array).

    Returns:
        Velocity vector(s) in [north, east] direction in same units as input speed.
    """
    direction_vector = cog_to_vector(cog)
    return (  # type: ignore[no-any-return]
        direction_vector * speed
        if isinstance(speed, float)
        else direction_vector * np.reshape(speed, (1, speed.size))
    )


def _show_velocity(track: pd.DataFrame, fig: go.Figure) -> None:
    """Draws line segments starting on posits corresponding to velocity."""
    vec = (speed_cog_to_vector(track[AISColumns.sog], track[AISColumns.cog]) * BASE_VECTOR_LENGTH).T
    endpts = track[[AISColumns.lat, AISColumns.lon]] + vec
    line_break = np.full(len(track), None)
    lat_vals = np.array(
        [track[AISColumns.lat].values, endpts[AISColumns.lat].values, line_break]
    ).T.flatten()
    lon_vals = np.array(
        [track[AISColumns.lon].values, endpts[AISColumns.lon].values, line_break]
    ).T.flatten()
    fig.add_trace(
        go.Scattergeo(
            lon=lon_vals,
            lat=lat_vals,
            mode="lines",
            line=dict(color="black", width=3),
            showlegend=False,
        )
    )


def _make_plot_pretty(fig: go.Figure) -> go.Figure:
    result = pts.simplify_style(fig)
    result.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.0,
        ),
        geo=dict(
            fitbounds="locations",
            showland=True,
            showlakes=True,
            showrivers=True,
            lonaxis=dict(showgrid=True, dtick=1),
            lataxis=dict(showgrid=True, dtick=1),
        ),
    )
    return result


def main(
    datafile: Annotated[
        Path,
        typer.Argument(help="Path to .csv containing AIS tracks."),
    ],
    outfile: Annotated[
        Optional[str],
        typer.Option("--out", "-o", help="Will save plot to this file as a .html "),
    ] = "",
    image: Annotated[
        Optional[str],
        typer.Option("--image", "-i", help="Will write plot to this image file"),
    ] = "",
    sort: Annotated[
        bool,
        typer.Option(
            "--sort",
            "-s",
            help="Sort dataframe by ship ID first, then increasing datetime.",
        ),
    ] = False,
    show_velocity: Annotated[
        bool,
        typer.Option(
            "--vel",
            "-e",
            help="Plot velocity vector over each posit.",
        ),
    ] = False,
    lat_range: Annotated[
        Tuple[float, float],
        typer.Option(
            "--lat-range",
            "-a",
            help="Only show tracks between these latitude values (lat min, lat max)."
            "\n \n show datafile.csv -a 19 22",
            show_default=False,
        ),
    ] = DEFAULT_RANGE,
    lon_range: Annotated[
        Tuple[float, float],
        typer.Option(
            "--lon-range",
            "-g",
            help="Only show tracks between these longitude values (lon min, lon max)."
            "\n \n show datafile.csv -g -66 -63"
            "",
            show_default=False,
        ),
    ] = DEFAULT_RANGE,
    region: Annotated[
        Optional[str],
        typer.Option(
            "--region",
            "-r",
            help="Only show tracks in the specified region. If specified, lat-range "
            "and lon-range will be ignored. To specify use either the number "
            "or 2-letter code from below.\n\n" + show_valid_regions(),
            show_default=False,
        ),
    ] = None,
) -> None:
    """Command-line function to plot tracks saved in a .csv

    For example, to automatically open a browser and plot the tracks in ais.csv:

    aisshow ais.csv

    To save in an html file called tracks.html:
    aisshow -o tracks.html ais.csv

    To save in a .jpg called out.jpg:
    aisshow -i out.jpg ais.csv

    """
    if region is not None:
        _region = get_region(region)
        lat_range, lon_range = _region.lat_range, _region.lon_range
    assert_valid_ranges(lat_range, lon_range)
    df = restrict_range(read_points_as_df(datafile), lat_range, lon_range)

    if sort:
        df = df.sort_values([AISColumns.tid, AISColumns.time])

    fig, n_tracks = make_line_plot(df, return_n=True, show_velocity=show_velocity or False)
    if _range_given(lat_range) and _range_given(lon_range):
        fig.update_geos(lataxis=dict(range=lat_range), lonaxis=dict(range=lon_range))

    print(f"Plotting {n_tracks:,d} tracks and {len(df):,d} posits.")

    if image:
        save_image(fig, image)
    if outfile:
        print(f"Saving plot in {outfile}.")
        fig.write_html(outfile)
    else:
        fig.show("browser")


def save_image(fig: go.Figure, filename: StrPath) -> None:
    """Save the plot as an image.

    Args:
        fig: Figure
        filename: Output file path.
    """
    print(f"Saving image in {filename}...")
    original_margin = fig.layout.margin
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    fig.write_image(filename)
    fig.update_layout(showlegend=True, margin=original_margin)
    print("... done")


def _range_given(coords: Tuple[float, float]) -> bool:
    return coords != DEFAULT_RANGE


def cli_main() -> None:
    typer.run(main)
