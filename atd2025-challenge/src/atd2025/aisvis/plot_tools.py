from typing import Any, Dict

import plotly.graph_objs as go  # type: ignore[import-untyped]

color_swatch = [
    "#3366CC",
    "#DC3912",
    "#FF9900",
    "#109618",
    "#990099",
    "#0099C6",
    "#DD4477",
    "#66AA00",
    "#B82E2E",
    "#316395",
]


standard_font = {"family": "Times New Roman", "size": 26, "color": "black"}
tick_font = {"family": "Times New Roman", "size": 20, "color": "black"}
legend_font = {"family": "Times New Roman", "size": 18, "color": "black"}


def simplify_style(fig: go.Figure, do_axes: bool = True, plot_title: str = "") -> go.Figure:
    """Return a Plotly figure with a clean, sparse style and easy-to-read labels.

    Parameters
    ----------
    fig: go.Figure
    do_axes: Optional[bool}
        If True, modify the axes by removing tick marks and modifying fonts.
        Default is True.
    plot_title: Optional[str]
        If specified, will include a title for the plot.

    Returns
    -------
    go.Figure
    """
    fig.update_layout(
        font=standard_font,
        title=_title_layout(),
        legend=_legend_layout(),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    if plot_title:
        fig.layout.title["text"] = plot_title
    if do_axes:
        simplify_axes(fig)

    return fig


def simplify_axes(fig: go.Figure) -> None:
    """Update go.Figure fig with pretty axes that have a clean, spare style.

    Parameters
    ----------
    fig: go.Figure
        Will update the axes on this figure.

    Returns
    -------
    None
    """
    fig.update_yaxes(ticks="", tickfont=tick_font)
    fig.update_xaxes(showline=False, ticks="", tickfont=tick_font)


def _title_layout() -> Dict[str, Any]:
    """Called by simplify_style."""
    # Written as a function to avoid unintended changes to _title_layout
    return {
        "y": 0.95,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": standard_font,
    }


def _legend_layout() -> dict:  # type: ignore[type-arg]
    """Called by simplify_style."""

    return dict(
        font=legend_font,
        orientation="h",
        yanchor="bottom",
        y=1.0,
        xanchor="center",
        x=0.5,
    )
