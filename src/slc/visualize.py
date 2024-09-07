"""Module for visualizing raster data, creating plots and saving visualizations as images.

Typical usage example:

    import matplotlib.pyplot as plt
    import pandas as pd
    from ltm.visualize import plot_report

    # Create a DataFrame
    df = pd.DataFrame({
        "A": [1, 2, 4],
        "B": [4, 6, 7],
    })

    # Plot the DataFrame
    plot_report(df, title="My Plot", xlabel="X-axis", ylabel="Y-axis", figsize=(10, 5), marker="o")

    plt.show()
"""

import io
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from typeguard import typechecked

from slc.data import combine_band_name, split_band_name


def _rgb_bands2indices(
    raster: np.ndarray | None = None,
    bands: Tuple[str, ...] | None = None,
    rgb_bands: List[str | None] | None = None,
) -> List[str]:
    # Check if bands or raster is provided
    if bands is None and raster is None:
        raise ValueError("Either bands or raster must be provided")

    # Check whether bands exists if rgb_bands is not None
    if rgb_bands is not None and bands is None:
        raise ValueError("bands must not be None if rgb_bands is not None")

    # Check if bands is equal to the number of bands in raster
    if bands is not None and len(bands) != raster.shape[0]:
        raise ValueError(
            f"bands must have same length as number of bands in raster: len(bands)={len(bands)} != raster.shape[0]={raster.shape[0]}"
        )

    # Default bands to tuple of integers if None
    if bands is None:
        bands = tuple(str(number) for number in range(raster.shape[0]))

    # Check if rgb_bands has length of three or is None
    if rgb_bands is not None and len(rgb_bands) != 3:
        raise ValueError("rgb_bands must be a list of length 3 or None")

    # Fill rgb_bands with bands and maybe None if it is None
    if rgb_bands is None:
        if len(bands) == 1:
            rgb_bands = [bands[0]] * 3
        elif len(bands) == 2:
            rgb_bands = [bands[0], bands[1], None]
        else:
            rgb_bands = list(bands[:3])

    # Get the indices of the bands
    rgb_indices = []
    for rgb_band in rgb_bands:
        if rgb_band is not None:
            rgb_indices.append(bands.index(rgb_band))
        else:
            rgb_indices.append(None)

    return rgb_indices


@typechecked
def _raster2rgb(
    raster: np.ndarray,
    bands: Tuple[str, ...] | None = None,
    rgb_bands: List[str | None] | None = None,
    mask_nan: bool = True,
) -> np.ndarray:
    """Creates an RGB image from a rasterio raster.

    Args:
        raster:
            Rasterio raster to convert to RGB.
        bands:
            Tuple of band names. Must not be None if rgb_bands is not None. Defaults to None.
        rgb_bands:
            A list of strings representing the band names to use for the RGB image. Defaults to the first three bands if None. Except for when there is only one band, then the RGB image will be grayscale. Or for two bands only R and G will be used. You get the idea. I had to do something for default.
        mask_nan:
            A boolean whether to mask NaN values. If any band has a NaN value, the whole pixel will be masked. Defaults to True.

    Returns:
        A NumPy array RGB image.
    """
    # Get RGB band indices
    rgb_indices = _rgb_bands2indices(raster, bands, rgb_bands)

    # Create RGB image bands
    rgb_plot = []
    for rgb_idx in rgb_indices:
        if rgb_idx is not None:
            rgb_plot.append(raster[rgb_idx])
        else:
            rgb_plot.append(np.zeros_like(raster[0, :, :]))

    # Stack RGB image bands
    rgb_plot = np.dstack(rgb_plot)

    # Mask NaN values
    if mask_nan:
        nan_mask = np.any(np.isnan(rgb_plot), axis=2)
        rgb_plot[nan_mask] = np.nan

    return rgb_plot


@typechecked
def _plot_timeseries(
    rgb_rasters: List[np.ndarray],
    reducer_title: str,
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    # Get num_composites
    num_composites = len(rgb_rasters)

    # Get min and max values for normalization
    min_value = np.nanmin(np.array(rgb_rasters))
    max_value = np.nanmax(np.array(rgb_rasters))

    # Plot the rasters below each other
    fig, axs = plt.subplots(nrows=num_composites, figsize=(10, 10))
    if num_composites == 1:
        axs = np.array([axs])
    fig.tight_layout()
    for i, (ax, rgb_raster) in enumerate(zip(axs, rgb_rasters), start=1):
        # normalize values and apply gamma correction
        rgb_raster -= min_value
        rgb_raster /= max_value - min_value
        rgb_raster **= 1 / 2.2
        rgb_raster[np.isnan(rgb_raster)] = 0
        ax.imshow(rgb_raster)
        ax.set_title(f"{i} {reducer_title}")
        ax.axis("off")

    plt.show()

    return fig, axs


@typechecked
def show_timeseries(
    raster_path: str,
    reducer: str,
    rgb_bands: List[str] | None = None,
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """Shows a timeseries of composites.

    Args:
        raster_path:
            A string representing the file path to the raster.
        reducer:
            A string representing the reducer to use when creating the composite.
        rgb_bands:
            A list of strings representing the bands to use for the RGB composite. If None the first bands are used. Defaults to None.

    Returns:
        A tuple of the matplotlib figure and axes.
    """
    # Read raster and get band names
    with rasterio.open(raster_path) as src:
        raster = src.read()
        bands = src.descriptions

    # Find the reducer title and the number of composites
    num_composites = 0
    for band in bands:
        composite_idx, _, curr_reducer, _ = split_band_name(band)
        if curr_reducer.lower() == reducer.lower():
            reducer = curr_reducer
            num_composites = max(num_composites, composite_idx)
    bands = [band.lower() for band in bands]

    if num_composites == 0:
        raise ValueError(f"No bands found with reducer {reducer}")

    # Get the rasters for the rgb bands
    rgb_rasters = []
    for i in range(1, num_composites + 1):
        rgb_raster = []
        if rgb_bands is not None:
            curr_rgb_bands = []
            for band in rgb_bands:
                if band is None:
                    curr_rgb_bands.append(None)
                else:
                    band_lower = combine_band_name(
                        i, band=band, reducer=reducer
                    ).lower()
                    curr_rgb_bands.append(band_lower)
        else:
            curr_rgb_bands = None

        rgb_raster = _raster2rgb(raster, tuple(bands), curr_rgb_bands)
        rgb_rasters.append(rgb_raster)

    # Plot the rasters
    return _plot_timeseries(rgb_rasters, reducer)


@typechecked
def plot_report(  # pylint: disable=too-many-arguments
    df: pd.DataFrame,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label_rotation: int = 0,
    replace_labels: dict | None = None,
    categorical_x: bool = True,
    **kwargs: Any,
) -> plt.Axes:
    """Plots a DataFrame with a title, x and y label and legend.

    The index of the DataFrame is used for the x-axis and the columns for the lines in the plot. The legend is placed to the right of the plot.

    Args:
        df:
            DataFrame to plot.
        title:
            Title of the plot. Defaults to None.
        xlabel:
            Label for the x-axis. Defaults to None.
        ylabel:
            Label for the y-axis. Defaults to None.
        label_rotation:
            Rotation of the x-axis labels. Defaults to 0.
        replace_labels:
            Dictionary to replace labels in the legend. Works also for replacing a subset of the labels. Defaults to None.
        categorical_x:
            Whether the x-axis is categorial and thus each index is an xtick. Not recommended for very long DataFrames. Defaults to True.
        **kwargs:
            Additional keyword arguments to pass to df.plot(), e.g. figsize=(10, 5) or marker="o".

    Returns:
        Axes of the plot.
    """
    # Create empty dictionary if replace is None
    if replace_labels is None:
        replace_labels = {}

    # Plot
    ax = df.plot(**kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set xticks
    if categorical_x:
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=label_rotation)
        ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)

    # Replace labels
    labels = ax.get_legend_handles_labels()[1]
    for i, label in enumerate(labels):
        if label in replace_labels.keys():
            labels[i] = replace_labels[label]

    # Set legend
    golden_ratio = (1 + 5**0.5) / 2
    ax.legend(labels, loc="center left", bbox_to_anchor=(1, 1 / golden_ratio))

    return ax


@typechecked
def fig2array(fig: plt.Figure | None = None) -> np.ndarray:
    """Converts a matplotlib figure to a numpy array.

    Args:
        fig:
            Matplotlib figure to convert. Defaults to None, which will use the current figure.

    Returns:
        Numpy array of the figure.
    """
    with io.BytesIO() as buff:
        if fig is None:
            fig = plt.gcf()
        fig.savefig(buff, format="png")
        buff.seek(0)
        im = plt.imread(buff)

    return im
