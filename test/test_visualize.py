import matplotlib.pyplot as plt
import numpy as np

from ltm.visualize import fig2array


def test_fig2array():
    # Create a sample figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

    # Convert the figure to an array
    arr = fig2array(fig)

    # Check the shape and type of the array
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (fig.canvas.get_width_height()[::-1] + (4,))

    # Check the values in the array
    assert np.allclose(
        arr[0, 0], [1, 1, 1, 1]  # pylint: disable=unsubscriptable-object
    )

    # Check that the array is in RGBA format
    assert np.allclose(arr[..., 3], 1)  # pylint: disable=unsubscriptable-object
