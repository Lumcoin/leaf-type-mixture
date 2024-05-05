import matplotlib.pyplot as plt
import numpy as np

from ltm.visualize import fig2array


def test_fig2array():  # TODO: check as it is AI generated
    # Create a sample figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

    # Convert the figure to an array
    arr = fig2array(fig)

    # Check the shape and type of the array
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (fig.canvas.get_width_height()[::-1] + (4,))

    # Check the values in the array
    assert np.allclose(arr[0, 0], [1, 1, 1, 1])
    assert np.allclose(arr[1, 1], [1, 1, 1, 1])
    assert np.allclose(arr[2, 2], [1, 1, 1, 1])

    # Check that the array is in RGB format
    assert np.allclose(arr[..., 3], 1)
