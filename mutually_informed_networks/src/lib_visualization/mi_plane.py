import matplotlib.pyplot as plt
import numpy as np


def plot_mi_plane(mi_x_per_layer: list[list[float]], mi_y_per_layer: list[list[float]]):
    """
    Plots the mutual information for each layer in a 2D plane where the x-axis represents
    the mutual information with the input and the y-axis represents the mutual information
    with the output.

    :param mi_x_per_layer: For each layer, a list of mutual information values with the input along each epoch.
    :param mi_y_per_layer: For each layer, a list of mutual information values with the output along each epoch.
    """
    fig, ax = plt.subplots()

    mi_x_el = np.array(mi_x_per_layer)  # shape (num_steps, num_layers)
    mi_y_el = np.array(mi_y_per_layer)  # shape (num_steps, num_layers)

    for layer_idx in range(mi_x_el.shape[1]):
        mi_x = mi_x_el[:, layer_idx].tolist()
        mi_y = mi_y_el[:, layer_idx].tolist()
        _plot_layer_in_mi_plane(ax, mi_x, mi_y, layer_idx)

    ax.grid()
    ax.set_xlabel('Mutual Information X')
    ax.set_ylabel('Mutual Information Y')
    ax.set_title('Mutual Information Plane')
    ax.legend()

    plt.savefig('mi_plane.png')  # TODO: save to a better place


def _plot_layer_in_mi_plane(ax, mi_x: list[float], mi_y: list[float], layer_idx: int):
    """ Plots the mutual information trajectory of a single layer in the MI plane connected with a line. """
    ax.plot(mi_x, mi_y, marker='o', linestyle='-', label=f'Layer {layer_idx}')
    return ax