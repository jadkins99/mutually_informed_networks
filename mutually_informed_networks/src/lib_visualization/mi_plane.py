import matplotlib.pyplot as plt


def plot_mi_plane(mi_x_per_layer: list[list[float]], mi_y_per_layer: list[list[float]]):
    """
    Plots the mutual information for each layer in a 2D plane where the x-axis represents
    the mutual information with the input and the y-axis represents the mutual information
    with the output.

    :param mi_x_per_layer: For each layer, a list of mutual information values with the input along each epoch.
    :param mi_y_per_layer: For each layer, a list of mutual information values with the output along each epoch.
    """


    fig, ax = plt.subplots()

    for layer_idx, (mi_x, mi_y) in enumerate(zip(mi_x_per_layer, mi_y_per_layer)):
        _plot_layer_in_mi_plane(ax, mi_x, mi_y, layer_idx)

    ax.set_xlabel('Mutual Information X')
    ax.set_ylabel('Mutual Information Y')
    ax.set_title('Mutual Information Plane')

    return ax


def _plot_layer_in_mi_plane(ax, mi_x: list[float], mi_y: list[float], layer_idx: int):
    """ Plots the mutual information trajectory of a single layer in the MI plane connected with a line. """
    ax.scatter(mi_x, mi_y, s='-o', label=f'Layer {layer_idx}')
    return ax