import os

import matplotlib.pyplot as plt
from seaborn import heatmap


def fc_matrix_heatmap(matrix, epoch, path, title=None):
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax = heatmap(
        matrix, cmap="RdBu", center=0.0, xticklabels=10, yticklabels=10
    )

    ax.set_title(title if title else f"FC matrix at {epoch} epochs")
    plt.savefig(os.path.join(path, f"fc_matrix_{epoch}.pdf"))
