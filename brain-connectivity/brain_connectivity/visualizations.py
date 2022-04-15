import matplotlib.pyplot as plt
from seaborn import heatmap


def plot_fc_matrix(matrix, epoch):
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax = heatmap(
        matrix,
        cmap='RdBu', center=0.0,
        xticklabels=10, yticklabels=10
    )
    ax.set_title(f'FC matrix at {epoch} epochs')
    plt.show()
