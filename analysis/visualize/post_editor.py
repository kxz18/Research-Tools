import matplotlib.pyplot as plt


def post_edit(ax, func=None, save_path=None):
    if func is not None:
        func(ax)

    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()

    return ax
