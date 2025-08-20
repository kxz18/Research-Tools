#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from .post_editor import post_edit


def high_dim_scatterplot(data, vec, x='dim1', y='dim2', hue=None, labels=None, post_edit_func=None, save_path=None, **kwargs):
    # dimension reduce with T-SNE
    X = TSNE(n_components=2, perplexity=min(30, len(data[vec]))).fit_transform(np.array(data[vec]))  # [N, 2]
    reduced_data = {key: data[key] for key in data if key != vec}
    reduced_data[x], reduced_data[y] = X[:, 0], X[:, 1]

    # draw scatterplot
    return scatterplot(reduced_data, x, y, hue, labels, post_edit_func, save_path, **kwargs)


def scatterplot(data, x, y, hue=None, labels=None, post_edit_func=None, save_path=None, use_color_bar_for_hue=False, **kwargs):
    ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, **kwargs)
    if labels is not None:
        for i, label in enumerate(labels):
            if label is None:
                continue
            ax.text(data[x][i], data[y][i], str(label), fontweight='bold')
    if use_color_bar_for_hue and (hue is not None):
        norm = plt.Normalize(min(data[hue]), max(data[hue]))
        sm = plt.cm.ScalarMappable(cmap=kwargs.get('palette', None), norm=norm)
        sm.set_array([])
        ax.get_legend().remove()
        ax.figure.colorbar(sm, ax=ax)
    post_edit(ax, post_edit_func, save_path)
    return ax


if __name__ == '__main__':
    '''example of scatter plots'''

    # 2-dimensional data
    data = {
        'dim1': [1, 2, 3, 3, 4, 5, 5, 6, 7],
        'dim2': [5, 6, 5, 2, 1, 2, 5, 6, 5],
        'type': ['eye', 'eye', 'eye', 'mouth', 'mouth', 'mouth', 'eye', 'eye', 'eye']
    }
    scatterplot(data, x='dim1', y='dim2', hue='type', save_path='scatterplot_eg1.png')

    # high-dimensional data
    n = 20
    class1 = np.random.randn(n, 100)
    class2 = np.random.randn(n, 100) + 5
    data = {
        'vectors': np.concatenate([class1, class2], axis=0),
        'class': [1 for _ in range(n)] + [2 for _ in range(n)]
    }
    high_dim_scatterplot(data, vec='vectors', hue='class', save_path='scatterplot_eg2.png')
