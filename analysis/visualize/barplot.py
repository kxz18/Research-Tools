#!/usr/bin/python
# -*- coding:utf-8 -*-
import seaborn as sns
sns.set_theme()

from .post_editor import post_edit
from .significance import add_significance


def barplot(data, x, y, hue=None, post_edit_func=None, save_path=None, overlay_points=False, point_size=10, significance_pairs=None, **kwargs):
    ax = sns.barplot(data=data, x=x, y=y, hue=hue, **kwargs)
    if overlay_points:
        sns.stripplot(x=x, y=y, hue=hue, data=data, dodge=True, size=point_size, alpha=0.6, ax=ax, legend=False)
    if significance_pairs is not None:
        add_significance(ax, data, x, y, hue, significance_pairs, plot_type='barplot')
    post_edit(ax, post_edit_func, save_path)
    return ax


if __name__ == '__main__':
    '''example of bar plot'''
    import numpy as np
    n = 10
    values1 = [np.random.randint(0, 10) for _ in range(n)]
    values2 = [np.random.randint(8, 12) for _ in range(n)]
    data = {
        'value': values1 + values2,
        'class': ['A' for _ in range(n)] + ['B' for _ in range(n)],
        'subtype': (['major' for _ in range(n // 2)] + ['minor' for _ in range(n // 2)]) * 2
    }
    # vertical layout
    barplot(data, x='class', y='value', hue='subtype', save_path='barplot_vertical.png')
    # horizontal layout
    barplot(data, x='value', y='class', hue='subtype', save_path='barplot_horizontal.png')

    # some suggestions on plotting scientific figures
    sns.set_theme(style='white', palette='Set2')  # palette
    sns.set_context('paper', font_scale=1.8)
    barplot(
        data, x='class', y='value', hue='subtype', save_path='barplot_sci.png',
        errorbar='sd', edgecolor='black', err_kws={'linewidth': 3.0}, capsize=0.4, alpha=0.5,
        overlay_points=True, point_size=10, significance_pairs=[(('A', 'minor'), ('B', 'minor')), (('A', 'major'), ('B', 'minor'))]
    )