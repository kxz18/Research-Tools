#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from .post_editor import post_edit


def barplot(data, x, y, hue=None, post_edit_func=None, save_path=None, **kwargs):
    ax = sns.barplot(data=data, x=x, y=y, hue=hue, **kwargs)
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
