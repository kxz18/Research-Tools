#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from .post_editor import post_edit


def heatmap(data, transpose=False, post_edit_func=None, save_path=None, **kwargs):
    if isinstance(data, list):
        data = np.array(data)
    if transpose:
        data = data.T
    ax = sns.heatmap(data, **kwargs)
    post_edit(ax, post_edit_func, save_path)
    return ax


if __name__ == '__main__':
    '''example of heatmap'''
    import numpy as np
    n, m = 10, 20
    values = np.random.randn(n, m)
    heatmap(values, transpose=False, save_path='heatmap_eg.png')
    heatmap(values, transpose=True, save_path='heatmap_eg_t.png')
