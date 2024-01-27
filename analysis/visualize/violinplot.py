#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from .post_editor import post_edit


def violinplot(data, x=None, y=None, hue=None, post_edit_func=None, save_path=None, **kwargs):
    '''
    data: dict<label, cnt>
    '''
    assert x is not None or y is not None
    ax = sns.violinplot(data, x=x, y=y, hue=hue, **kwargs)

    post_edit(ax, post_edit_func, save_path)
    return ax


if __name__ == '__main__':
    '''example of violinplot'''
    import numpy as np
    n_sample = 20
    classes = ['A', 'B', 'C', 'D']
    types = ['1', '0']
    data = {
        'class': [],
        'type': [],
        'value': np.random.randn(n_sample * len(classes))
    }
    for c in classes:
        for i in range(n_sample):
            data['class'].append(c)
            data['type'].append(types[i % len(types)])

    violinplot(data, x='class', y='value', hue='type', save_path='violinplot_eg.jpg')
    violinplot(data, y='value', hue='type', save_path='violinplot_v_eg.jpg')
    violinplot(data, x='value', hue='type', save_path='violinplot_h_eg.jpg')
    violinplot(data, x='class', y='value', save_path='violinplot_no_hue_eg.jpg')