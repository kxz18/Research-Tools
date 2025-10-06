#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from .post_editor import post_edit


def regplot(data, x, y, hue=None, reg_share=False, post_edit_func=None, save_path=None, **kwargs):
    '''
    Args:
        reg_share: whether to draw a unified regression line over all classes
    '''
    fig_reg = not (hue is not None and reg_share)
    data = pd.DataFrame(data)
    fig = sns.lmplot(data=data, x=x, y=y, hue=hue, fit_reg=fig_reg, **kwargs)
    ax = fig.axes[0, 0]
    if not fig_reg:
        ax = sns.regplot(data=data, x=x, y=y, scatter=False, ax=ax)
    post_edit(ax, post_edit_func, save_path)
    return ax


if __name__ == '__main__':
    '''example of regplots'''

    # 2-dimensional data
    data = {
        'dim1': np.array([1, 2, 3, 3, 4, 5, 5, 6, 7]),
        'dim2': np.array([5, 6, 5, 2, 1, 2, 5, 6, 5]),
        'type': np.array(['eye', 'eye', 'eye', 'mouth', 'mouth', 'mouth', 'eye', 'eye', 'eye'])
    }
    regplot(data, x='dim1', y='dim2', hue='type', save_path='regplot_eg1.png')
    regplot(data, x='dim1', y='dim2', hue='type', reg_share=True, save_path='regplot_reg_share_eg2.png')