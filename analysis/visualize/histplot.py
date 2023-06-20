#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def histplot(data, x, hue=None, density=False, save_path=None, **kwargs):
    ax = sns.histplot(data=data, x=x, hue=hue, stat='density' if density else 'count', **kwargs)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    return ax


if __name__ == '__main__':
    '''example of hist plot'''
    import numpy as np
    n = 1000
    values1 = [np.random.randint(0, 10) for _ in range(n)]
    values2 = [np.random.randint(8, 12) for _ in range(n)]
    data = {
        'value': values1 + values2,
        'class': [0 for _ in range(n)] + [1 for _ in range(n)]
    }
    histplot(data, x='value', hue='class', density=False, save_path='histplot_eg1.png')
    histplot(data, x='value', hue='class', density=True, save_path='histplot_eg2.png')