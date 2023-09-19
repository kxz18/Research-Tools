#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from .post_editor import post_edit


def piechart(data, post_edit_func=None, save_path=None, **kwargs):
    '''
    data: dict<label, cnt>
    '''
    cnts, labels = [], []
    for label in data:
        cnts.append(data[label])
        labels.append(label)
    palette = kwargs.get('palette', 'pastel')
    colors = sns.color_palette(palette)[:len(labels)]
    dpi = kwargs.get('dpi', 400)

    plt.figure(dpi=dpi)
    plt.pie(cnts, labels=labels, colors=colors, autopct='%.0f%%')
    ax = plt.gca()

    post_edit(ax, post_edit_func, save_path)
    return ax


if __name__ == '__main__':
    '''example of pychart'''

    data = {
        'Type 1': 10,
        'Type 2': 23,
        'Type 3': 1,
    }

    piechart(data, save_path='piechart_eg.jpg')