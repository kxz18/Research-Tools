# -*- coding:utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def interp_unsorted(x_array, y_array, kind='cubic', interp_num=1000):
    sort_idx = sorted([i for i in range(len(x_array))], key=lambda idx: x_array[idx])
    x_array = [x_array[i] for i in sort_idx]
    y_array = [y_array[i] for i in sort_idx]
    x_interp = np.linspace(x_array[0], x_array[-1], interp_num)
    y_interp = interp1d(x_array, y_array, kind=kind)(x_interp)
    return x_interp, y_interp


def lineplot(data, x, y, hue=None, smooth=None, save_path=None, **kwargs):
    '''
    smooth method: quadratic, cubic
    '''
    if smooth is not None:
        new_data = {
            x: [],
            y: []
        }
        if hue is None:
            new_data[x], new_data[y] = interp_unsorted(data[x], data[y], kind=smooth)
        else:
            new_data[hue] = []
            data_dict = {}
            for _x, _y, _hue in zip(data[x], data[y], data[hue]):
                if _hue not in data_dict:
                    data_dict[_hue] = [[], []]
                data_dict[_hue][0].append(_x)
                data_dict[_hue][1].append(_y)
            for _hue in data_dict:
                ori_x, ori_y = data_dict[_hue]
                x_interp, y_interp = interp_unsorted(ori_x, ori_y, kind=smooth)
                new_data[x].extend(x_interp)
                new_data[y].extend(y_interp)
                new_data[hue].extend([_hue for _ in x_interp])
        data = new_data

    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, **kwargs)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()

    return ax


if __name__ == '__main__':
    '''example of lineplot'''

    data = {
        'x': [1, 2, 3, 4, 1, 2, 3, 4],
        'y': [1, 8, 27, 64, 2, 5, 10, 17],  # x^3 and x^2 + 1
        'func': ['x^3' for _ in range(4)] + ['x^2 + 1' for _ in range(4)] 
    }
    lineplot(data, x='x', y='y', hue='func', save_path='lineplot_eg1.png')
    lineplot(data, x='x', y='y', hue='func', smooth='cubic', save_path='lineplot_eg2.png')