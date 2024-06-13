import numpy as np
import matplotlib.pyplot as plt


def post_edit(ax, func=None, save_path=None):
    if func is not None:
        func(ax)

    if save_path is not None:
        plt.savefig(save_path, dpi=600)
        plt.clf()

    return ax


def auto_set_ticks_continous(ax, values, axis, num_ticks=6, r=0):
    '''
    axis: 'x' or 'y'
    '''
    lb, ub = min(values), max(values)
    soft_lb, soft_ub = lb - (ub - lb) * 0.05, ub + (ub - lb) * 0.05
    ticks = np.linspace(lb, ub, num_ticks, endpoint=True)
    if r > 0:
        p = pow(10, r)
        tick_last = round(ticks[-1], r)
        ticks = [int(v * p) / p for v in ticks]
        ticks[-1] = tick_last
    getattr(ax, f'set_{axis}ticks')(ticks)
    getattr(plt, f'{axis}lim')(soft_lb, soft_ub)


def auto_set_ticks_integer(ax, values, axis, num_ticks=6):
    '''
    axis: 'x' or 'y'
    '''
    lb, ub = min(values), max(values)
    span = max(1, int((ub - lb) / num_ticks + 0.5))
    ticks = np.arange(lb - lb % span, ub + span, span)
    soft_lb = ticks[0] - (span / 2 if ticks[0] == lb else 0)
    soft_ub = ticks[-1] + (span / 2 if ticks[-1] == ub else 0)

    getattr(ax, f'set_{axis}ticks')(ticks)
    getattr(plt, f'{axis}lim')(soft_lb, soft_ub)