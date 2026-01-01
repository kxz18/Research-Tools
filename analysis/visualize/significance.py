#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def _get_borderline_one_side(r, l, h):
    if r is None: return l, h
    rl, rh = r[0], r[1]
    eps = 0.001 * (h - l)
    if (rl - eps < l and l < rh + eps) or (rl - eps < h and h < rh + eps):
        return r[1], r[1] + h - l
    return l, h

def _get_borderline(r1, r2, l, h):
    l1, h1 = _get_borderline_one_side(r1, l, h)
    l2, h2 = _get_borderline_one_side(r2, l, h)
    return max(l1, l2), max(h1, h2)


def add_significance_pair(ax, data, x_name, y_name, hue_name, col_a, col_b, mark_ranges):
    # statistical annotation
    x_order = [t.get_text() for t in ax.get_xticklabels()]
    hue_order = [t.get_text() for t in ax.legend_.texts] if (ax.legend_ is not None) else None
    patch_to_key = {}
    if hue_order is None:
        for i, patch in enumerate(ax.patches): patch_to_key[x_order[i]] = patch
    else:
        n_x, n_hue = len(x_order), len(hue_order)
        for i, patch in enumerate(ax.patches):
            if i == (n_x * n_hue): break
            x_idx = i % n_x
            h_idx = i // n_x

            patch_to_key[(x_order[x_idx], hue_order[h_idx])] = patch
    
    x1 = patch_to_key[col_a].get_x() + patch_to_key[col_a].get_width() / 2
    x2 = patch_to_key[col_b].get_x() + patch_to_key[col_b].get_width() / 2

    y_vals, x1_y_vals, x2_y_vals = [], [], []
    hue = [None] * len(data[x_name]) if hue_name is None else data[hue_name]
    for x, y, h in zip(data[x_name], data[y_name], hue):
        mark = x if h is None else (x, h)
        if mark == col_a:
            x1_y_vals.append(y)
            y_vals.append(y)
        elif mark == col_b:
            x2_y_vals.append(y)
            y_vals.append(y)
    span = (max(data[y_name]) - min(data[y_name])) * 0.05
    y, h, col = max(y_vals) + span, span, 'k'
    y = _get_borderline(mark_ranges.get(col_a, None), mark_ranges.get(col_b, None), y, y + h)[0]

    p = ttest_ind(x1_y_vals, x2_y_vals).pvalue
    p_str = f'{p:.3e}' if p < 0.01 else str(round(p, 3))
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, f'p = {p_str}', ha='center', va='bottom', color=col)
    return (y, y + 2 * h)    # annotation range


def add_significance(ax, data, x_name, y_name, hue_name, pairs):
    # make ylim larger so that the p-value can be fit into the figure
    mark_ranges = {}
    for x1, x2 in pairs:
        if isinstance(x1, list): x1 = tuple(x1)
        if isinstance(x2, list): x2 = tuple(x2)
        # shift y position according to how many times this group has been marked
        l, h = add_significance_pair(ax, data, x_name, y_name, hue_name, x1, x2, mark_ranges)
        if h * 1.1 >= ax.get_ylim()[1]:
            ax.set_ylim(top=h * 1.1)
        if x1 not in mark_ranges: mark_ranges[x1] = [l, h]
        if x2 not in mark_ranges: mark_ranges[x2] = [l, h]
        mark_ranges[x1][0] = min(mark_ranges[x1][0], l)
        mark_ranges[x1][1] = max(mark_ranges[x1][1], h)
        mark_ranges[x2][0] = min(mark_ranges[x2][0], l)
        mark_ranges[x2][1] = max(mark_ranges[x2][1], h)
