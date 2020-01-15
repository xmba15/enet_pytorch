#!/usr/bin/env python
# -*- coding: utf-8 -*-


def mean_iu_acc(output, target, num_classes, epsilon=1e-9):
    # mean_iu
    import numpy as np

    sum_iu = 0
    for i in range(num_classes):
        n_ii = t_i = sum_n_ji = epsilon
        for p, gt in zip(output, target):
            n_ii += np.sum(gt[p == i] == i)
            t_i += np.sum(gt == i)
            sum_n_ji += np.sum(p == i)
        sum_iu += float(n_ii) / (t_i + sum_n_ji - n_ii)
    mean_iu = sum_iu / num_classes

    return mean_iu
