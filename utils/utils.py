#!/usr/bin/env python
# -*- coding: utf-8 -*-


def inf_loop(data_loader):
    from itertools import repeat

    for loader in repeat(data_loader):
        yield from loader
