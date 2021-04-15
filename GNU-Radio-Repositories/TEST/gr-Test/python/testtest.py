#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 UTSA.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy
from gnuradio import gr


class testtest(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self,
            name="testtest",
            in_sig=[numpy.float32],
            out_sig=[numpy.float32])

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        out[:] = in0
        return len(output_items[0])

