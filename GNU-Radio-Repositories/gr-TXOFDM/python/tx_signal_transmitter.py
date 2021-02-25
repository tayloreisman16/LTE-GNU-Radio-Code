#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 UTSA.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
import pickle
from gnuradio import gr


class tx_signal_transmitter(gr.sync_block):

    def __init__(self, case, pickle_directory, pickle_file):
        gr.sync_block.__init__(self,
                               name="SimpleTx",
                               in_sig=None,
                               out_sig=[np.complex64])
        f = open(str(pickle_directory) + str(pickle_file))
        self.tx_data = pickle.load(f)
        f.close()

    def work(self, input_items, output_items):
        # in0 = input_items[0]
        out = output_items[0]
        # <+signal processing here+>
        # print(self.tx_data)
        # print(out.shape)
        # print(self.tx_data.shape)
        out[0:self.tx_data.shape[1]] = self.tx_data[0, :]
        # print(out[0:10])
        return len(output_items[0])

