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


class TxSignalTransmitter(gr.sync_block):

    def __init__(self, pickle_directory, pickle_file):
        gr.sync_block.__init__(self,
                               name="SimpleTx",
                               in_sig=None,
                               out_sig=[np.complex64])
        f = open(str(pickle_directory) + str(pickle_file))
        self.tx_data = pickle.load(f)
        f.close()

    def work(self, input_items, output_items):
        out = output_items[0]
        out[0:self.tx_data.shape[1]] = self.tx_data[0, :]
        return len(output_items[0])

