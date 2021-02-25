#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2018 <+YOU OR YOUR COMPANY+>.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

import numpy
import pickle
from gnuradio import gr


class SimpleTx(gr.sync_block):
    """
    docstring for block SimpleTx
    """
    def __init__(self, case, pickle_directory):
        gr.sync_block.__init__(self,
                               name="SimpleTx",
                               in_sig=None,
                               out_sig=[numpy.complex64])
        f = open(str(pickle_directory) + '/' + 'tx_data_' + str(case) + '.pckl')
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

