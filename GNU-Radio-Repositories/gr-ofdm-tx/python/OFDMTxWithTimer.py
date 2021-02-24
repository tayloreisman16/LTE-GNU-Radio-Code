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
import platform
from gnuradio import gr

pickle_directory = '/home/tayloreisman/Downloads/GNU_Radio_Project_Feb_11_v2/SDRFile_Beta_v4/'
# pickle_directory = '/home/tayloreisman/Downloads/test.jpeg'
# img_directory = '/home/tayloreisman/Downloads/test.jpeg'


class OFDMTxWithTimer(gr.sync_block):
    """
    docstring for block OFDMTxWithTimer
    """
    def __init__(self, case, pickle_directory):
        self.work_call_count = 0
        self.work_calls_per_timer_block = 30
        self.timer_count = 0
        self.max_timer = 1  # Number of rows in input matrix
        self.case = case
        # Insert Pickle File Here
        self.pickle_directory = pickle_directory
        # f = open(str(self.pickle_directory) + str(case) + "_" + "tx_data.pckl", 'rb')
        # self.tx_mat = pickle.load(f)
        # f.close()

        gr.sync_block.__init__(self,
            name="OFDMTxWithTimer",
            in_sig = None,
            out_sig=[numpy.complex64])

    def work(self, input_items, output_items):
        # in0 = input_items[0]
        out = output_items[0]
        
        # tx_data = numpy.concatenate((self.tx_mat[self.timer_count][:], self.tx_mat[self.timer_count][:]), axis=0)
        tx_data = self.tx_mat[self.timer_count][:]
        # print(tx_data.shape)
        out[0: tx_data.shape[0]] = tx_data
        # print(out)

        self.work_call_count += 1
        # print(platform.python_version())

        if self.work_call_count % self.work_calls_per_timer_block == 0:
            self.timer_count += 1

        if self.timer_count % self.max_timer == 0:
            self.timer_count = 0
        
        return len(output_items[0])

