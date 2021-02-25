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
import math
import pickle
from gnuradio import gr

pickle_file_directory = '/home/taylor/Downloads/GNU_Radio_Project/SDR File Beta v1/'


class OFDMTransmitter(gr.sync_block):
    """
    docstring for block OFDMTransmitter
    """

    def __init__(self, case, num_ofdm_symb, fft_size, num_data_bins):
        gr.sync_block.__init__(self,
                               name="OFDMTransmitter",
                               in_sig=None,
                               out_sig=[numpy.complex64])
        self.case = case
        self.num_ofdm_symb = num_ofdm_symb
        self.fft_size = fft_size
        self.num_data_bins = num_data_bins
        # print(self.num_ofdm_symb)
        self.CP_size = int(self.fft_size / 4)
        self.symb_len = int(self.fft_size + self.CP_size)
        # print(self.symb_len)
        self.data_no = 0
        self.num_unique_data = 1
        # print(self.num_ofdm_symb*self.symb_len)

        self.work_calls_per_data_set_tx = int(math.ceil((self.symb_len * self.num_ofdm_symb) / 4095.0))
        self.num_repeat_per_data_set = 20
        print(self.work_calls_per_data_set_tx)
        self.total_work_calls_per_data_set = self.work_calls_per_data_set_tx * self.num_repeat_per_data_set

        self.file_flag = True  # self
        self.data_repeat_count = 0  # self
        self.left_over_flag = False  # self
        self.mode = "init"  # self "init" ot "left_over"
        self.data_left_over = []  # self
        self.data_in = []
        self.work_call_count = 0

    def work(self, input_items,output_items):

        # in0 = input_items[0]
        out = output_items[0]
        self.work_call_count += 1
        # print(self.work_call_count)

        if self.file_flag:
            # read from pickle

            file_name = "tx_data_" + str(self.case) + "_" + str(self.data_no) + ".pckl"
            # print(file_name)
            self.data_no += 1

            f = open(pickle_file_directory + file_name, 'rb')
            self.data_in = pickle.load(f)
            # print(str(pickle_file_directory) + str(file_name))

            self.file_flag = False
        # print(self.mode)
        # print("Length of data in", len(self.data_in))
        # if repeat_flag == True
        if self.mode == "left_over":
            data_out = self.data_left_over
            self.data_repeat_count += 1
            self.mode = "init"
        else:
            if self.data_in.shape[1] > 4095:
                # print("I AM HERE")
                # print()
                data_out = self.data_in[0, 0: 4095]
                # print(data_out.shape)
                self.data_left_over = self.data_in[0, 4095:]  # self
                self.mode = "left_over"
            else:
                data_out = self.data_in[0, :]
                self.data_repeat_count += 1
                self.mode = "init"

        # print(self.data_repeat_count)

        # if self.work_call_count == self.total_work_calls_per_data_set:
        # print(self.total_work_calls_per_data_set)
        if self.work_call_count % self.total_work_calls_per_data_set == 0:
            self.file_flag = True
            self.data_repeat_count = 0
            # print(self.work_call_count)

        if self.data_no == self.num_unique_data:
            self.data_no = 0
            # if self.data_repeat_count == self.num_repeat_per_data_set:
            #     self.data_repeat_count = 0
            # print(self.work_call_count)
        # print data_out.shape
        # print data_out.shape[1]
        # print(data_out.shape)
        out[0:len(data_out)] = data_out
        return len(output_items[0])