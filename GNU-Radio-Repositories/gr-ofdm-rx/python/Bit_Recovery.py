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

import numpy as np
import scipy.io
import csv
import pickle
import datetime
from gnuradio import gr


# file_name = '/home/taylor/Desktop/rx_data.csv'

class BitRecovery(gr.sync_block):
    """
    docstring for block BitRecovery
    """

    def __init__(self, modulation, directory_name, diagnostics):
        gr.sync_block.__init__(self,
                               name="BitRecovery",
                               in_sig=[np.complex64],
                               out_sig=None)
        self.modulation = modulation
        self.directory_name = directory_name
        self.diagnostics = diagnostics

        self.symbol_count = 0
        self.mapping_type = 'QPSK'
        self.EC_code = None
        self.data_in = None
        self.softbit0 = None
        self.softbit1 = None
        self.hardbit = None

    def work(self, input_items, output_items):
        in0 = input_items[0]
        # print("In", in0[100:110])

        DataIn = in0
        self.data_in = DataIn[:, np.newaxis]
        self.softbit0 = np.zeros(
            (self.data_in.shape[0], 2 * self.data_in.shape[0] * self.data_in.shape[1] * self.data_in.shape[2]))
        self.softbit1 = np.zeros(
            (self.data_in.shape[0], 2 * self.data_in.shape[0] * self.data_in.shape[1] * self.data_in.shape[2]))
        self.hardbit = np.zeros(
            (self.data_in.shape[0], 2 * self.data_in.shape[0] * self.data_in.shape[1] * self.data_in.shape[2]))
        self.symbol_count += 1

        data_shp0 = self.data_in.shape[0]
        data_shp1 = self.data_in.shape[1]
        data_shp2 = self.data_in.shape[2]

        num_el_data = data_shp0 * data_shp1 * data_shp2

        if self.mapping_type == 'BPSK':
            print('BPSK demod not currently implemented')
            exit(0)
        elif self.mapping_type == 'QPSK':
            cmplx_phsrs = np.exp(1j * 2 * (np.pi / 8) * np.array([1, -1, 3, 5]))

            # permute the dimensions of data_in
            data_rearrange = np.transpose(self.data_in, [2, 1, 0])

            data0 = np.reshape(data_rearrange.T, (int(num_el_data / data_shp0), int(data_shp0)))

            for loop in range(data_shp0):
                iq_data = data0[:, loop]

                llrp0 = np.zeros(
                    2 * data_shp1 * len(self.used_bins_data))  # 2 for QPSK - each IQ phasor corresponds to 2 bits
                llrp1 = np.zeros(2 * data_shp1 * len(self.used_bins_data))

                cmplx_phsrs_ext = np.tile(cmplx_phsrs, (len(iq_data), 1)).T
                data_ext = np.tile(iq_data, (4, 1))

                dist = abs(data_ext - cmplx_phsrs_ext)

                dmin = np.min(dist, 0)
                dmin_ind = np.argmin(dist, 0)

                dz = cmplx_phsrs[dmin_ind]

                ez = iq_data - dz

                sigma00 = np.mean(abs(dmin))
                sigma0 = np.sqrt(0.5 * sigma00 * sigma00)
                d_factor = 1 / sigma0 ** 2

                K = 2 / np.sqrt(2)

                for kk in range(len(iq_data)):
                    if dz[kk].real >= 0 and dz[kk].imag >= 0:
                        llrp0[2 * kk] = -0.5 * abs(ez[kk].real)
                        llrp1[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                        llrp0[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                        llrp1[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))

                    elif dz[kk].real <= 0 and dz[kk].imag >= 0:
                        llrp0[2 * kk] = -0.5 * abs(ez[kk].real)
                        llrp1[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                        llrp1[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                        llrp0[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))
                    elif dz[kk].real <= 0 and dz[kk].imag <= 0:
                        llrp1[2 * kk] = -0.5 * abs(ez[kk].real)
                        llrp0[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                        llrp1[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                        llrp0[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))
                    elif dz[kk].real >= 0 and dz[kk].imag <= 0:
                        llrp1[2 * kk] = -0.5 * abs(ez[kk].real)
                        llrp0[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                        llrp0[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                        llrp1[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))

                llrp0 *= d_factor
                llrp1 *= d_factor

                # softbit00 = np.zeros(np.size(llrp0))
                # softbit11 = np.zeros(np.size(llrp1))

                self.softbit0[loop, 1::2] = llrp0[0::2]
                self.softbit0[loop, 0::2] = llrp0[1::2]

                self.softbit1[loop, 1::2] = llrp1[0::2]
                self.softbit1[loop, 0::2] = llrp1[1::2]

                self.hardbit[loop, :] = np.ceil(0.5 * (np.sign(self.softbit1[loop, :] - self.softbit0[loop, :]) + 1))

                est_bits = self.hardbit[loop, :]

        if self.count == 30:
            var_mat = dict()
            var_mat['softbit0'] = self.softbit0
            var_mat['softbit1'] = self.softbit1
            var_mat['hardbit'] = self.hardbit
            scipy.io.savemat(str(self.directory_name) + '/' + 'rx_bit_recov.mat', var_mat)

        if self.diagnostics == 1:
            date_time = datetime.datetime.now().strftime('%Y_%m_%d_%Hh_%Mm')
            file_name0 = self.directory_name + '/softbit0_' + date_time + '.pckl'
            file_name1 = self.directory_name + '/softbit1_' + date_time + '.pckl'

            f = open(file_name0, 'wb')
            pickle.dump(softbit0, f, protocol=2)
            f.close()

            f = open(file_name1, 'wb')
            pickle.dump(softbit1, f, protocol=2)
            f.close()

            file_name = open(self.directory_name + '/rx_data.csv', 'a')
            with file_name:
                writer = csv.writer(file_name)
                writer.writerows(hardbit)

        self.count += 1
        return len(input_items[0])

