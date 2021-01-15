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

        self.tz1 = np.zeros(4, dtype=complex)
        self.tz1[0] = 1.
        self.tz1[1] = -1.
        self.tz1[2] = 3.
        self.tz1[3] = 5.
        self.tmpZ = (1j * (2.0 * np.pi / 8.)) * self.tz1

        self.CDAT = np.exp(self.tmpZ)

        self.modbit = [[0, 0], [0, 1], [1, 0], [1, 1]]

        # self.K = 0.707106781186547
        self.K = 1.414213562373095
        self.directory_name = directory_name
        self.count = 0

    def write_to_file(self, file_name, var_to_write, var_string):
        f = open(file_name, 'a')
        f.write('\n' + var_string + str(var_to_write))
        f.close()

    def work(self, input_items, output_items):
        in0 = input_items[0]
        # print("In", in0[100:110])

        DataIn = in0
        DataIn = DataIn[:, np.newaxis]
        # print("Shape of DataIn", DataIn.shape)
        print("Work call", self.count)
        print("DataIn", DataIn[100:110, 0])


        NumBits = len(DataIn) * 2
        NumSymbs = len(DataIn)

        llrp0 = np.zeros(NumBits, dtype=float)
        llrp1 = np.zeros(NumBits, dtype=float)
        CDATExt = np.tile(self.CDAT, (NumSymbs, 1))
        DataExt = np.tile(DataIn, (1, 4))
        

        Zobs = DataExt - CDATExt
        dminind = np.argmin(abs(Zobs), 1)
        dmin = np.min(abs(Zobs), 1)
        dz = self.CDAT[dminind]
        ez = np.zeros(len(DataIn), dtype=complex)
        # ez = self.DataIn-self.CDATExt[:][dminind]

        for loopq1 in list(range(len(DataIn))):
            # print "DataIN[loopq1]: ", type(DataIn[loopq1][0])
            # print "CDATExt[loopq1]: ", type(CDATExt[loopq1][dminind[loopq1]])
            # np.issubdtype(DataIn, float)
            # ez[loopq1] = DataIn[loopq1][0] - CDATExt[loopq1][dminind[loopq1]]
            ez[loopq1] = DataIn[loopq1][0] - dz[loopq1]

            # ez[loopq1] = np.subtract(DataIn[loopq1], CDATExt[loopq1][dminind[loopq1]])

        sigmaest0 = 0.7071067811865476 * np.mean(np.abs(dmin))
        dfact = 1.0 / (sigmaest0 * sigmaest0)

        for LoopK in list(range(len(DataIn))):
            if DataIn[LoopK].real >= 0 and DataIn[LoopK].imag >= 0:
                llrp0[2 * LoopK] = -0.5 * dfact * (abs(ez[LoopK].real))
                llrp1[2 * LoopK] = -0.5 * dfact * (self.K - abs(ez[LoopK].real))
                llrp0[2 * LoopK + 1] = -0.5 * dfact * abs(ez[LoopK].imag)
                llrp1[2 * LoopK + 1] = -0.5 * dfact * (self.K - abs(ez[LoopK].imag))
            elif DataIn[LoopK].real <= 0 and DataIn[LoopK].imag >= 0:
                llrp1[2 * LoopK] = -0.5 * dfact * (abs(ez[LoopK].real))
                llrp0[2 * LoopK] = -0.5 * dfact * (self.K - abs(ez[LoopK].real))
                llrp0[2 * LoopK + 1] = -0.5 * dfact * abs(ez[LoopK].imag)
                llrp1[2 * LoopK + 1] = -0.5 * dfact * (self.K - abs(ez[LoopK].imag))
            elif DataIn[LoopK].real <= 0 and DataIn[LoopK].imag <= 0:
                llrp1[2 * LoopK] = -0.5 * dfact * (abs(ez[LoopK].real))
                llrp0[2 * LoopK] = -0.5 * dfact * (self.K - abs(ez[LoopK].real))
                llrp1[2 * LoopK + 1] = -0.5 * dfact * abs(ez[LoopK].imag)
                llrp0[2 * LoopK + 1] = -0.5 * dfact * (self.K - abs(ez[LoopK].imag))
            elif DataIn[LoopK].real >= 0 and DataIn[LoopK].imag <= 0:
                llrp0[2 * LoopK] = -0.5 * dfact * (abs(ez[LoopK].real))
                llrp1[2 * LoopK] = -0.5 * dfact * (self.K - abs(ez[LoopK].real))
                llrp1[2 * LoopK + 1] = -0.5 * dfact * abs(ez[LoopK].imag)
                llrp0[2 * LoopK + 1] = -0.5 * dfact * (self.K - abs(ez[LoopK].imag))

        softbit0 = np.zeros((NumBits, 1), dtype=float)
        softbit1 = np.zeros((NumBits, 1), dtype=float)
        # hardbit = np.zeros((NumBits, 1), dtype=int)
        # xe = [x for x in self.llrp0 if x % 2 == 0]
        # xo = [x for x in self.llrp0 if x % 2 != 0]

        '''for loopq2 in list(range(0, len(softbit0), 2)):
            softbit0[loopq2 + 1] = llrp0[loopq2]
            softbit0[loopq2] = llrp0[loopq2 + 1]
            softbit1[loopq2 + 1] = llrp1[loopq2]
            softbit1[loopq2] = llrp1[loopq2 + 1]

        posneg_data = np.sign(softbit1)

        bit_data = np.zeros(len(posneg_data), dtype=int)
        for data_index in range(len(posneg_data)):
            if posneg_data[data_index] == -1:
                bit_data[data_index] = 0
            if posneg_data[data_index] == 1 or posneg_data[data_index] == 0:
                bit_data[data_index] = 1'''
        softbit0 = llrp0
        softbit1 = llrp1

        

        # print("Soft bit 0", softbit0)
        # print("Soft bit 1", softbit1)

        xyz = np.array(0.5 * (np.sign(softbit1 - softbit0) + 1.))
        hardbit = xyz.astype(int)
        hardbit = hardbit[:, np.newaxis]

        if self.count == 30:
            var_mat = dict()
            var_mat['DataExt'] = DataExt
            var_mat['softbit0'] = softbit0
            var_mat['softbit1'] = softbit1
            var_mat['hardbit'] = hardbit
            scipy.io.savemat('/home/tayloreisman/Desktop/bit_recov.mat', var_mat)

        if self.diagnostics == 1:

            date_time = datetime.datetime.now().strftime('%Y_%m_%d_%Hh_%Mm')
            file_name0 = self.directory_name + 'softbit0_' + date_time + '.pckl'
            file_name1 = self.directory_name + 'softbit1_' + date_time + '.pckl'

            f = open(file_name0, 'wb')
            pickle.dump(softbit0, f, protocol=2)
            f.close()

            f = open(file_name1, 'wb')
            pickle.dump(softbit1, f, protocol=2)
            f.close()

            file_name = open(self.directory_name + 'rx_data.csv', 'a')
            with file_name:
                writer = csv.writer(file_name)
                writer.writerows(hardbit)



        self.count += 1
        return len(input_items[0])

