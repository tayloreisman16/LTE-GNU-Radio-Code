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
import pickle
import datetime
from gnuradio import gr

file_name = 'chan_est_time.pckl'

class SynchAndChanEst(gr.sync_block):
    """
    docstring for block SynchAndChanEst
    """
    def __init__(self, num_ofdm_symb, nfft, cp_len, num_synch_bins, synch_dat, num_data_bins, SNR, directory_name, file_name_cest, diagnostics):
        self.num_ofdm_symb = num_ofdm_symb
        self.nfft = nfft
        self.cp_len = cp_len

        self.num_synch_bins = num_synch_bins

        # # Zadoff Chu Generation
        # self.prime_num = 23
        #
        # if self.num_synch_bins % 2 == 0:
        #     seq0 = np.array(range(self.num_synch_bins))
        #     self.zadoff_chu = np.exp(-1j * (2 * np.pi / self.num_synch_bins) * self.prime_num * seq0 * seq0 / 2)
        #
        # elif self.num_synch_bins % 2 == 1:
        #     seq0 = np.array(range(self.num_synch_bins))
        #     self.zadoff_chu = np.exp(
        #         -1j * (2 * np.pi / self.num_synch_bins) * self.prime_num * seq0 * (seq0 + 1) / 2)

        self.synch_bins_used_N = (list(range(-int(self.num_synch_bins / 2), 0, 1)) +
                               list(range(1, int(self.num_synch_bins / 2) + 1, 1)))

        self.synch_bins_used_P = list((np.array(self.synch_bins_used_N) + self.nfft) % self.nfft)

        self.L_synch = len(self.synch_bins_used_P)
        self.synch_dat = synch_dat # example [1, 1] or [2, 1]


        self.M = [self.synch_dat[0], self.num_synch_bins]
        self.MM = np.prod(self.M)

        # Zadoff Chu Generation
        self.p = 37
        if self.num_synch_bins % 2 == 0:
            tmp0 = np.array(range(self.MM))
            xx = tmp0 * tmp0
        elif self.num_synch_bins % 2 == 1:
            tmp0 = np.array(range(self.MM))
            xx = tmp0 * (tmp0+1)


        tmpvsynch = [(-1j * (2 * np.pi / self.MM) * self.p / 2.0) * kk for kk in xx]
        self.zadoff_chu = np.exp(tmpvsynch)
        # print(self.zadoff_chu.shape)
        # print(self.M)
        # print(self.synch_bins_used_P)

        self.num_data_bins = num_data_bins

        self.bins_used_N = (list(range(-int(self.num_data_bins / 2), 0, 1)) +
                          list(range(1, int(self.num_data_bins / 2) + 1, 1)))
        self.bins_used_P = list((np.array(self.bins_used_N) + self.nfft) % self.nfft)
        # print("Bins used P", self.bins_used_P)

        self.cor_obs = -1

        self.del_mat_exp = np.tile(np.exp((1j * (2.0 * np.pi / self.nfft)) * (
            np.outer(list(range(self.cp_len + 1)), list(self.synch_bins_used_P)))), (1, self.M[0]))

        self.stride_val = self.cp_len - 1
        self.start_samp = self.cp_len
        self.rx_b_len = self.nfft + self.cp_len

        self.max_num_corr = 100

        self.time_synch_ref = np.zeros((self.max_num_corr, 3))
        self.est_chan_time = np.zeros((self.max_num_corr, self.nfft), dtype=complex)
        self.est_synch_freq = np.zeros((self.max_num_corr, len(self.zadoff_chu)), dtype=complex)

        self.est_chan_freq_P = np.zeros((self.max_num_corr, self.nfft), dtype=complex)
        self.est_data_freq = np.zeros((self.max_num_corr, self.num_data_bins), dtype=complex)

        self.rx_data_time = np.zeros((1, self.M[0] * self.nfft), dtype=complex) # first dimension = no of antennas
        self.synchdat00 = np.zeros((1, self.M[0] * self.num_synch_bins), dtype=complex)

        self.del_mat = []
        self.eq_gain = []
        self.eq_gain_ext = []
        self.eq_gain_q = []

        self.SNR = SNR
        self.count = 0

        self.diagnostics = diagnostics
        self.directory_name = directory_name
        self.file_name_cest = file_name_cest

        gr.sync_block.__init__(self,
            name="SynchAndChanEst",
            in_sig=[np.complex64],
            out_sig=[np.complex64])


    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]


        n_trials = int(np.around(len(in0) / self.stride_val))

        for P in list(range(n_trials)):
            if self.M[0] * self.rx_b_len + P * self.stride_val + self.nfft + self.start_samp < len(in0):
                for LL in list(range(self.M[0])):
                    aaa = self.rx_b_len * LL + P * self.stride_val + self.start_samp
                    bbb = self.rx_b_len * LL + P * self.stride_val + self.start_samp + self.nfft
                    self.rx_data_time[0][LL * self.nfft:(LL + 1) * self.nfft] = in0[aaa:bbb]

                tmp_1vec = np.zeros((self.M[0], self.nfft), dtype=complex)
                for LL in list(range(self.M[0])):
                    tmp_1vec[LL][:] = np.fft.fft(self.rx_data_time[0][LL * self.nfft:(LL + 1) * self.nfft], self.nfft)
                    self.synchdat00[0][LL * self.num_synch_bins:(LL + 1) * self.num_synch_bins] = tmp_1vec[LL][self.synch_bins_used_P]

                synchdat0 = np.reshape(self.synchdat00, (1, self.num_synch_bins * self.M[0]))
                # print(synchdat0.shape)
                p_est = np.sqrt(len(synchdat0[0]) / sum(np.multiply(synchdat0[0][:], np.conj(synchdat0[0][:]))))

                synchdat = [p_est * kk for kk in synchdat0]
                tmp_2mat = np.matmul(self.del_mat_exp, np.diag(synchdat[0]))
                self.del_mat = np.matmul(tmp_2mat, np.conj(self.zadoff_chu))

                dmax_ind = np.argmax((abs(self.del_mat)))
                dmax_val = np.max((abs(self.del_mat)))

                if dmax_val > 0.4 * len(synchdat[0]):
                    tim_synch_ind = self.time_synch_ref[max(self.cor_obs, 0)][0]
                    if ((P * self.stride_val + self.start_samp - tim_synch_ind > 2 * self.cp_len + self.nfft)
                            or self.cor_obs == -1):


                        self.cor_obs += 1

                        self.time_synch_ref[self.cor_obs][0] = P * self.stride_val + self.start_samp
                        self.time_synch_ref[self.cor_obs][1] = dmax_ind
                        self.time_synch_ref[self.cor_obs][2] = int(dmax_val)

                        del_vec = self.del_mat_exp[dmax_ind][:]
                        data_recov = np.matmul(np.diag(del_vec), synchdat[0])

                        zcwn = [(1.0 / self.SNR) + qq for qq in np.ones(len(self.zadoff_chu))]
                        tmp_v1 = np.divide(np.matmul(np.diag(data_recov), np.conj(self.zadoff_chu)), zcwn)

                        chan_est00 = np.reshape(tmp_v1, (self.M[0], self.L_synch))
                        chan_est = np.sum(chan_est00, axis=0) / float(self.M[0])

                        chan_est1 = np.zeros((1, self.nfft), dtype=np.complex)
                        chan_est1[0][self.synch_bins_used_P] = chan_est
                        self.est_chan_freq_P[self.cor_obs][:] = chan_est1[0][:]

                        chan_est_tim = np.fft.ifft(chan_est1, self.nfft)

                        if self.diagnostics == 1:

                            date_time = datetime.datetime.now().strftime('%Y_%m_%d_%Hh_%Mm')

                            f = open(str(self.directory_name) + str(date_time) + str(self.file_name_cest) + '.pckl', 'wb')
                            pickle.dump(chan_est_tim, f, protocol=2)
                            f.close()

                        self.est_chan_time[self.cor_obs][0:self.nfft] = chan_est_tim[0][0:self.nfft]
                        chan_mag = np.matmul(np.diag(chan_est), np.conj(chan_est))
                        eq_gain_0 = [1.0 / self.SNR + vv for vv in chan_mag]

                        self.eq_gain = np.divide(np.conj(chan_est), eq_gain_0)
                        self.eq_gain_ext = np.tile(self.eq_gain, self.M[0])
                        self.est_synch_freq[self.cor_obs][:] = np.matmul(np.diag(self.eq_gain_ext), data_recov)
        # <+signal processing here+>

        for P in list(range(self.cor_obs+1)):

            if self.time_synch_ref[P][0] + self.M[0] * self.rx_b_len + self.nfft - 1 <= len(in0):
                data_ptr = int(self.time_synch_ref[P][0] + self.M[0]*self.rx_b_len)
                print("data pointer", data_ptr)

                data_buff_time = in0[data_ptr: data_ptr + self.nfft]

                t_vec = np.fft.fft(data_buff_time, self.nfft)
                # print(self.bins_used_P)
                freq_data_0 = t_vec[self.bins_used_P]
                p_est0 = np.sqrt(len(freq_data_0)/(np.dot(freq_data_0, np.conj(freq_data_0))))

                data_recov_0 = freq_data_0 * p_est0

                arg_val = ([((1j * (2 * np.pi / self.nfft)) *
                            self.time_synch_ref[P][1]) * kk for kk in self.bins_used_P])

                data_recov_z = np.matmul(np.diag(data_recov_0), np.exp(arg_val))

                chan_est_dat = self.est_chan_freq_P[P][self.bins_used_P]

                chan_mag_z = np.matmul(np.diag(chan_est_dat), np.conj(chan_est_dat))
                eq_gain_z = [1.0 / self.SNR + vv for vv in chan_mag_z]
                self.eq_gain_q = np.divide(np.conj(chan_est_dat), eq_gain_z)

                self.est_data_freq[P][:] = np.matmul(np.diag(self.eq_gain_q), data_recov_z)
        # print("P", P)
        # print("Data Est Freq Shape", self.est_data_freq.shape)
        corr_size = self.num_ofdm_symb/sum(self.synch_dat)
        # print(corr_size)
        # print("Data Est Freq", self.est_data_freq)
        # print(self.est_data_freq[0:corr_size][:])
        data_out = np.reshape(self.est_data_freq[0:corr_size][:], (1, corr_size * np.size(self.est_data_freq, 1)))
        # print(data_out.shape)

        if self.count > 0:
            # out[0:data_out.shape[1]] = data_out
            out[0:np.size(data_out, 1)] = data_out[:, np.newaxis]
            # print(out[0:np.size(data_out, 1)])
        self.count += 1
        self.cor_obs = 0
        return len(output_items[0])

