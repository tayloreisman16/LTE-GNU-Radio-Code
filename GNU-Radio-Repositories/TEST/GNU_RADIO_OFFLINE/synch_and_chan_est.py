#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 UTSA.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
import pickle
# import datetime
import matplotlib.pyplot as plt
from gnuradio import gr


class SynchAndChanEst(gr.sync_block):
    def __init__(self, num_ofdm_symb, nfft, cp_len,
                 num_synch_bins, synch_dat, num_data_bins, channel, snr, scale_factor_gate, directory_name,
                 file_name_cest, diagnostics, genie):
        self.num_ofdm_symb = num_ofdm_symb
        self.nfft = nfft
        self.cp_len = cp_len
        self.genie = genie
        self.scale_factor_gate = scale_factor_gate
        self.channel = channel

        self.channel_band = 960e3 * 0.97
        self.fs = self.channel_band
        self.bin_spacing = 15e3
        
        self.num_synch_bins = num_synch_bins

        self.synch_bins_used_N = (list(range(-int(self.num_synch_bins / 2), 0, 1)) +
                                  list(range(1, int(self.num_synch_bins / 2) + 1, 1)))

        self.synch_bins_used_P = list((np.array(self.synch_bins_used_N) + self.nfft) % self.nfft)

        self.L_synch = len(self.synch_bins_used_P)
        self.synch_dat = synch_dat  # example [1, 1] or [2, 1]
        self.num_data_symbs_blk = self.synch_dat[1]
        # self.symb_blk_length = sum(self.sync_dat) * self.rx_b_len
        # self.data_blk_length = self.num_data_symbs * self.rx_b_len
        self.M = [self.synch_dat[0], self.num_synch_bins]
        self.MM = int(np.prod(self.M))

        # Zadoff Chu Generation
        self.p = 23

        x0 = np.array(range(0, int(self.MM)))
        x1 = np.array(range(1, int(self.MM) + 1))
        if self.MM % 2 == 0:
            self.zadoff_chu = np.exp(-1j * (2 * np.pi / self.MM) * self.p * (x0**2 / 2))
        else:
            self.zadoff_chu = np.exp(-1j * (2 * np.pi / self.MM) * self.p * (x0 * x1) / 2)

        plt.plot(self.zadoff_chu.real)
        plt.plot(self.zadoff_chu.imag)
        plt.title('Reference Zadoff-Chu Signal')
        plt.show()

        self.num_data_bins = num_data_bins

        self.bins_used_N = (list(range(-int(self.num_data_bins / 2), 0, 1)) +
                            list(range(1, int(self.num_data_bins / 2) + 1, 1)))
        self.bins_used_P = list((np.array(self.bins_used_N) + self.nfft) % self.nfft)

        self.corr_obs = -1

        self.del_mat_exp = np.tile(np.exp((1j * (2.0 * np.pi / self.nfft)) * (
            np.outer(list(range(self.cp_len + 1)), list(self.synch_bins_used_P)))), (1, self.M[0]))

        self.stride_val = 1
        self.start_samp = self.cp_len
        self.rx_b_len = self.nfft + self.cp_len

        self.max_num_corr = 100

        self.time_synch_ref = np.zeros(3)
        self.est_chan_time = np.zeros((num_ofdm_symb, self.nfft), dtype=complex)
        self.est_synch_freq = np.zeros((num_ofdm_symb, len(self.zadoff_chu)), dtype=complex)

        self.est_chan_freq_P = np.zeros((num_ofdm_symb, self.nfft), dtype=complex)
        self.est_data_freq = np.zeros((num_ofdm_symb, self.num_data_bins), dtype=complex)

        self.rx_data_time = np.zeros((1, self.M[0] * self.nfft), dtype=complex)  # first dimension = no of antennas
        self.synchdat00 = np.zeros((1, self.M[0] * self.num_synch_bins), dtype=complex)

        self.del_mat = []
        self.eq_gain = []
        self.eq_gain_ext = []
        self.eq_gain_q = []

        self.SNR = snr
        self.SNR_lin = 10 ** (self.SNR / 20)
        self.count = 0

        self.diagnostic = diagnostics
        self.directory_name = directory_name
        self.file_name_cest = file_name_cest
        
        self.num_ant_txrx = 1  # Hard Coded
        
        # Genie Channel Variables
        if self.genie == 1:
            self.max_impulse = self.nfft
            self.genie_chan_time = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.max_impulse), dtype=complex)
            
            num_bins0 = np.floor(self.channel_band / self.bin_spacing)
            num_bins1 = 4 * np.floor(num_bins0 / 4)
            all_bins = np.array(list(range(-int(num_bins1 / 2), 0)) + list(range(1, int(num_bins1 / 2) + 1)))
            self.used_bins = (self.nfft + all_bins)

        gr.sync_block.__init__(self,
                               name="SynchAndChanEst",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])

    def give_genie_chan(self):
        h = np.zeros((self.num_ant_txrx, self.num_ant_txrx), dtype=object)
        channel_time = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.max_impulse), dtype=complex)
        channel_freq = np.zeros((self.num_ant_txrx, self.num_ant_txrx, int(self.nfft)), dtype=complex)
        if self.num_ant_txrx == 1:
            if self.channel == 'Ideal':
                h[0, 0] = np.array([1])
            else:
                h[0, 0] = np.array([0.3977, 0.7954 - 0.3977j, -0.1988, 0.0994, -0.0398])

        for rx in range(self.num_ant_txrx):
            for tx in range(self.num_ant_txrx):
                channel_time[rx, tx, 0:len(h[rx, tx])] = h[rx, tx] / np.linalg.norm(h[rx, tx])
                channel_freq[rx, tx, :] = np.fft.fft(channel_time[rx, tx, 0:len(h[rx, tx])], self.nfft)

        genie_chan_time = channel_time
        return genie_chan_time

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        n_trials = int(np.around(len(in0) / self.stride_val))
        n_unique_symb = int(np.floor(len(in0) / self.rx_b_len))
        n_data_symb = int(n_unique_symb * (self.synch_dat[1] / sum(self.synch_dat)))

        for P in list(range(n_trials)):
            if self.M[0] * self.rx_b_len + P * self.stride_val + self.nfft + self.start_samp < len(in0):
                for LL in list(range(self.M[0])):
                    aaa = self.rx_b_len * LL + P * self.stride_val + self.start_samp
                    bbb = self.rx_b_len * LL + P * self.stride_val + self.start_samp + self.nfft
                    self.rx_data_time[0][LL * self.nfft:(LL + 1) * self.nfft] = in0[aaa:bbb]

                tmp_1vec = np.zeros((self.M[0], self.nfft), dtype=complex)
                for LL in list(range(self.M[0])):
                    tmp_1vec[LL][:] = np.fft.fft(self.rx_data_time[0][LL * self.nfft:(LL + 1) * self.nfft], self.nfft)
                    self.synchdat00[0][LL * self.num_synch_bins:(LL + 1) * self.num_synch_bins] = \
                        tmp_1vec[LL][self.synch_bins_used_P]

                synchdat0 = np.reshape(self.synchdat00, (1, self.num_synch_bins * self.M[0]))
                p_est = np.sqrt(len(synchdat0[0]) / sum(np.multiply(synchdat0[0][:], np.conj(synchdat0[0][:]))))

                synchdat = [p_est * kk for kk in synchdat0]
                tmp_2mat = np.matmul(self.del_mat_exp, np.diag(synchdat[0]))
                self.del_mat = np.matmul(tmp_2mat, np.conj(self.zadoff_chu))

                dmax_ind = np.argmax((abs(self.del_mat)))
                dmax_val = np.max((abs(self.del_mat)))

                if dmax_val > self.scale_factor_gate * len(synchdat[0]):
                    tim_synch_ind = self.time_synch_ref[0]
                    if ((P * self.stride_val + self.start_samp - tim_synch_ind > 2 * self.cp_len + self.nfft)
                            or self.corr_obs == -1):

                        self.corr_obs += 1

                        self.time_synch_ref[0] = P * self.stride_val + self.start_samp
                        self.time_synch_ref[1] = dmax_ind
                        self.time_synch_ref[2] = int(dmax_val)

                        del_vec = self.del_mat_exp[dmax_ind][:]
                        data_recov = np.matmul(np.diag(del_vec), synchdat[0])

                        zcwn = [(1.0 / self.SNR_lin) + qq for qq in np.ones(len(self.zadoff_chu))]
                        tmp_v1 = np.divide(np.matmul(np.diag(data_recov), np.conj(self.zadoff_chu)), zcwn)

                        chan_est00 = np.reshape(tmp_v1, (self.M[0], self.L_synch))
                        chan_est = np.sum(chan_est00, axis=0) / float(self.M[0])

                        chan_est1 = np.zeros((1, self.nfft), dtype=np.complex)
                        chan_est1[0][self.synch_bins_used_P] = chan_est
                        self.est_chan_freq_P[self.corr_obs][:] = chan_est1[0][:]

                        if self.diagnostic == 1 and self.count == 0 and self.genie == 1:
                            chan_q = self.give_genie_chan()
                            xax = np.array(range(0, self.nfft - 2))
                            yax1 = 20 * np.log10(abs(chan_est1))
                            yax2 = 20 * np.log10(abs(np.fft.fft(chan_q, self.nfft)))
                            ypred = [x for i, x in enumerate(yax1[0, 1:]) if i != 31]
                            ygeni = [x for i, x in enumerate(yax2[0, 0, 1:]) if i != 31]

                            plt.plot(xax, ypred, 'r')
                            plt.plot(xax, ygeni, 'b')
                            plt.show()

                        chan_est_tim = np.fft.ifft(chan_est1, self.nfft)

                        if self.diagnostic == 1:

                            # date_time = datetime.datetime.now().strftime('%Y_%m_%d_%Hh_%Mm')

                            f = open(str(
                                self.directory_name) + '_' + str(self.file_name_cest), 'wb')
                            pickle.dump(chan_est_tim, f, protocol=2)
                            f.close()

                        self.est_chan_time[self.corr_obs][0:self.nfft] = chan_est_tim[0][0:self.nfft]
                        chan_mag = np.matmul(np.diag(chan_est), np.conj(chan_est))
                        eq_gain_0 = [1.0 / self.SNR + vv for vv in chan_mag]

                        self.eq_gain = np.divide(np.conj(chan_est), eq_gain_0)
                        self.eq_gain_ext = np.tile(self.eq_gain, self.M[0])
                        self.est_synch_freq[self.corr_obs][:] = np.matmul(np.diag(self.eq_gain_ext), data_recov)
                        break

        for P in list(range(n_unique_symb)[::sum(self.synch_dat)]):
            data_ptr = int(self.time_synch_ref[0] + self.M[0] * self.rx_b_len * (P + 1))
            if self.time_synch_ref[0] + self.M[0] * self.rx_b_len * (P + 1) + self.nfft - 1 <= len(in0):

                for N in range(self.synch_dat[1]):
                    start = data_ptr + self.rx_b_len * N
                    end = data_ptr + self.rx_b_len * N + self.nfft
                    data_buff_time = in0[start: end]

                    t_vec = np.fft.fft(data_buff_time, self.nfft)

                    freq_data_0 = t_vec[self.bins_used_P]
                    p_est0 = np.sqrt(len(freq_data_0)/(np.dot(freq_data_0, np.conj(freq_data_0))))

                    data_recov_0 = freq_data_0 * p_est0

                    arg_val = ([((1j * (2 * np.pi / self.nfft)) *
                                 self.time_synch_ref[1]) * kk for kk in self.bins_used_P])

                    data_recov_z = np.matmul(np.diag(data_recov_0), np.exp(arg_val))

                    chan_est_dat = self.est_chan_freq_P[0][self.bins_used_P]

                    chan_mag_z = np.matmul(np.diag(chan_est_dat), np.conj(chan_est_dat))
                    eq_gain_z = [1.0 / self.SNR_lin + vv for vv in chan_mag_z]
                    self.eq_gain_q = np.divide(np.conj(chan_est_dat), eq_gain_z)

                    self.est_data_freq[P + N][:] = np.matmul(np.diag(self.eq_gain_q), data_recov_z)
        data_demod = np.delete(self.est_data_freq, list(range(3, self.est_data_freq.shape[0], sum(self.synch_dat))), axis=0)
        if self.diagnostic == 1:
            plt.plot(self.est_data_freq[0][:].real, self.est_data_freq[0][:].imag, 'o')
            plt.show()

        data_out = np.reshape(data_demod[:][:], (1, n_data_symb * np.size(data_demod, 1)))

        if self.count > 0:
            out[0:np.size(data_out, 1)] = data_out[:, np.newaxis]

        self.count += 1
        self.corr_obs = 0
        return len(output_items[0])
