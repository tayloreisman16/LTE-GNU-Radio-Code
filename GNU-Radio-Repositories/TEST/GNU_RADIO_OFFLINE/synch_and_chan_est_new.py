from abc import ABC

import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
from gnuradio import gr


class SynchAndChanEst(gr.sync_block):

    def __init__(self, num_ant_txrx, num_ofdm_symb, nfft, cp_len, num_synch_bins, synch_dat, num_data_bins, snr, directory_name,
                 file_name_cest, diagnostics, genie):
        gr.sync_block.__init__(self,
                               name="SynchAndChanEst",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.num_ant_txrx = num_ant_txrx
        self.num_symbols = num_ofdm_symb
        self.nfft = nfft
        self.cp_len = cp_len
        self.snr = snr
        self.rx_buff_len = self.nfft + self.cp_len
        self.diagnostic = diagnostics
        self.genie = genie
        self.ref_sigs = 0.0


        self.synch_data = synch_dat
        num_synchdata_patterns = int(np.ceil(self.num_symbols / sum(self.synch_data)))
        num_symbols = sum(self.synch_data) * num_synchdata_patterns
        symbol_pattern0 = np.concatenate((np.zeros(self.synch_data[0]), np.ones(self.synch_data[1])))
        self.symbol_pattern = np.tile(symbol_pattern0, num_synchdata_patterns)

        self.channel_band = 960e3 * 0.97
        self.fs = self.channel_band
        self.bin_spacing = 15e3

        self.UG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        self.SG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        self.VG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        self.U = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.num_used_bins), dtype=complex)
        self.S = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.num_used_bins), dtype=complex)
        self.V = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.num_used_bins), dtype=complex)

        self.stream_size = 1
        #  Genie Variables
        self.max_impulse = self.nfft
        self.genie_chan_time = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.max_impulse), dtype=complex)

        self.num_bins0 = np.floor(self.channel_band / self.bin_spacing)
        self.num_bins1 = 4 * np.floor(self.num_bins0 / 4)
        self.all_bins = np.array(
            list(range(-int(self.num_bins1 / 2), 0)) + list(range(1, int(self.num_bins1 / 2) + 1)))
        self.used_bins = (self.nfft + self.all_bins)


        ref_bins0 = np.random.randint(1, int(self.num_bins1 / 2) + 1, size=int(np.floor(self.num_bins1 * self.ref_sigs / 2)))
        ref_bins = np.unique(ref_bins0)
        ref_only_bins = np.sort(np.concatenate((-ref_bins, ref_bins)))
        data_only_bins = np.setdiff1d(self.all_bins, ref_only_bins)
        self.data_only_bins = (self.nfft + data_only_bins) % self.nfft
        self.ref_only_bins = (self.nfft + ref_only_bins) % self.nfft
        self.used_bins_data = (self.nfft + self.all_bins) % self.nfft

        self.num_used_bins = num_data_bins
        self.used_bins_synch = num_synch_bins
        self.max_impulse = self.nfft

        self.channel_time = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.max_impulse), dtype=complex)
        self.channel_freq = np.zeros((self.num_ant_txrx, self.num_ant_txrx, int(self.nfft)), dtype=complex)
        self.h_f = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.num_used_bins))

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
