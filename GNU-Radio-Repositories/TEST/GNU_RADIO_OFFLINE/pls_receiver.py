from numpy import conj, sqrt, convolve, zeros, var, diag, angle, dot, exp, array, real, argwhere, complex64
from numpy.random import normal
from numpy.fft import fft
from numpy.linalg import svd


class PhyLayerSecReceiver:
    def __init__(self, bandwidth, bin_spacing, num_ant, bit_codebook, NFFT, CP, synch, symb_pattern, total_num_symb, num_data_symb, num_synch_symb,
                 snr_db, snr_type):
        gr.sync_block.__init__(self,
                               name="SynchAndChanEst",
                               in_sig=[complex64],
                               out_sig=[complex64])
        self.bandwidth = bandwidth
        self.bin_spacing = bin_spacing
        self.num_ant = num_ant
        self.bit_codebook = bit_codebook  # TODO: Add function to calculate codebook!

        self.NFFT = NFFT
        self.CP = CP
        self.OFDMsymb_len = self.NFFT + self.CP
        self.num_data_bins = num_data_bins

        self.used_data_bins = used_data_bins
        self.subband_size = self.num_ant

        self.num_subbands = pls_params.num_subbands
        self.num_PMI = self.num_subbands

        self.synch = synch
        self.symb_pattern = symb_pattern

        self.channel_time = channel_time

        self.total_num_symb = total_num_symb
        self.total_symb_len = self.total_num_symb*self.OFDMsymb_len

        self.num_data_symb = num_data_symb
        self.num_synch_symb = num_synch_symb
        self.SNRdB = snr_db
        self.SNR_type = snr_type