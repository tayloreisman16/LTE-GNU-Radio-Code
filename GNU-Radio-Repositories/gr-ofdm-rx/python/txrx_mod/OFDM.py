import numpy as np


class OFDM:
    def __init__(self, len_CP, num_used_bins, modulation_type, NFFT, delta_f):
        self.len_CP = len_CP
        self.num_used_bins = num_used_bins
        self.modulation_type = modulation_type
        self.NFFT = NFFT
        self.bin_spacing = delta_f

        if modulation_type == 'BPSK':
            self.num_bits_bin = 1
        elif modulation_type == 'QPSK':
            self.num_bits_bin = 2
