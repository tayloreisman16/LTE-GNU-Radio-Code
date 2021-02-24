import numpy as np


class SystemModel:
    def __init__(self, SDR_profile):

        # Varying parameters for each SDR profile case
        self.system_scenario = SDR_profile['system_scenario']
        self.diagnostic = SDR_profile['diagnostic']
        self.wireless_channel = SDR_profile['wireless_channel']
        self.channel_band = SDR_profile['channel_band']
        self.bin_spacing = SDR_profile['bin_spacing']
        self.channel_profile = SDR_profile['channel_profile']
        self.CP_type = SDR_profile['CP_type']
        self.num_ant_txrx = SDR_profile['num_ant_txrx']
        self.param_est = SDR_profile['param_est']
        self.MIMO_method = SDR_profile['MIMO_method']  # Make this 0 (or something) for single antenna
        self.SNR = SDR_profile['SNR']
        self.ebno_db = SDR_profile['ebno_db']
        self.num_symbols = SDR_profile['num_symbols']
        self.stream_size = SDR_profile['stream_size']

        # Mostly unchanged paramters
        self.sig_datatype = 'Complex'
        self.phy_chan = 'Data'
        self.modulation_type = 'QPSK'
        self.bits_per_bin = 2
        self.synch_data = np.array([1, 3])
        self.SNR_type = 'Digital'  # Digital, Analog
        self.ref_sigs = 0.0

        # calculated stuff

        self.NFFT = 2**(np.ceil(np.log2(round(self.channel_band / self.bin_spacing))))
        self.num_bins0 = np.floor(self.channel_band / self.bin_spacing)
        self.num_synch_bins = self.NFFT - 2
        self.fs = self.bin_spacing * self.NFFT

        self.CP_samp = 16
        self.plot_num = 1
