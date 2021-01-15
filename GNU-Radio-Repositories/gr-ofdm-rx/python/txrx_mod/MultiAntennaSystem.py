import numpy as np
import matplotlib.pyplot as plt


class MultiAntennaSystem:
    def __init__(self, OFDM_data, num_ant_txrx, MIMO_method, all_bins, num_symbols, symbol_pattern, fs, channel_profile, diagnostic, wireless_channel, stream_size, data_only_bins, ref_only_bins):

        self.OFDM_data = OFDM_data  # OFDM class object
        self.NFFT = int(self.OFDM_data.NFFT)
        self.len_CP = int(self.OFDM_data.len_CP)
        self.num_ant_txrx = num_ant_txrx
        self.MIMO_method = MIMO_method
        self.all_bins = all_bins
        self.used_bins = (self.NFFT + all_bins) % self.NFFT

        self.num_symbols = num_symbols
        self.symbol_pattern = symbol_pattern
        self.fs = fs
        self.channel_profile = channel_profile
        self.diagnostic = diagnostic
        self.wireless_channel = wireless_channel
        self.stream_size = stream_size

        self.data_only_bins = (self.NFFT + data_only_bins) % self.NFFT  # positive bin indices only
        self.ref_only_bins = (self.NFFT + ref_only_bins) % self.NFFT  # positive bin indices only

        self.num_used_bins = self.OFDM_data.num_used_bins
        self.max_impulse = self.NFFT  # Maximum number of channel taps (time domain)

        # This buffer does not include Cyclic Prefix
        self.buffer_data_tx = np.zeros((self.num_ant_txrx, int(self.NFFT * self.num_symbols)), dtype=complex)
        # This buffer does not include Cyclic Prefix
        self.buffer_data_rx = np.zeros((self.num_ant_txrx, int(self.NFFT * self.num_symbols)), dtype=complex)

        # Total length of all symbols put together
        self.symb_len_total = int((self.NFFT + self.len_CP) * self.num_symbols)

        # This buffer includes Cyclic Prefix (time domain)
        self.buffer_data_tx_time = np.zeros((self.num_ant_txrx, self.symb_len_total), dtype=complex)
        # This buffer includes Cyclic Prefix (time domain)
        self.buffer_data_rx_time = np.zeros((self.num_ant_txrx, self.symb_len_total + self.max_impulse - 1), dtype=complex)

        # print(self.buffer_data_tx_time.shape, self.buffer_data_rx_time.shape)

        # channelmattim, channelmatfreq
        self.channel_time = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.max_impulse), dtype=complex)
        self.channel_freq = np.zeros((self.num_ant_txrx, self.num_ant_txrx, int(self.NFFT)), dtype=complex)

        # Channel matrices after FFT for each used bin
        self.h_f = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.num_used_bins), dtype=complex)

        # print(self.h_f.shape)

        if self.MIMO_method == 'SpMult':
            # Placeholders for SVD. Spatial multiplexing only.
            self.U = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.num_used_bins), dtype=complex)
            self.S = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.num_used_bins), dtype=complex)
            self.V = np.zeros((self.num_ant_txrx, self.num_ant_txrx, self.num_used_bins), dtype=complex)

        test_case = 0
        h = np.zeros((self.num_ant_txrx, self.num_ant_txrx), dtype=object)
        if self.num_ant_txrx == 1:
            if test_case == 0:
                h[0, 0] = np.array([0.3977, 0.7954 - 0.3977j, -0.1988, 0.0994, -0.0398])
            else:
                print('# Load from MATLAB channel toolbox - currently not done')
                exit(0)

        elif self.num_ant_txrx == 2:
            if test_case == 0:
                h[0, 0] = np.array([0.3977, 0.7954 - 0.3977j, -0.1988, 0.0994, -0.0398])
                h[0, 1] = np.array([0.8423j, 0.5391, 0, 0, 0])
                h[1, 0] = np.array([0.1631, -0.0815 + 0.9784j, 0.0978, 0, 0])
                h[1, 1] = np.array([0.0572j, 0.3659j, 0.5717 - 0.5717j, 0.4574, 0])
            else:
                print('# Load from MATLAB channel toolbox - currently not done')
                exit(0)

        for rx in range(self.num_ant_txrx):
            for tx in range(self.num_ant_txrx):
                if self.wireless_channel == 'AWGN':
                    self.channel_time[rx, tx, 1] = 1
                else:
                    if test_case == 0:
                        # need to normalize the channel
                        self.channel_time[rx, tx, 0:len(h[rx, tx])] = h[rx, tx] / np.linalg.norm(h[rx, tx])
                    else:
                        # channels from MATLAB toolbox already normalized
                        print('# Load normalized channels from MATLAB toolbox - currently not done')
                        exit(0)

                    # Take FFT of channels
                    self.channel_freq[rx, tx, :] = np.fft.fft(self.channel_time[rx, tx, 0:len(h[rx, tx])], self.NFFT)
                    self.h_f[rx, tx, :] = self.channel_freq[rx, tx, self.used_bins.astype(int)]

        self.genie_chan_time = self.channel_time

        max_ind = 0
        for i in range(self.channel_time.shape[0]):
            # for j in range(self.channel_time.shape(1)):
            for j in range(1):
                hh = abs(self.channel_time[i, j, :])
                v, i = hh.max(0), hh.argmax(0)

                if i > max_ind:
                    max_ind = i
                    max_val = v

        self.chan_max_offset = max_ind - 1  # Maybe -1 not required??
        # print(self.chan_max_offset)
        # print(self.h_f[0, 0, 0:10])

    def multi_ant_binary_map(self, Caz, binary_info, synch_data):
        # self =
        # self.binary_info = binary_info
        # self.symbol_pattern = symbol_pattern
        self.zchu = Caz.ZChu0
        self.used_bins_synch = Caz.used_bins.astype(int)
        self.used_bins_data = self.used_bins.astype(int)
        # self.NFFT = int.NFFT)
        # self.num_ant_txrx = self.num_ant_txrx
        self.M = Caz.M
        self.synch_state = Caz.synch_state
        self.synch_data = synch_data

        self.num_used_bins_data = self.num_used_bins
        self.num_bits_bin = self.OFDM_data.num_bits_bin
        self.modulation_type = self.OFDM_data.modulation_type
        # print(self.num_bits_bin)

        # print(self.used_bins_synch)

        # Counter for number of data symbols
        loop_data = 0
        for symb in range(len(self.symbol_pattern)):
            if self.symbol_pattern[symb] == 0:
                # synch signal (control)
                x_in = np.zeros((self.num_ant_txrx, int(len(self.zchu) / self.M[0])), dtype=complex)
                for ant in range(self.num_ant_txrx):
                    pass
                    # only send on antenna 1
                    if ant == 0:
                        start = int(self.synch_state * len(self.zchu) / self.M[0])
                        fin = int((self.synch_state + 1) * len(self.zchu) / self.M[0])
                        x_in[ant, :] = self.zchu[start: fin]
                    self.synch_state = self.synch_state % self.M[0]
                self.buffer_data_tx[:, (self.NFFT * symb) + self.used_bins_synch] = x_in
            elif self.symbol_pattern[symb] == 1:

                start = self.num_bits_bin * self.num_used_bins_data * loop_data
                fin = self.num_bits_bin * self.num_used_bins_data * (loop_data + 1)
                x_in = binary_info[:, start: fin]
                # print(x_in.shape)

                # Binary to complex mapping
                if self.modulation_type == 'BPSK':
                    cmplx_data = 2 * x_in - 1

                elif self.modulation_type == 'QPSK':
                    binary_wts = 2**np.array(range(self.num_bits_bin - 1, -1, -1))
                    # print(binary_wts)
                    cmplx_data = np.zeros((self.stream_size, self.num_used_bins_data), dtype=complex)
                    # for each antenna stream - loopB
                    for stream in range(self.stream_size):
                        # for each frequency bin  - loopA
                        for fbin in range(self.num_used_bins_data):
                            bits_in_bin = x_in[stream, fbin * self.num_bits_bin: (fbin + 1) * self.num_bits_bin]
                            # print(binary_wts.T.shape)
                            # Convert binary to decimal
                            decimal_data = np.dot(bits_in_bin, binary_wts.T)
                            if decimal_data == 0:
                                cmplx_data[stream, fbin] = np.exp(1j * 2 * np.pi / 8 * 1)
                            elif decimal_data == 1:
                                cmplx_data[stream, fbin] = np.exp(1j * 2 * np.pi / 8 * -1)
                            elif decimal_data == 2:
                                cmplx_data[stream, fbin] = np.exp(1j * 2 * np.pi / 8 * 3)
                            elif decimal_data == 3:
                                cmplx_data[stream, fbin] = np.exp(1j * 2 * np.pi / 8 * 5)

                loop_data += 1  # increment data symbol counter

                if self.num_ant_txrx == 1:
                    self.buffer_data_tx[0, (self.NFFT * symb) + self.used_bins_data] = cmplx_data[0, :]
                elif self.num_ant_txrx == 2 and self.MIMO_method == 'SpMult':
                    print('2 antennas with spatial multiplexing - not implemented yet')
                    exit(0)

                    # print(self.buffer_data_tx[0, 1020:1030])
    def multi_ant_symb_gen(self, num_symbols):
        pass
        min_pow = 1e-30

        for symb in range(num_symbols):
            scale_factor = 1
            P = 0

            for ant in range(self.num_ant_txrx):
                freq_data = self.buffer_data_tx[ant, symb * self.NFFT: (symb + 1) * self.NFFT]
                data_ifft = np.fft.ifft(freq_data, self.NFFT)
                cyclic_prefix = data_ifft[-self.len_CP:]
                data_time = np.concatenate((cyclic_prefix, data_ifft))  # add CP
                sig_energy = abs(np.dot(data_time, np.conj(data_time).T))
                # power scaling to normalize to 1
                if sig_energy > min_pow and ant == 0:
                    scale_factor = np.sqrt(len(data_time) / sig_energy)

                data_time *= scale_factor

                start = symb * (self.NFFT + self.len_CP)
                fin = (symb + 1) * (self.NFFT + self.len_CP)
                self.buffer_data_tx_time[ant, start:fin] = data_time
                # print(np.var(data_time))
                P += np.var(data_time)

            for ant in range(self.num_ant_txrx):
                start = symb * (self.NFFT + self.len_CP)
                fin = (symb + 1) * (self.NFFT + self.len_CP)
                self.buffer_data_tx_time[ant, start: fin] *= (1 / np.sqrt(P))
                # print(self.buffer_data_tx_time.shape)

    def rx_signal_gen(self):
        for rx in range(self.num_ant_txrx):
            rx_sig_ant = 0  # sum rx signal at each antenna
            for tx in range(self.num_ant_txrx):
                chan = self.channel_time[rx, tx, :]
                tx_sig = self.buffer_data_tx_time[tx, :]
                rx_sig_ant += np.convolve(tx_sig, chan)
                # print(self.buffer_data_tx_time[tx, :].shape)
                # print(self.channel_time[rx, tx, :].shape)

            self.buffer_data_rx_time[rx, :] = rx_sig_ant

        # print(self.buffer_data_rx_time[0])

    def additive_noise(self, SNR_type, SNR_dB, wireless_channel, sig_datatype):
        self.SNR_lin = 10**(SNR_dB / 10)
        sig_pow = np.var(self.buffer_data_tx_time)  # Determine the expected value of tx signal power

        bits_per_symb = len(self.used_bins_data) * self.num_bits_bin
        samp_per_symb = self.NFFT + self.len_CP

        # Calculate noise variance
        if SNR_type == 'Digital':
            self.noise_var = (1 / bits_per_symb) * samp_per_symb * sig_pow * 10**(-SNR_dB / 10)
        elif SNR_type == 'Analog':
            self.noise_var = sig_pow * 10**(-SNR_dB / 10)

        self.SNR_analog = sig_pow / self.noise_var
        # print(self.noise_var)
        # Add noise at each receive antenna

        for rx in range(self.num_ant_txrx):

            if sig_datatype == 'Real':
                awg_noise = np.sqrt(self.noise_var) * np.random.normal(0, 1, self.buffer_data_rx_time[rx, :].shape)
                # print(awg_noise)
            elif sig_datatype == 'Complex':
                awg_noise = np.sqrt(self.noise_var / 2) * np.random.normal(0, 1, self.buffer_data_rx_time[rx, :].shape) + 1j * np.sqrt(self.noise_var / 2) * np.random.normal(0, 1, self.buffer_data_rx_time[rx, :].shape)

            self.buffer_data_rx_time[rx, :] += awg_noise
