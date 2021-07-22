import numpy as np
import matplotlib.pyplot as plt


class PLSRxProcess:
    def __init__(self, pls_dictionary_case):
        self.pls_profiles = {
            0: {'bandwidth': 960e3,
                'bin_spacing': 15e3,
                'num_ant': 2,
                'bit_codebook': 1,
                'synch_data_pattern': [2, 1]}}
        self.snr = -30

        self.bandwidth = self.pls_profiles[pls_dictionary_case]['bandwidth']
        self.bin_spacing = self.pls_profiles[pls_dictionary_case]['bin_spacing']
        self.num_ant = self.pls_profiles[pls_dictionary_case]['num_ant']
        self.bit_codebook = self.pls_profiles[pls_dictionary_case]['bit_codebook']
        self.synch_data_pattern = self.pls_profiles[pls_dictionary_case]['synch_data_pattern']

    def awgn(self, in_signal):
        """
        Adds AWGN noise on per antenna basis
        :param in_signal: Input signal
        :return noisy_signal: Signal with AWGN added on per antenna basis
        """

        sig_pow = np.var(in_signal[0, 0:80])  # Determine the expected value of tx signal power

        noise_var = sig_pow * (10 ** (- self.snr / 10))

        for rx in range(self.num_ant):
            awg_noise = np.sqrt(noise_var / 2) * (
                    np.random.normal(0, 1, in_signal[rx, :].shape) + 1j * np.random.normal(0, 1,
                                                                                           in_signal[rx, :].shape))

            in_signal[rx, :] += awg_noise
            if rx == 0:
                plt.plot(awg_noise.real)
                plt.title('Noise for Ant 0 Real')
                plt.show()

        noisy_signal = in_signal
        return noisy_signal

    def rx_signal_gen(self, buffer_tx_time, channel_sel):
        """
        Generates the time domain rx signal at each receive antenna (Convolution with channel and add noise)
        :param buffer_tx_time: Time domain tx signal streams on each antenna (matrix)
        :return buffer_rx_time: Time domain rx signal at each receive antenna
        """
        max_impulse = 64
        total_symb_len = 960

        NFFT = int(np.floor(self.bandwidth / self.bin_spacing))
        CP = int(0.25 * NFFT)
        num_data_bins = 4
        subband_size = self.num_ant

        DC_index = int(NFFT / 2)
        neg_data_bins = list(range(DC_index - int(num_data_bins / 2), DC_index))
        pos_data_bins = list(range(DC_index + 1, DC_index + int(num_data_bins / 2) + 1))
        used_data_bins = np.array(neg_data_bins + pos_data_bins)

        h = np.zeros((self.num_ant, self.num_ant), dtype=object)
        h_f = np.zeros((self.num_ant, self.num_ant, num_data_bins), dtype=complex)
        channel_time = np.zeros((self.num_ant, self.num_ant, max_impulse), dtype=complex)
        channel_freq = np.zeros((self.num_ant, self.num_ant, NFFT), dtype=complex)

        if channel_sel == 0:
            h[0, 0] = np.array([1, 0.7954, -0.1988, 0.0994, -0.0398])
            h[0, 1] = np.array([0.8423, 0.5391, 0, 0, 0])
            h[1, 0] = np.array([0.1631, -0.0815, 0.0978, 0, 0])
            h[1, 1] = np.array([0.0572, 0.3659, 0.5717, 0.4574, 0])
        elif channel_sel == 1:
            h[0, 0] = np.array([0.3977])
            h[0, 1] = np.array([0.8423j])
            h[1, 0] = np.array([0.1631])
            h[1, 1] = np.array([0.0572j])
        elif channel_sel == 2:
            h[0, 0] = np.array([0.3977, 0.7954 - 0.3977j, -0.1988, 0.0994, -0.0398])
            h[0, 1] = np.array([0.8423j, 0.5391, 0, 0, 0])
            h[1, 0] = np.array([0.1631, -0.0815 + 0.9784j, 0.0978, 0, 0])
            h[1, 1] = np.array([0.0572j, 0.3659j, 0.5717 - 0.5717j, 0.4574, 0])
        else:
            h[0, 0] = np.array([1])
            h[0, 1] = np.array([1])
            h[1, 0] = np.array([1])
            h[1, 1] = np.array([1])

        for rx in range(self.num_ant):
            for tx in range(self.num_ant):
                channel_time[rx, tx, 0:len(h[rx, tx])] = h[rx, tx] / np.linalg.norm(h[rx, tx])

                channel_freq[rx, tx, :] = np.fft.fft(channel_time[rx, tx, 0:len(h[rx, tx])], NFFT)
                h_f[rx, tx, :] = channel_freq[rx, tx, used_data_bins.astype(int)]

        buffer_rx_time = np.zeros((self.num_ant, total_symb_len + max_impulse - 1), dtype=complex)
        for rx in range(self.num_ant):
            rx_sig_ant = 0  # sum rx signal at each antenna
            for tx in range(self.num_ant):
                chan = channel_time[rx, tx, :]
                tx_sig = buffer_tx_time[tx, :]
                rx_sig_ant += np.convolve(tx_sig, chan)

            buffer_rx_time[rx, :] = rx_sig_ant
        plt.plot(buffer_rx_time.real[0])
        plt.title(f'Real RX Buffer before Noise, Ant 0')
        plt.show()
        buffer_rx_time = self.awgn(buffer_rx_time)
        plt.plot(buffer_rx_time.real[0])
        plt.title(f'Real RX Buffer after Noise, Ant 0')
        plt.show()
        return buffer_rx_time
