from numpy import zeros, ones, array, tile, concatenate, reshape, real,  \
    sum, prod, sqrt, exp, angle, var, log2,   \
    ceil, floor, argwhere, bitwise_xor,   \
    conj, diag, dot,  \
    pi
from numpy.fft import fft, ifft
from numpy.random import uniform, choice, seed
from numpy.linalg import svd, qr
# import pickle as pckl
import matplotlib.pyplot as plt


class PhyLayerSecAIO:
    def __init__(self, pvt_info_length, pls_dictionary_case):
        # gr.sync_block.__init__(self,
        #                        name="PLS_AIO",
        #                        in_sig=[complex64],
        #                        out_sig=[complex64])
        self.pvt_info_len = pvt_info_length
        pls_profiles = {
            0: {'bandwidth': 960e3,
                'bin_spacing': 15e3,
                'num_ant': 2,
                'bit_codebook': 1,
                'synch_data_pattern': [2, 1]},
            }
        # PLS Parameters #
        self.bandwidth = pls_profiles[pls_dictionary_case]['bandwidth']
        self.bin_spacing = pls_profiles[pls_dictionary_case]['bin_spacing']
        self.num_ant = pls_profiles[pls_dictionary_case]['num_ant']
        self.bit_codebook = pls_profiles[pls_dictionary_case]['bit_codebook']
        self.synch_data_pattern = pls_profiles[pls_dictionary_case]['synch_data_pattern']
        self.bit_codebook = self.codebook_gen()

        self.nfft = int(floor(self.bandwidth / self.bin_spacing))
        self.CP = int(0.25 * self.nfft)
        self.OFDMsymb_len = self.nfft + self.CP
        self.num_data_bins = 4
        self.num_synch_bins = self.nfft - 2
        self.subband_size = self.num_ant

        if self.num_data_bins == 1:
            self.used_data_bins = array([10])

        dc_index = int(self.nfft / 2)
        neg_data_bins = list(range(dc_index - int(self.num_data_bins / 2), dc_index))
        pos_data_bins = list(range(dc_index + 1, dc_index + int(self.num_data_bins / 2) + 1))
        self.used_data_bins = array(neg_data_bins + pos_data_bins)

        neg_synch_bins = list(range(dc_index - int(self.num_synch_bins / 2), dc_index))
        pos_synch_bins = list(range(dc_index + 1, dc_index + int(self.num_synch_bins / 2) + 1))
        self.used_synch_bins = array(neg_synch_bins + pos_synch_bins)

        self.num_subbands = int(floor(self.num_data_bins/self.subband_size))
        self.num_PMI = self.num_subbands
        self.max_impulse = self.nfft

        self.key_len = self.num_subbands * self.bit_codebook
        # End PLS Parameters #

        # PLS Script Parameters #
        self.num_data_symb = int(ceil(self.pvt_info_len / self.num_subbands * int(log2(len(self.bit_codebook)))))
        self.num_synch_symb = self.synch_data_pattern[0] * self.num_data_symb

        self.total_num_symb = self.num_synch_symb + self.num_data_symb
        self.num_synchdata_patterns = int(self.total_num_symb / sum(self.synch_data_pattern))

        self.symb_pattern0 = concatenate((zeros(self.synch_data_pattern[0]), ones(self.synch_data_pattern[1])))
        self.symb_pattern = tile(self.symb_pattern0, self.num_synchdata_patterns)
        self.state = 0

        # Synch Signal Gen #
        self.num_synch_bins = self.nfft - 2

        self.prime_no = [23, 41]
        self.prime_nos = self.prime_no * self.num_data_symb
        self.num_unique_synch = len(self.prime_no)
        assert len(self.prime_nos) == self.num_synch_symb

        self.synch_signals = zeros((self.num_synch_symb, self.OFDMsymb_len), dtype=complex)
        self.synch_start = list()
        self.synch_mask = zeros((self.num_ant, self.OFDMsymb_len * len(self.symb_pattern)), dtype=complex)

        self.synch = self.synch_signal_gen()

        # PLS Transmitter Config #
        self.num_bits_symb = int((self.num_data_bins / self.subband_size))

        # GNU State Config
        self.state = 0

        # Work Config #
        self.ref_sig_A = zeros((self.num_data_symb, self.num_data_bins), dtype=complex)
        self.ref_sig_B = zeros((self.num_data_symb, self.num_data_bins), dtype=complex)

        self.lsv_B0 = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        self.rsv_B0 = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        self.lsv_A = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        self.rsv_A = zeros((self.num_data_symb, self.num_subbands), dtype=object)

        self.precoders_A = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        self.precoders_B = zeros((self.num_data_symb, self.num_subbands), dtype=object)

        # Dummy Data
        self.pvt_info_bits_b = [0, 0, 0, 1, 1, 0, 1, 1]

    def work(self, input_items, output_items):
        buffer_tx_time = input_items[:]
        out = output_items[:]

        buffer_rx_time = zeros((2, 960), dtype=complex)
        # TODO: Zero out output when in RX states
        if self.state == 0:
            buffer_rx_time, self.ref_sig_A, self.precoders_A = self.transmit_signal_gen('Alice0', self.num_data_symb)
        elif self.state == 1:
            self.lsv_B0, self.rsv_B0, _ = self.receive_sig_process(buffer_tx_time, self.ref_sig_A)

            buffer_rx_time, self.ref_sig_B, self.precoders_B = self.transmit_signal_gen('Bob',
                                                                                        self.num_data_symb,
                                                                                        self.pvt_info_bits_b,
                                                                                        self.lsv_B0)
            plt.plot(buffer_rx_time[0].real)
            plt.plot(buffer_rx_time[0].imag)
            plt.plot(buffer_rx_time[1].real)
            plt.plot(buffer_rx_time[1].imag)
            plt.title('TX Signal from Bob to Alice @ Bob')
            plt.show()
        elif self.state == 2:
            self.lsv_A, self.rsv_A, obs_info_bits_A = self.receive_sig_process(buffer_tx_time, self.ref_sig_B)
            b_key_obs_at_a = concatenate(reshape(obs_info_bits_A, prod(obs_info_bits_A.shape)))
            num_bit_err = bitwise_xor(b_key_obs_at_a, self.pvt_info_bits_b).sum()
            print(f'Number of bit errors Bob to Alice: {num_bit_err}')
            print('B Key OBS:', b_key_obs_at_a)
        else:
            print('Out of States')

        self.state += 1

        out[:] = buffer_rx_time  # TODO: What size should Buffer RX Time be?
        # return len(output_items)
        return out[:]

    def codebook_gen(self):
        num_precoders = 2**self.bit_codebook
        codebook = zeros(num_precoders, dtype=object)

        for p in range(0, num_precoders):
            precoder = zeros((self.num_ant, self.num_ant), dtype=complex)
            for m in range(0, self.num_ant):
                for n in range(0, self.num_ant):
                    w = exp(1j*2*pi*(n/self.num_ant)*(m + p/num_precoders))
                    precoder[n, m] = (1/sqrt(self.num_ant))*w

            codebook[p] = precoder
            plt.title(f'Codebook Classification: {p}')
            plt.scatter(codebook[p].real, codebook[p].imag)
            plt.show()

        return codebook

    def synch_signal_gen(self):
        for symb in range(self.num_synch_symb):
            synch_symb = self.zadoff_chu_gen(self.prime_nos[symb])  # size NFFT - 2

            synch_freq = zeros(self.nfft, dtype=complex)
            synch_freq[self.used_synch_bins] = synch_symb  # size NFFT

            synch_ifft = ifft(synch_freq)  # size NFFT
            synch_cp = synch_ifft[-self.CP:]  # size CP
            synch_time = concatenate((synch_cp, synch_ifft))  # size NFFT + CP

            power_est = sum(synch_time*conj(synch_time))/len(synch_time)
            norm_synch = synch_time/sqrt(power_est)

            self.synch_signals[symb, :] = norm_synch

        total_symb_count = 0
        synch_symb_count = 0

        for symb in self.symb_pattern.tolist():
            if symb == 0:
                modulo_switch = int(synch_symb_count % int(self.num_ant * self.num_unique_synch))
                symb_start = total_symb_count * self.OFDMsymb_len
                symb_end = symb_start + self.OFDMsymb_len

                self.synch_start.append(symb_start)
                if modulo_switch == 0 or modulo_switch == 1:
                    self.synch_mask[0, symb_start: symb_end] = self.synch_signals[synch_symb_count]
                elif modulo_switch == 2 or modulo_switch == 3:
                    self.synch_mask[1, symb_start: symb_end] = self.synch_signals[synch_symb_count]
                synch_symb_count += 1

            total_symb_count += 1
        return self.synch_mask

    def zadoff_chu_gen(self, prime):
        x0 = array(range(0, self.num_synch_bins))
        x1 = array(range(1, self.num_synch_bins + 1))
        if self.num_synch_bins % 2 == 0:
            zadoff_chu = exp(-1j * (2 * pi / self.num_synch_bins) * prime * (x0**2 / 2))
        else:
            zadoff_chu = exp(-1j * (2 * pi / self.num_synch_bins) * prime * (x0 * x1) / 2)

        return zadoff_chu

    def transmit_signal_gen(self, *args):
        """
        This method deals with all the transmitter functions - gen ref signals, precoding, OFDM mod, IFFT, CP
        :param args: 1. Which node is transmitting - Alice or Bob? 2. Total number of data symbols
        :return: Time domain buffer of OFDM symbols (synch + data). Ready to send over channel.
        """

        tx_node = args[0]
        num_data_symb = args[1]

        if tx_node == 'Alice0':
            precoders = self.unitary_gen()
        elif tx_node == 'Bob':
            pvt_info_bits = args[2]
            rotation_mat = args[3]
            bits_subband = self.map_bits2subband(pvt_info_bits)
            dft_precoders = self.codebook_select(bits_subband)
            precoders = self.rotated_preocder('Bob', dft_precoders, rotation_mat)
        elif tx_node == 'Alice1':
            precoders = None
        else:
            precoders = None
            print('Error')

        ref_sig = self.ref_signal_gen()
        freq_bin_data = self.apply_precoders(precoders, ref_sig, num_data_symb)
        time_ofdm_data_symbols = self.ofdm_modulate(num_data_symb, freq_bin_data)
        buffer_tx_time = self.synch_data_mux(time_ofdm_data_symbols)
        return buffer_tx_time, ref_sig, precoders

    def unitary_gen(self):
        """
        Generate random unitary matrices for each symbol for each sub-band
        :return: matrix of random unitary matrices
        """
        unitary_mats = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        for symb in range(self.num_data_symb):
            for sb in range(0, self.num_subbands):
                q, r = qr(uniform(0, 1, (self.num_ant, self.num_ant))
                          + 1j * uniform(0, 1, (self.num_ant, self.num_ant)))

                # unitary_mats[symb, sb] = identity(self.num_ant)
                unitary_mats[symb, sb] = dot(q, diag(diag(r) / abs(diag(r))))
        return unitary_mats

    def map_bits2subband(self, pvt_info_bits):
        """
        Based on the number of bits in the codebook index, each sub-band (group of bins=num antennas) in each symbol
        is assigned bits from the private info bit stream
        :param pvt_info_bits: Private information to be transmitted
        :return bits_subband: Bits in each sub-band for each data symbol
        """
        bits_subband = zeros((self.num_data_symb, self.num_subbands), dtype=object)

        for symb in range(self.num_data_symb):
            symb_bits_start = symb * self.num_bits_symb
            symb_bits_end = symb_bits_start + self.num_bits_symb
            symb_bits = pvt_info_bits[symb_bits_start: symb_bits_end]
            # Map secret key to subbands
            for sb in range(self.num_subbands):
                sb_bits_start = sb * int(log2(len(self.bit_codebook)))
                sb_bits_fin = sb_bits_start + int(log2(len(self.bit_codebook)))

                bits_subband[symb, sb] = symb_bits[sb_bits_start: sb_bits_fin]

        return bits_subband

    def codebook_select(self, bits_subband):
        """
        selects the DFT precoder from the DFT codebook based. Bits are converted to decimal and used as look up index.
        :param bits_subband: Bits in each sub-band for each data symbol
        :return dft_precoder: Selected DFT preocder from codebook for each sub-band in each data symbol
        """
        dft_precoder = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        for symb in range(self.num_data_symb):
            for sb in range(self.num_subbands):
                bits = bits_subband[symb, sb]
                start = int(log2(len(self.bit_codebook))) - 1
                bi2dec_wts = 2**(array(range(start, -1, -1)))
                codebook_index = int(sum(bits*bi2dec_wts))
                dft_precoder[symb, sb] = self.bit_codebook[codebook_index]
        plt.title('DFT Precoder: Binary-Codebook 1')
        plt.plot(dft_precoder[1, 1].real, dft_precoder[1, 1].imag, 'o')
        plt.show()

        return dft_precoder

    def rotated_preocder(self, tx_node, dft_precoders, rotation_mat):
        """
        applies channel-based rotation matrix (LSV from previous SVD) to the DFT precoder
        :param tx_node: Who is transmitting - Alice or Bob?
        :param dft_precoders: selected DFT precoders from codebook based on secret bits
        :param rotation_mat: channel-based rotation matrix (LSV from previous SVD)
        :return:
        """
        precoders = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        if tx_node == 'Bob':
            for symb in range(self.num_data_symb):
                for sb in range(self.num_subbands):
                    precoders[symb, sb] = dot(conj(rotation_mat[symb, sb]), conj(dft_precoders[symb, sb]).T)

        return precoders

    def ref_signal_gen(self):
        """
        Generate QPSK reference signals for each frequency bin in each data symbol
        :return: Matrix of QPSK reference signals
        Same ref signal on both antennas in a bin. (Can be changed later)
        """
        seed(250)
        ref_sig = zeros((self.num_data_symb, self.num_data_bins), dtype=complex)
        for symb in range(self.num_data_symb):
            for fbin in range(self.num_data_bins):
                ref_sig[symb, fbin] = exp(1j * (pi / 4) * (choice(array([1, 3, 5, 7]))))
        plt.plot(ref_sig[0, 0].real, ref_sig[0, 0].imag, 'o')
        plt.title('Reference Signal (Should be in Lower Left Quadrant)')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()
        return ref_sig

    def apply_precoders(self, precoders, ref_sig, num_data_symb):
        """
        Applies precoders in each frequency bin.
        Example: no of antenans = 2 => sub-band size = 2 bins. One precoder matrix is split into 2 columns. Each column
        is applied in a bin in that sub-band
        :param precoders: matrix of precoders for each sub-band and for each data OFDM symbol
        :param ref_sig: matrix of reference signals for each frequency bin in each each data OFDM symbol
        Same ref signal on both antennas in a bin. (Can be changed later)
        :param num_data_symb: Total number of OFDM Data symbols
        :return: frequency bin data for each symbol (precoder*ref signal) and on each antenna
        """
        freq_bin_data = zeros((self.num_ant, num_data_symb * self.num_data_bins), dtype=complex)
        for symb in range(num_data_symb):
            # print(symb)
            symb_start = symb * self.num_data_bins
            symb_end = symb_start + self.num_data_bins

            fbin_val = zeros((self.num_ant, self.num_data_bins), dtype=complex)
            for sb in range(self.num_subbands):
                precoder = precoders[symb, sb]

                sb_start = sb * self.subband_size
                sb_end = sb_start + self.subband_size

                fbin_val[:, sb_start: sb_end] = precoder

            for fbin in range(self.num_data_bins):
                fbin_val[:, fbin] *= ref_sig[symb, fbin]

            freq_bin_data[:, symb_start: symb_end] = fbin_val
        return freq_bin_data

    def ofdm_modulate(self, num_data_symb, freq_bin_data):
        """
        Takes frequency bin data and places them in appropriate frequency bins, on each antenna
        Then takes FFT and adds CP to each data symbol
        :param num_data_symb: Total number of data OFDM symbols
        :param freq_bin_data: Frequency domain input data
        :return: Time domain tx stream of data symbols on each antenna
        """
        min_pow = 1e-30
        time_ofdm_symbols = zeros((self.num_ant, num_data_symb * self.OFDMsymb_len), dtype=complex)
        for symb in range(num_data_symb):
            freq_data_start = symb * self.num_data_bins
            freq_data_end = freq_data_start + self.num_data_bins

            time_symb_start = symb * self.OFDMsymb_len
            time_symb_end = time_symb_start + self.OFDMsymb_len

            p = 0
            for ant in range(self.num_ant):

                ofdm_symb = zeros(self.nfft, dtype=complex)
                ofdm_symb[self.used_data_bins] = freq_bin_data[ant, freq_data_start:freq_data_end]
                # plt.stem(array(range(-int(self.NFFT/2), int(self.NFFT/2))), abs(ofdm_symb))
                # plt.show()
                data_ifft = ifft(ofdm_symb, self.nfft)
                cyclic_prefix = data_ifft[-self.CP:]
                data_time = concatenate((cyclic_prefix, data_ifft))  # add CP

                sig_energy = abs(dot(data_time, conj(data_time).T))
                # power scaling to normalize to 1
                if sig_energy > min_pow and ant == 0:
                    scale_factor = sqrt(len(data_time) / sig_energy)
                else:
                    scale_factor = 1
                data_time *= scale_factor
                p += var(data_time)
                time_ofdm_symbols[ant, time_symb_start: time_symb_end] = data_time

            for ant in range(self.num_ant):
                time_ofdm_symbols[ant, time_symb_start: time_symb_end] *= (1 / sqrt(p))

        return time_ofdm_symbols

    def receive_sig_process(self, buffer_tx_time, ref_sig):
        """
        This method deals with all the receiver functions - generate rx signal (over channel + noise), synhronization,
        channel estimation, SVD, precoder detection, bit recovery
        :param buffer_tx_time: Time domain buffer of OFDM symbols (synch + data). Ready to send over channel.
        :param ref_sig: Matrix of QPSK reference signals. QPSK values for each frequency bin in each data symbol
        :return:
        """

        # synchronization - return just data symbols with CP removed
        buffer_rx_data = self.synchronize(buffer_tx_time)

        # Channel estimation in each of the used bins
        chan_est_bins = self.channel_estimate(buffer_rx_data, ref_sig)

        # Map bins to sub-bands to form matrices for SVD - gives estimated channel matrix in each sub-band
        chan_est_sb = self.bins2subbands(chan_est_bins)

        # SVD in each sub-band
        lsv, _, rsv = self.sv_decomp(chan_est_sb)

        # rsv is supposed to be the received dft precoder
        bits_sb_estimate = self.pmi_estimate(rsv)[1]
        return lsv, rsv, bits_sb_estimate

    def synchronize(self, buffer_rx_time):
        """
        Time domain synchronization - correlation with Zadoff Chu Synch mask
        :param buffer_rx_time: Time domain rx signal at each receive antenna
        :return buffer_rx_data: Time domain rx signal at each receive antenna with CP removed
        """
        buffer_rx_data = zeros((self.num_ant, self.num_data_symb * self.nfft), dtype=complex)

        total_symb_count = 0
        synch_symb_count = 0
        data_symb_count = 0
        for symb in self.symb_pattern.tolist():
            symb_start = total_symb_count * self.OFDMsymb_len
            symb_end = symb_start + self.OFDMsymb_len
            # print(symb_start, symb_end)
            if int(symb) == 0:
                synch_symb_count += 1
            else:
                # print(symb, symb_start, symb_end)
                data_start = data_symb_count * self.nfft
                data_end = data_start + self.nfft
                # print(data_start, data_end)
                # print(time_ofdm_data_symbols[:, data_start: data_end])
                data_with_cp = buffer_rx_time[:, symb_start: symb_end]
                data_without_cp = data_with_cp[:, self.CP:]
                buffer_rx_data[:, data_start: data_end] = data_without_cp
                data_symb_count += 1

            total_symb_count += 1
        # print(buffer_rx_data.shape)
        return buffer_rx_data


    def channel_estimate(self, buffer_rx_data, ref_sig):
        """
        In PLS, only refeence signals are sent. So we use the data symbols to estimate the chanel rather than the synch.
        :param buffer_rx_data: Buffer of rx data symbols with CP removed
        :param ref_sig: QPSK ref signals on each bin for each data symbol
        :return chan_est_bins: Estimated channel in each of the used data bins
        """
        chan_est_bins_sort = zeros((self.num_ant, self.num_data_symb, int(self.num_data_bins / self.subband_size),
                                    self.subband_size), dtype=complex)
        chan_est_bins = zeros((self.num_ant, self.num_data_symb * self.num_data_bins), dtype=complex)
        count = 0

        for symb in range(self.num_data_symb):
            symb_start = symb * self.nfft
            symb_end = symb_start + self.nfft

            used_symb_start = symb*self.num_data_bins
            used_symb_end = used_symb_start + self.num_data_bins
            for ant in range(self.num_ant):
                time_data = buffer_rx_data[ant, symb_start: symb_end]
                data_fft = fft(time_data, self.nfft)
                data_in_used_bins = data_fft[self.used_data_bins]

                est_channel = data_in_used_bins*conj(ref_sig[symb, :])/(abs(ref_sig[symb, :]))
                chan_est_bins[ant, used_symb_start: used_symb_end] = est_channel
                # est_channel = data_in_used_bins*conj(ref_sig[symb, :])/(1 + (1 / SNR_lin))
                for subband_index in range(int(self.num_data_bins / self.subband_size)):
                    start = subband_index * self.subband_size
                    end = start + self.subband_size
                    chan_est_bins_sort[:, symb, subband_index, :] = est_channel[start: end]
                count += 1

        return chan_est_bins

    # def channel_check(self, chan_est_bins_sort, ref_sig, precoders):
    #     chan_symb0_sb_0 = chan_est_bins_sort[:, 0, 0, :]
    #     ref_sig_symb0_sb_0 = tile(ref_sig[0, 0:2], (2, 1))
    #     precoder_symb0_sb_0 = precoders[0, 0]
    #     g = (ref_sig_symb0_sb_0 * precoder_symb0_sb_0)
    #     chan = chan_symb0_sb_0 / g
    #     return 0

    def bins2subbands(self, chan_est_bins):
        """
        Example: if num antenna = 2, every 2 adjacent bins form a sub-band. Each bin has a column vector (num antennas)
        By combining two adjacent column vectors, we get a matrix in each sub-band.
        :param chan_est_bins: Estimated channel at each used data frequency bin on each antenna
        :return chan_est_sb: channel matrix for each sub-band for each symbol
        """

        chan_est_sb = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        for symb in range(self.num_data_symb):
            symb_start = symb*self.num_data_bins
            symb_end = symb_start + self.num_data_bins
            chan_est = chan_est_bins[:, symb_start: symb_end]
            for sb in range(self.num_subbands):
                sb_start = sb*self.subband_size
                sb_end = sb_start + self.subband_size

                chan_est_sb[symb, sb] = chan_est[:, sb_start: sb_end]
                # chan_est_sb_avg = chan_est_sb[:, 0]
        return chan_est_sb

    def sv_decomp(self, chan_est_sb):
        """
        Perform SVD for the matrix in each sub-band in each data symbol
        :param chan_est_sb: Estimated channel in each sub-band in each data symbol
        :return lsv, sval, rsv: Left, Right Singular Vectors and Singular Values for the matrix in each sub-band
        in each data symbol
        """
        lsv = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        sval = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        rsv = zeros((self.num_data_symb, self.num_subbands), dtype=object)

        for symb in range(self.num_data_symb):
            for sb in range(0, self.num_subbands):
                u, s, vh = svd(chan_est_sb[symb, sb])
                v = conj(vh).T
                ph_shift_u = diag(exp(-1j * angle(u[0, :])))
                ph_shift_v = diag(exp(-1j * angle(v[0, :])))
                lsv[symb, sb] = dot(u, ph_shift_u)
                sval[symb, sb] = s
                rsv[symb, sb] = dot(v, ph_shift_v)

        return lsv, sval, rsv

    def pmi_estimate(self, rx_precoder):
        """
        Apply minumum distance to estimate the transmitted precoder, its index in the codebook and the binary equivalent
        of the index
        :param rx_precoder: observed precoder (RSV of SVD)
        :return PMI_sb_estimate, bits_sb_estimate: Preocder matrix index and bits for each sub-band for each data symbol
        """
        pmi_sb_estimate = zeros((self.num_data_symb, self.num_subbands), dtype=int)
        bits_sb_estimate = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        if self.state == 2:
            plt.title(f'RX Precoder: Binary 0')
            plt.scatter(rx_precoder[0, 0].real, rx_precoder[0, 0].imag)
            plt.show()
            plt.title(f'RX Precoder: Binary 1')
            plt.scatter(rx_precoder[1, 1].real, rx_precoder[1, 1].imag)
            plt.show()
        for symb in range(self.num_data_symb):
            for sb in range(self.num_subbands):
                dist = zeros(len(self.bit_codebook), dtype=float)

                for prec in range(len(self.bit_codebook)):
                    diff = rx_precoder[symb, sb] - self.bit_codebook[prec]
                    # plt.scatter(rx_precoder[symb, sb].real, rx_precoder[symb, sb].imag)
                    # plt.show()
                    diff_squared = real(diff*conj(diff))
                    dist[prec] = sqrt(diff_squared.sum())
                min_dist = min(dist)
                pmi_estimate = argwhere(dist == min_dist)
                pmi_sb_estimate[symb, sb] = pmi_estimate
                bits_sb_estimate[symb, sb] = self.dec2binary(pmi_estimate, self.bit_codebook.shape)

        return pmi_sb_estimate, bits_sb_estimate

    @staticmethod
    def dec2binary(x, num_bits):
        """
        Covert decimal number to binary array of ints (1s and 0s)
        :param x: input decimal number
        :param num_bits: Number bits required in the binary format
        :return bits: binary array of ints (1s and 0s)
        """
        bit_str = [char for char in format(x[0, 0], '0' + str(int(log2(len(num_bits)))) + 'b')]
        bits = array([int(char) for char in bit_str])
        return bits

    def synch_data_mux(self, time_ofdm_data_symbols):
        """
        Multiplexes synch and data symbols according to the symbol pattern
        Takes synch mask from synch class and inserts dats symbols next to the synchs
        :param time_ofdm_data_symbols: NFFT+CP size OFDM data symbols
        :return: time domain tx symbol stream per antenna (contains synch and data symbols) (matrix)
        """

        buffer_tx_time = self.synch  # Add data into this
        # plt.plot(buffer_tx_time[0, :].real)
        # plt.plot(buffer_tx_time[0, :].imag)
        # plt.show()
        total_symb_count = 0
        synch_symb_count = 0
        data_symb_count = 0
        for symb in self.symb_pattern.tolist():
            symb_start = total_symb_count*self.OFDMsymb_len
            symb_end = symb_start + self.OFDMsymb_len
            # print(symb_start, symb_end)
            if int(symb) == 0:
                synch_symb_count += 1
            else:
                # print(symb, symb_start, symb_end)
                data_start = data_symb_count*self.OFDMsymb_len
                data_end = data_start + self.OFDMsymb_len

                buffer_tx_time[:, symb_start: symb_end] = time_ofdm_data_symbols[:, data_start: data_end]
                data_symb_count += 1

            total_symb_count += 1

        return buffer_tx_time
