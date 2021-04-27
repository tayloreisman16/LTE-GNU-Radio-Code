from numpy import zeros, array, tile, reshape, size, newaxis, delete,   \
    sum, prod, divide, sqrt, exp, log10,     \
    floor,   \
    conj, matmul, diag, outer, dot,  \
    complex64, int16,   \
    pi
from numpy.fft import fft, ifft
from numpy.linalg import norm
import pickle
import matplotlib.pyplot as plt
from gnuradio import gr


class ChannelEstimate(gr.sync_block):
    def __init__(self, num_ofdm_symb, nfft, cp_len,
                 num_synch_bins, synch_dat, num_data_bins, channel, snr, scale_factor_gate, directory_name,
                 file_name_cest, plot_iq, channel_graph_plot, perfect_chan_est, save_channel_file):
        self.num_ofdm_symb = num_ofdm_symb
        self.nfft = nfft
        self.cp_len = cp_len
        self.scale_factor_gate = scale_factor_gate
        self.channel = channel
        self.plot_iq = plot_iq
        self.channel_graph_plot = channel_graph_plot
        self.perfect_chan_est = perfect_chan_est
        self.save_channel_file = save_channel_file

        self.channel_band = 960e3 * 0.97
        self.fs = self.channel_band
        self.bin_spacing = 15e3

        self.num_synch_bins = num_synch_bins

        self.synch_bins_used_N = (list(range(-int(self.num_synch_bins / 2), 0, 1)) +
                                  list(range(1, int(self.num_synch_bins / 2) + 1, 1)))

        self.synch_bins_used_P = list((array(self.synch_bins_used_N) + self.nfft) % self.nfft)

        self.L_synch = len(self.synch_bins_used_P)
        self.synch_dat = synch_dat  # example [1, 1] or [2, 1]
        self.num_data_symbs_blk = self.synch_dat[1]
        # self.symb_blk_length = sum(self.sync_dat) * self.rx_b_len
        # self.data_blk_length = self.num_data_symbs * self.rx_b_len
        self.M = [self.synch_dat[0], self.num_synch_bins]
        self.MM = int(prod(self.M))

        # Zadoff Chu Generation
        self.p = 23

        x0 = array(range(0, int(self.MM)))
        x1 = array(range(1, int(self.MM) + 1))
        if self.MM % 2 == 0:
            self.zadoff_chu = exp(-1j * (2 * pi / self.MM) * self.p * (x0 ** 2 / 2))
        else:
            self.zadoff_chu = exp(-1j * (2 * pi / self.MM) * self.p * (x0 * x1) / 2)

        plt.plot(self.zadoff_chu.real)
        plt.plot(self.zadoff_chu.imag)
        plt.title('Reference Zadoff-Chu Signal')
        plt.show()

        self.num_data_bins = num_data_bins

        self.bins_used_N = (list(range(-int(self.num_data_bins / 2), 0, 1)) +
                            list(range(1, int(self.num_data_bins / 2) + 1, 1)))
        self.bins_used_P = list((array(self.bins_used_N) + self.nfft) % self.nfft)

        self.corr_obs = -1

        self.del_mat_exp = tile(exp((1j * (2.0 * pi / self.nfft)) * (
            outer(list(range(self.cp_len + 1)), list(self.synch_bins_used_P)))), (1, self.M[0]))

        self.stride_val = 1
        self.start_samp = self.cp_len
        self.rx_b_len = self.nfft + self.cp_len

        self.max_num_corr = 100

        self.time_synch_ref = zeros(3)
        self.est_chan_time = zeros((num_ofdm_symb, self.nfft), dtype=complex)
        self.est_synch_freq = zeros((num_ofdm_symb, len(self.zadoff_chu)), dtype=complex)

        self.est_chan_freq_P = zeros((num_ofdm_symb, self.nfft), dtype=complex)
        self.est_data_freq = zeros((num_ofdm_symb, self.num_data_bins), dtype=complex)

        self.rx_data_time = zeros((1, self.M[0] * self.nfft), dtype=complex)  # first dimension = no of antennas
        self.synchdat00 = zeros((1, self.M[0] * self.num_synch_bins), dtype=complex)

        self.del_mat = []
        self.eq_gain = []
        self.eq_gain_ext = []
        self.eq_gain_q = []

        self.SNR = snr
        self.SNR_lin = 10 ** (self.SNR / 20)
        self.count = 0

        self.directory_name = directory_name
        self.file_name_cest = file_name_cest

        self.num_ant_txrx = 1  # Hard Coded

        # Genie Channel Variables
        if self.channel_graph_plot == 1:
            self.max_impulse = self.nfft
            self.genie_chan_time = zeros((self.num_ant_txrx, self.num_ant_txrx, self.max_impulse), dtype=complex)

            num_bins0 = floor(self.channel_band / self.bin_spacing)
            num_bins1 = 4 * floor(num_bins0 / 4)
            all_bins = array(list(range(-int(num_bins1 / 2), 0)) + list(range(1, int(num_bins1 / 2) + 1)))
            self.used_bins = (self.nfft + all_bins)

        gr.sync_block.__init__(self,
                               name="SynchAndChanEst",
                               in_sig=[complex64, int16],
                               out_sig=[complex64])

    def work(self, input_items, output_items):
        in0 = input_items[0]
        self.time_synch_ref[0] = input_items[1]
        for P in list(range(n_unique_symb)[::sum(self.synch_dat)]):
            data_ptr = int(self.time_synch_ref[0] + self.M[0] * self.rx_b_len * (P + 1))
            if self.time_synch_ref[0] + self.M[0] * self.rx_b_len * (P + 1) + self.nfft - 1 <= len(in0):

                for N in range(self.synch_dat[1]):
                    start = data_ptr + self.rx_b_len * N
                    end = data_ptr + self.rx_b_len * N + self.nfft
                    data_buff_time = in0[start: end]

                    t_vec = fft(data_buff_time, self.nfft)

                    freq_data_0 = t_vec[self.bins_used_P]
                    p_est0 = sqrt(len(freq_data_0)/(dot(freq_data_0, conj(freq_data_0))))

                    data_recov_0 = freq_data_0 * p_est0

                    arg_val = ([((1j * (2 * pi / self.nfft)) *
                                 self.time_synch_ref[1]) * kk for kk in self.bins_used_P])

                    data_recov_z = matmul(diag(data_recov_0), exp(arg_val))

                    chan_est_dat = self.est_chan_freq_P[0][self.bins_used_P]

                    chan_mag_z = matmul(diag(chan_est_dat), conj(chan_est_dat))
                    eq_gain_z = [1.0 / self.SNR_lin + vv for vv in chan_mag_z]
                    self.eq_gain_q = divide(conj(chan_est_dat), eq_gain_z)

                    self.est_data_freq[P + N][:] = matmul(diag(self.eq_gain_q), data_recov_z)
        data_demod = delete(self.est_data_freq, list(range(3, self.est_data_freq.shape[0], sum(self.synch_dat))), axis=0)
        if self.plot_iq == 1:
            plt.plot(self.est_data_freq[0][:].real, self.est_data_freq[0][:].imag, 'o')
            plt.show()

        data_out = reshape(data_demod[:][:], (1, n_data_symb * size(data_demod, 1)))

        if self.count > 0:
            out[0:size(data_out, 1)] = data_out[:, newaxis]

        self.count += 1
        self.corr_obs = 0
        return len(output_items[0])