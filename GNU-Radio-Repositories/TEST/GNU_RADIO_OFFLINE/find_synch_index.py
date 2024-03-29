from numpy import zeros, ones, array, tile, reshape, size, newaxis,   \
    sum, multiply, prod, divide, sqrt, exp, log10,     \
    floor, around,   \
    conj, matmul, diag, outer,   \
    argmax, max,   \
    complex64, int16,   \
    pi
from numpy.fft import fft, ifft
from numpy.linalg import norm
import pickle
import matplotlib.pyplot as plt
# from gnuradio import gr


# class SynchronizeIndex(gr.sync_block):
class SynchronizeIndex():
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

        # gr.sync_block.__init__(self,
        #                        name="SynchAndChanEst",
        #                        in_sig=[complex64],
        #                        out_sig=[complex64, int16])

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        out1 = output_items[1]

        n_trials = int(around(len(in0) / self.stride_val))
        n_unique_symb = int(floor(len(in0) / self.rx_b_len))
        n_data_symb = int(n_unique_symb * (self.synch_dat[1] / sum(self.synch_dat)))

        for P in list(range(n_trials)):
            if self.M[0] * self.rx_b_len + P * self.stride_val + self.nfft + self.start_samp < len(in0):
                for LL in list(range(self.M[0])):
                    aaa = self.rx_b_len * LL + P * self.stride_val + self.start_samp
                    bbb = self.rx_b_len * LL + P * self.stride_val + self.start_samp + self.nfft
                    self.rx_data_time[0][LL * self.nfft:(LL + 1) * self.nfft] = in0[aaa:bbb]

                tmp_1vec = zeros((self.M[0], self.nfft), dtype=complex)
                for LL in list(range(self.M[0])):
                    tmp_1vec[LL][:] = fft(self.rx_data_time[0][LL * self.nfft:(LL + 1) * self.nfft], self.nfft)
                    self.synchdat00[0][LL * self.num_synch_bins:(LL + 1) * self.num_synch_bins] = \
                        tmp_1vec[LL][self.synch_bins_used_P]

                synchdat0 = reshape(self.synchdat00, (1, self.num_synch_bins * self.M[0]))
                p_est = sqrt(len(synchdat0[0]) / sum(multiply(synchdat0[0][:], conj(synchdat0[0][:]))))

                synchdat = [p_est * kk for kk in synchdat0]
                tmp_2mat = matmul(self.del_mat_exp, diag(synchdat[0]))
                self.del_mat = matmul(tmp_2mat, conj(self.zadoff_chu))

                dmax_ind = argmax((abs(self.del_mat)))
                dmax_val = max((abs(self.del_mat)))

                if dmax_val > self.scale_factor_gate * len(synchdat[0]):
                    tim_synch_ind = self.time_synch_ref[0]
                    if ((P * self.stride_val + self.start_samp - tim_synch_ind > 2 * self.cp_len + self.nfft)
                            or self.corr_obs == -1):

                        self.corr_obs += 1

                        self.time_synch_ref[0] = P * self.stride_val + self.start_samp
                        self.time_synch_ref[1] = dmax_ind
                        self.time_synch_ref[2] = int(dmax_val)

                        del_vec = self.del_mat_exp[dmax_ind][:]
                        data_recov = matmul(diag(del_vec), synchdat[0])

                        zcwn = [(1.0 / self.SNR_lin) + qq for qq in ones(len(self.zadoff_chu))]
                        tmp_v1 = divide(matmul(diag(data_recov), conj(self.zadoff_chu)), zcwn)

                        chan_est00 = reshape(tmp_v1, (self.M[0], self.L_synch))
                        chan_est = sum(chan_est00, axis=0) / float(self.M[0])

                        chan_est1 = zeros((1, self.nfft), dtype=complex)
                        chan_est1[0][self.synch_bins_used_P] = chan_est
                        self.est_chan_freq_P[self.corr_obs][:] = chan_est1[0][:]

                        if self.count == 0 and self.channel_graph_plot == 1:
                            chan_q = self.give_genie_chan()
                            xax = array(range(0, self.nfft - 2))
                            yax1 = 20 * log10(abs(chan_est1))
                            yax2 = 20 * log10(abs(fft(chan_q, self.nfft)))
                            ypred = [x for i, x in enumerate(yax1[0, 1:]) if i != 31]
                            ygeni = [x for i, x in enumerate(yax2[0, 0, 1:]) if i != 31]

                            plt.plot(xax, ypred, 'r')
                            plt.plot(xax, ygeni, 'b')
                            plt.show()

                        chan_est_tim = ifft(chan_est1, self.nfft)

                        if self.save_channel_file == 1:
                            f = open(str(
                                self.directory_name) + '_' + str(self.file_name_cest), 'wb')
                            pickle.dump(chan_est_tim, f, protocol=2)
                            f.close()

                        self.est_chan_time[self.corr_obs][0:self.nfft] = chan_est_tim[0][0:self.nfft]
                        chan_mag = matmul(diag(chan_est), conj(chan_est))
                        eq_gain_0 = [1.0 / self.SNR + vv for vv in chan_mag]

                        self.eq_gain = divide(conj(chan_est), eq_gain_0)
                        self.eq_gain_ext = tile(self.eq_gain, self.M[0])
                        self.est_synch_freq[self.corr_obs][:] = matmul(diag(self.eq_gain_ext), data_recov)
                        break
        if self.count > 0:
            out0[:] = in0
            out1[0:size(self.time_synch_ref, 1)] = self.time_synch_ref[:, newaxis]

        self.count += 1
        return len(output_items[0])

    def give_genie_chan(self):
        h = zeros((self.num_ant_txrx, self.num_ant_txrx), dtype=object)
        channel_time = zeros((self.num_ant_txrx, self.num_ant_txrx, self.max_impulse), dtype=complex)
        channel_freq = zeros((self.num_ant_txrx, self.num_ant_txrx, int(self.nfft)), dtype=complex)
        if self.num_ant_txrx == 1:
            if self.channel == 'Ideal':
                h[0, 0] = array([1])
            elif self.channel == 'IMT1':
                h[0, 0] = array([0, 1])
            elif self.channel == 'Fading':
                h[0, 0] = array([0.3977, 0.7954 - 0.3977j, -0.1988, 0.0994, -0.0398])
            else:
                print('# Please select either Ideal or Fading for channel type.')
        elif self.num_ant_txrx == 2:
            if self.channel == 'Ideal':
                h[0, 0] = array([1])
                h[0, 1] = array([1])
                h[1, 0] = array([1])
                h[1, 1] = array([1])
            elif self.channel == 'IMT1':
                h[0, 0] = array([0, 1])
                h[0, 1] = array([0, 1])
                h[1, 0] = array([0, 1])
                h[1, 1] = array([0, 1])
            elif self.channel == 'Fading':
                h[0, 0] = array([0.3977, 0.7954 - 0.3977j, -0.1988, 0.0994, -0.0398])
                h[0, 1] = array([0.8423j, 0.5391, 0, 0, 0])
                h[1, 0] = array([0.1631, -0.0815 + 0.9784j, 0.0978, 0, 0])
                h[1, 1] = array([0.0572j, 0.3659j, 0.5717 - 0.5717j, 0.4574, 0])
            else:
                print('# Please select either Ideal or Fading for channel type.')
        for rx in range(self.num_ant_txrx):
            for tx in range(self.num_ant_txrx):
                channel_time[rx, tx, 0:len(h[rx, tx])] = h[rx, tx] / norm(h[rx, tx])
                channel_freq[rx, tx, :] = fft(channel_time[rx, tx, 0:len(h[rx, tx])], self.nfft)

        genie_chan_time = channel_time
        return genie_chan_time
