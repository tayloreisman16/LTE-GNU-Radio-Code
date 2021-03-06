import numpy as np
import matplotlib.pyplot as plt


class RxBasebandSystem:
    def __init__(self, multi_ant_sys, Caz, param_est, case):
        self.num_ant_txrx = multi_ant_sys.num_ant_txrx #o
        self.MIMO_method = multi_ant_sys.MIMO_method
        self.NFFT = multi_ant_sys.NFFT
        self.len_CP = multi_ant_sys.len_CP #o

        self.rx_buff_len = self.NFFT + self.len_CP
        self.rx_buffer_time0 = multi_ant_sys.buffer_data_rx_time[::]
        print(multi_ant_sys.buffer_data_rx_time.shape)
        print(self.rx_buffer_time0.shape)

        self.used_bins_data = multi_ant_sys.used_bins_data
        self.SNR = multi_ant_sys.SNR_lin

        self.channel_freq = multi_ant_sys.channel_freq
        self.symbol_pattern = multi_ant_sys.symbol_pattern

        # Dont know what these are right now
        self.UG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        self.SG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        self.VG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))

        self.U = multi_ant_sys.U
        self.S = multi_ant_sys.S
        self.V = multi_ant_sys.V

        self.stream_size = multi_ant_sys.stream_size
        self.noise_var = multi_ant_sys.noise_var

        self.ref_only_bins = multi_ant_sys.ref_only_bins  #i
        self.chan_max_offset = multi_ant_sys.chan_max_offset  #i
        self.h_f = multi_ant_sys.h_f  # channel FFT #dontneed
        self.genie_chan_time = multi_ant_sys.genie_chan_time  #dontneed

        self.synch_ref = Caz.ZChu0 #i (import file)
        self.used_bins_synch = multi_ant_sys.used_bins_synch  # Same as Caz.used_bins.astype(int) #i

        # window: CP to end of symbol
        self.ptr_o = np.array(range(self.len_CP, self.len_CP + self.NFFT)).astype(int)
        self.ptr_i = self.ptr_o - np.ceil(self.len_CP / 2).astype(int)

        # Start from the middle of the CP
        self.rx_buffer_time = self.rx_buffer_time0[:, self.ptr_i]

        # print(self.rx_buffer_time)
        lmax_s = int(len(self.symbol_pattern) - sum(self.symbol_pattern))
        lmax_d = int(sum(self.symbol_pattern))

        self.time_synch_ref = np.ones((self.rx_buffer_time0.shape[0], 3))  # NEED TO CHANGE THIS IN GNURADIO
        self.time_synch_ref = np.zeros((self.num_ant_txrx, lmax_s, 2))  # ONE OF THESE 2 WILL BE REMOVED

        '''obj.EstChanFreqP=zeros(obj.MIMOAnt,LMAXS,obj.Nfft);
           obj.EstChanFreqN=zeros(obj.MIMOAnt, LMAXS,length(obj.SynchBinsUsed));
           obj.EstChanTim=zeros(obj.MIMOAnt, LMAXS,2);
           obj.EstSynchFreq=zeros(obj.MIMOAnt, LMAXS,length(obj.SynchBinsUsed));'''

        self.est_chan_freq_p = np.zeros((self.num_ant_txrx, lmax_s, self.NFFT), dtype=complex)
        self.est_chan_freq_n = np.zeros((self.num_ant_txrx, lmax_s, len(self.used_bins_synch)), dtype=complex)
        self.est_chan_time = np.zeros((self.num_ant_txrx, lmax_s, 2), dtype=complex)
        self.est_synch_freq = np.zeros((self.num_ant_txrx, lmax_s, len(self.used_bins_synch)), dtype=complex)

        # print(self.est_chan_freq_p.shape, self.est_chan_freq_n.shape, self.est_chan_time.shape, self.est_synch_freq.shape)
        if self.num_ant_txrx == 1:
            self.est_data_freq = np.zeros((self.num_ant_txrx, lmax_d, len(self.used_bins_data)), dtype=complex)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            self.est_data_freq = np.zeros((1, lmax_d, len(self.used_bins_data)), dtype=complex) #pass
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SPMult':
            self.est_data_freq = np.zeros((2, lmax_d, len(self.used_bins_data)), dtype=complex) #pass

        # Max length of channel impulse is CP
        self.est_chan_impulse = np.zeros((self.num_ant_txrx, lmax_s, self.NFFT), dtype=complex)

        self.M = Caz.M.astype(int)
        self.synch_data = Caz.synch_data
        self.synch_state = Caz.synch_state

        self.param_est = param_est
        self.case = case

        self.SNR_analog = multi_ant_sys.SNR_analog
        self.stride_val = None
        self.corr_obs = None
        self.start_samp = None
        self.del_mat = None

    def param_est_synch(self, sys_model):


        self.stride_val = np.ceil(self.len_CP / 2)
        self.time_synch_ref = np.zeros((self.num_ant_txrx, 250, 3))  # There are two more in the init.

        for m in range(1):
            self.corr_obs = -1

            chan_q = self.genie_chan_time[m, 0, :] # 2048
            self.start_samp = (self.len_CP - 4) - 1

            total_loops = int(np.ceil(self.rx_buffer_time0.shape[1] / self.stride_val))
            # print(total_loops)
            d_long = np.zeros(total_loops)

            ptr_adj, loop_count, sym_count = 0, 0, 0

            tap_delay = 5
            x = np.zeros(tap_delay)
            ptr_synch0 = np.zeros(1000)
            while loop_count <= total_loops:
                print(loop_count)
                if self.corr_obs == -1:
                    ptr_frame = loop_count * self.stride_val + self.start_samp + ptr_adj
                elif self.corr_obs < 5:
                    ptr_frame += sum(self.synch_data) * (self.NFFT + self.len_CP)
                else:
                    ptr_frame = (np.ceil(np.dot(xp[-1:], b) - self.len_CP/4))[0]


                # print(self.rx_buffer_time0.shape[1])
                if (self.M[0] - 1) * self.rx_buff_len + self.NFFT + ptr_frame < self.rx_buffer_time0.shape[1]:
                    # if (self.MM[0] - 1)*self.rx_buff_len + self.NFFT + ptr_frame - 1 < self.rx_buffer_time0.shape[1]:
                    for i in range(self.M[0]):
                        # print(i)
                        start = int(i * self.rx_buff_len + ptr_frame)
                        fin = int(i * self.rx_buff_len + ptr_frame + self.NFFT)
                        self.rx_buffer_time[i * self.NFFT: (i + 1) * self.NFFT] = self.rx_buffer_time0[m, start:fin]

                    # Take FFT of the window
                    fft_vec = np.zeros((self.M[0], self.NFFT), dtype=complex)
                    for i in range(self.M[0]):
                        start = i * self.NFFT
                        fin = (i + 1) * self.NFFT
                        fft_vec[i, 0:self.NFFT] = np.fft.fft(self.rx_buffer_time[start: fin], self.NFFT)

                    synch_dat00 = fft_vec[:, self.used_bins_synch]
                    synch_dat0 = np.reshape(synch_dat00, (1, synch_dat00.shape[0] * synch_dat00.shape[1]))
                    pow_est = sum(sum(synch_dat0 * np.conj(synch_dat0))).real / synch_dat0.shape[1]
                    synch_dat = synch_dat0 / np.sqrt(pow_est)

                    # from transmit antenna 1 only?
                    chan_freq0 = np.reshape(self.channel_freq[m, 0, self.used_bins_synch], (1, np.size(self.used_bins_synch)))

                    chan_freq = np.tile(chan_freq0, (1, self.M[0]))

                    bins = self.used_bins_synch[:, None]
                    cp_dels = np.array(range(self.len_CP + 1))[:, None]
                    p_mat0 = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(bins, cp_dels.T))

                    p_mat = np.tile(p_mat0, (self.M[0], 1))

                    # maybe replace index 0 with m
                    self.del_mat = np.dot(np.conj(self.synch_ref)[None, :], np.dot(np.diag(synch_dat[0]), p_mat))
                    dd = abs(self.del_mat[0, :])
                    dmax, dmax_ind0 = dd.max(0), dd.argmax(0)
                    dmax_ind = dmax_ind0 - 1
                    d_long[loop_count] = dmax
                    # print('no')
                    if dmax > 0.5 * synch_dat.shape[1] or self.corr_obs > -1:
                        if dmax_ind > np.ceil(0.75 * self.len_CP):
                            if self.corr_obs == 0:
                                ptr_adj += np.ceil(0.5 * self.len_CP)
                                ptr_frame = loop_count * self.stride_val + self.start_samp + ptr_adj
                            elif self.corr_obs < 5:
                                ptr_frame += np.ceil(0.5 * self.len_CP)

                            # Take FFT of the window
                            fft_vec = np.zeros((self.M[0], self.NFFT), dtype=complex)
                            for i in range(self.M[0]):
                                start = i * self.NFFT
                                fin = (i + 1) * self.NFFT
                                fft_vec[i, 0:self.NFFT] = np.fft.fft(self.rx_buffer_time[start: fin], self.NFFT)

                            synch_dat00 = fft_vec[:, self.used_bins_synch]
                            synch_dat0 = np.reshape(synch_dat00, (1, synch_dat00.shape[0] * synch_dat00.shape[1]))
                            pow_est = sum(sum(synch_dat0 * np.conj(synch_dat0))).real / synch_dat0.shape[1]
                            synch_dat = synch_dat0 / np.sqrt(pow_est)

                            # from transmit antenna 1 only?
                            chan_freq0 = np.reshape(self.channel_freq[m, 0, self.used_bins_synch],
                                                    (1, np.size(self.used_bins_synch)))

                            chan_freq = np.tile(chan_freq0, (1, self.M[0]))

                            bins = self.used_bins_synch[:, None]
                            cp_dels = np.array(range(self.len_CP + 1))[:, None]
                            p_mat0 = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(bins, cp_dels.T))

                            p_mat = np.tile(p_mat0, (self.M[0], 1))

                            # maybe replace index 0 with m
                            self.del_mat = np.dot(np.conj(self.synch_ref)[None, :],
                                                  np.dot(np.diag(synch_dat[0]), p_mat))
                            dd = abs(self.del_mat[0, :])
                            dmax, dmax_ind0 = dd.max(0), dd.argmax(0)
                            dmax_ind = dmax_ind0 - 1
                            d_long[loop_count] = dmax

                        time_synch_ind = self.time_synch_ref[m, max(self.corr_obs, 1), 0]

                        if ptr_frame - time_synch_ind > (2 * self.len_CP + self.NFFT) or self.corr_obs == -1:
                            self.corr_obs += 1

                            self.time_synch_ref[m, self.corr_obs, 0] = ptr_frame
                            self.time_synch_ref[m, self.corr_obs, 1] = dmax_ind
                            self.time_synch_ref[m, self.corr_obs, 2] = dmax

                            ptr_synch0[sym_count % tap_delay] = sum(self.time_synch_ref[m, self.corr_obs, 0:2])
                            x[sym_count % tap_delay] = sym_count * sum(sys_model.synch_data)  # No need for +1 on lhs
                            sym_count += 1

                            x2 = x[0:min(self.corr_obs, tap_delay)]
                            x_plus = np.concatenate((x2, np.atleast_1d(sym_count * sum(sys_model.synch_data))))
                            xp = np.zeros((len(x_plus), 2))
                            xp[:, 0] = np.ones(len(x_plus))
                            xp[:, 1] = x_plus

                            if self.corr_obs > 3:
                                y = ptr_synch0[0:min(tap_delay, self.corr_obs)]
                                # print(y)
                                X = np.zeros((len(x2), 2))
                                X[:, 0] = np.ones(len(x2))
                                X[:, 1] = x2

                                b = np.linalg.lstsq(X, y)[0]
                                print(b)

                            # recovered data with delay removed - DataRecov in MATLAB code
                            data_recov0 = np.dot(np.diag(synch_dat[0]), p_mat[:, dmax_ind+1])  #

                            h_est1 = np.zeros((self.NFFT, 1), dtype=complex)
                            # TmpV1 in MATLAB code
                            data_recov = (data_recov0 * np.conj(self.synch_ref)) / (1 + (1/self.SNR))

                            h_est00 = np.reshape(data_recov, (data_recov.shape[0], self.M[0]))
                            h_est0 = h_est00.T

                            h_est = np.sum(h_est0, axis=0)/(self.M[0])

                            h_est1[self.used_bins_synch, 0] = h_est

                            self.est_chan_freq_p[m, self.corr_obs, 0:len(h_est1)] = h_est1[:, 0]
                            self.est_chan_freq_n[m, self.corr_obs, 0:len(h_est)] = h_est

                            # if sys_model.diagnostic == 1 and loop_count == 0:
                            #     xax = np.array(range(0, self.NFFT)) * sys_model.fs/self.NFFT
                            #     yax1 = 20*np.log10(abs(h_est1))
                            #     yax2 = 20*np.log10(abs(np.fft.fft(chan_q, self.NFFT)))
                            #
                            #     plt.plot(xax, yax1, 'r')
                            #     plt.plot(xax, yax2, 'b')
                            #     plt.show()

                            h_est_time = np.fft.ifft(h_est1[:, 0], self.NFFT)
                            self.est_chan_impulse[m, self.corr_obs, 0:len(h_est_time)] = h_est_time

                            h_est_ext = np.tile(h_est, (1, self.M[0])).T
                            print("equalized synch")

                            synch_equalized = (data_recov0 * np.conj(h_est_ext[:, 0])) / ((np.conj(h_est_ext[:, 0])*h_est_ext[:, 0]) + (1/self.SNR))
                            self.est_synch_freq[m, self.corr_obs, 0:len(self.used_bins_synch)*self.M[0]] = synch_equalized

                            if sys_model.diagnostic == 1 and loop_count == 0:
                                plt.plot(synch_equalized.real, synch_equalized.imag, '.')
                                plt.show()





                loop_count += 1
                # print(loop_count)

    def rx_data_demod(self):
        if self.num_ant_txrx == 1:
            m = 0 # Just an antenna index
            for p in range(self.corr_obs + 1):
                for data_sym in range(self.synch_data[1]):
                    if sum(self.time_synch_ref[m, p, :]) + self.NFFT < self.rx_buffer_time0.shape[1]:
                        data_ptr = int(self.time_synch_ref[m, p, 0] + (data_sym+1)*self.rx_buff_len)
                        self.rx_buffer_time = self.rx_buffer_time0[m, data_ptr: data_ptr + self.NFFT] # -1

                        fft_vec = np.fft.fft(self.rx_buffer_time, self.NFFT)

                        freq_dat0 = fft_vec[self.used_bins_data]

                        p_est = sum(freq_dat0 * np.conj(freq_dat0)) / len(freq_dat0)

                        data_recov0 = freq_dat0 / np.sqrt(p_est)

                        if self.param_est == 'Estimated':
                            print('hello')
                            h_est = self.est_chan_freq_p[m, p, self.used_bins_data]
                        elif self.param_est == 'Ideal':
                            print('bye')
                            h_est = self.h_f[m, 0, :]

                        del_rotate = np.exp(1j*2*(np.pi/self.NFFT)*self.used_bins_data * self.time_synch_ref[m, p, 1])
                        data_recov = np.dot(np.diag(data_recov0), del_rotate)

                        data_equalized = (data_recov * np.conj(h_est)) / ((np.conj(h_est)*h_est) + (1/self.SNR))
                        self.est_data_freq[m, p*self.synch_data[1] + data_sym, 0:len(self.used_bins_data)] = data_equalized

                        data = self.est_data_freq[m, p, 0:len(self.used_bins_data)]
                        p_est1 = sum(data * np.conj(data)) / len(data)

                        self.est_data_freq[m, p*self.synch_data[1] + data_sym, 0:len(self.used_bins_data)] /= np.sqrt(p_est1)



        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            print('STCode currently not supported')
            exit(0)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SpMult':
            print('SpMult currently not supported')
            exit(0)
