import numpy as np
from SystemModel import SystemModel
from OFDM import OFDM
from SynchSignal import SynchSignal
from MultiAntennaSystem import MultiAntennaSystem
from RxBasebandSystem import RxBasebandSystem
import matplotlib.pyplot as plt
import pickle

SNR = 50  # dB

num_cases = 1

SDR_profiles = {0: {'system_scenario': '4G5GSISO-TU',
                    'diagnostic': 1,
                    'wireless_channel': 'Fading',
                    'channel_band': 960e3,
                    'bin_spacing': 15e3,
                    'channel_profile': 'LTE-TU',
                    'CP_type': 'Normal',
                    'num_ant_txrx': 1,
                    'param_est': 'Estimated',
                    'MIMO_method': 'SpMult',
                    'SNR': SNR,
                    'ebno_db': [100, 100, 100, 100, 100, 100, 100, 100, 100],
                    'num_symbols': [48, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                    'stream_size': 1},
                1: {'system_scenario': 'WIFIMIMOSM-A',
                    'diagnostic': 0,
                    'wireless_channel': 'Fading',
                    'channel_band': 0.9 * 20e6,
                    'bin_spacing': 312.5e3,
                    'channel_profile': 'Indoor A',
                    'CP_type': 'Extended',
                    'num_ant_txrx': 2,
                    'param_est': 'Ideal',
                    'MIMO_method': 'SpMult',
                    'SNR': SNR,
                    'ebno_db': [6, 7, 8, 9, 10, 14, 16, 20, 24],
                    'num_symbols': [10, 10, 10, 10, 10, 10, 10, 10, 10],
                    'stream_size': 2}}

for case in range(num_cases):
    sys_model = SystemModel(SDR_profiles[case])

    if SDR_profiles[case]['diagnostic'] == 0:
        loop_runs = len(SDR_profiles[case]['ebno_db'])
    else:
        loop_runs = 1

    for loop_iter in range(loop_runs):
        sig_datatype = sys_model.sig_datatype
        chan_type = sys_model.wireless_channel
        phy_chan = sys_model.phy_chan
        NFFT = sys_model.NFFT

        num_bins0 = sys_model.num_bins0  # Max umber of occupied bins for data
        num_bins1 = 4 * np.floor(num_bins0 / 4)  # Make number of bins a multiple of 4 for MIMO

        # positive and negative bin indices
        all_bins = np.array(list(range(-int(num_bins1 / 2), 0)) + list(range(1, int(num_bins1 / 2) + 1)))
        # positive and negative bin indices
        ref_bins0 = np.random.randint(1, int(num_bins1 / 2) + 1, size=int(np.floor(num_bins1 * sys_model.ref_sigs / 2)))
        ref_bins = np.unique(ref_bins0)
        # positive and negative bin indices
        ref_only_bins = np.sort(np.concatenate((-ref_bins, ref_bins)))  # Bins occupied by pilot (reference) signals
        # positive and negative bin indices - converted to & replaced by positive only in MultiAntennaSystem class
        data_only_bins = np.setdiff1d(all_bins, ref_only_bins)  # Actual bins occupied by data

        num_used_bins = len(data_only_bins)
        modulation_type = sys_model.modulation_type
        bits_per_bin = sys_model.bits_per_bin

        if SDR_profiles[case]['diagnostic'] == 0:
            SNR_dB = sys_model.ebno_db[loop_iter]
        else:
            SNR_dB = SNR

        synch_data = sys_model.synch_data
        num_synchdata_patterns = int(np.ceil(sys_model.num_symbols[loop_iter] / sum(synch_data)))
        num_symbols = sum(synch_data) * num_synchdata_patterns

        # 0 - synch symbol, 1 - data symbol
        symbol_pattern0 = np.concatenate((np.zeros(synch_data[0]), np.ones(synch_data[1])))
        symbol_pattern = np.tile(symbol_pattern0, num_synchdata_patterns)

        # sum of symbol_pattern gives the total number of data symbols

        binary_info = np.random.randint(0, 2, (
        sys_model.stream_size, int(sum(symbol_pattern) * num_used_bins * bits_per_bin)))

        fs = sys_model.fs  # Sampling frequency
        Ts = 1 / fs  # Sampling period

        delta_f = sys_model.bin_spacing
        if sys_model.CP_type == 'Normal':
            len_CP = round(NFFT / 4)  # cyclic prefix (IN Samples !!)
        elif sys_model.CP_type == 'Extended':
            len_CP = round(NFFT / 4 + NFFT / 8)  # cyclic prefix (IN Samples !!)
        else:
            print('Wrong CP Type')
            exit(0)

        num_ant_txrx = sys_model.num_ant_txrx
        param_est = sys_model.param_est
        MIMO_method = sys_model.MIMO_method
        num_synch_bins = sys_model.num_synch_bins
        # print(num_synch_bins)
        channel_profile = sys_model.channel_profile

        # OFDM class object
        OFDM_data = OFDM(len_CP, num_used_bins, modulation_type, NFFT, delta_f)

        diagnostic = sys_model.diagnostic
        wireless_channel = sys_model.wireless_channel
        stream_size = sys_model.stream_size

        # print(all_bins)
        # syschan
        # MultiAntennaSystem class object
        multi_ant_sys = MultiAntennaSystem(OFDM_data, num_ant_txrx, MIMO_method, all_bins, num_symbols, symbol_pattern,
                                           fs, channel_profile, diagnostic, wireless_channel, stream_size,
                                           data_only_bins, ref_only_bins)

        # SynchSignal class object
        Caz = SynchSignal(len_CP, num_synch_bins, num_ant_txrx, NFFT, synch_data)

        multi_ant_sys.multi_ant_binary_map(Caz, binary_info, synch_data)

        # At this point multi_ant_sys.buffer_data_tx contains all the transmit synch and QPSK data
        # placed in the corect bins
        # plt.plot(multi_ant_sys.buffer_data_tx.real, multi_ant_sys.buffer_data_tx.imag, '.')
        # plt.show()
        multi_ant_sys.multi_ant_symb_gen(num_symbols)

        # **** multi_ant_sys.buffer_data_tx_time is the variable to pckl for GNURadio transmitter **** #
        f = open('4g5g_input_data.pckl', 'wb')
        pickle.dump(multi_ant_sys.buffer_data_tx_time, f, protocol=2)
        f.close()
        print("Shape: ", multi_ant_sys.buffer_data_tx_time.shape)

        # Receive signal after convolution with channel
        multi_ant_sys.rx_signal_gen()

        # Receive signal with noise added
        multi_ant_sys.additive_noise(sys_model.SNR_type, SNR_dB, wireless_channel, sys_model.sig_datatype)

        rx_sys = RxBasebandSystem(multi_ant_sys, Caz, param_est, case)

        rx_sys.param_est_synch(sys_model)
        # print('Number of synch symbols found', rx_sys.corr_obs)
        # rx_sys.corr_obs is one less than the total number of synchs present in the buffer
        rx_sys.rx_data_demod()

        if sys_model.diagnostic == 1:
            # IQ plot
            rx_newshape = rx_sys.est_data_freq.shape[0] * rx_sys.est_data_freq.shape[1] * rx_sys.est_data_freq.shape[2]
            rx_phasors = np.reshape(rx_sys.est_data_freq, (1, rx_newshape))

            plt.plot(rx_phasors[0, :].real, rx_phasors[0, :].imag, '.')
            plt.show()

        dbg77 = 1
