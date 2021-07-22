import numpy as np
import pickle
import matplotlib.pyplot as plt
from synch_and_chan_est import SynchAndChanEst
from pls_aio import PhyLayerSecAIO
from PLSParameters import PLSParameters
# from find_synch_index import SynchronizeIndex
# from channel_estimate import ChannelEstimate
# from pls_receiver import PhyLayerSecReceiver
# from TXRX_Parameters import sdr_profile
from rx_process import PLSRxProcess
legacy = False
pls = True
snr = 1000
channel_profile = 'Fading'
channel_select = 1

nfft = 64
cp_len = 16
num_synch_bins = nfft - 2


if pls is True:
    pvt_info_length = 8
    pls_dictionary_case = 0
    block = PhyLayerSecAIO(pvt_info_length, pls_dictionary_case)
    tx_data = np.zeros((2, 960), dtype=complex)
    out_buffer = np.zeros((2, 960), dtype=complex)
    rx_process = PLSRxProcess(0)
    for n in range(3):
        output_buffer = block.work(tx_data, out_buffer)
        tx_chan_noise = rx_process.rx_signal_gen(output_buffer, channel_select)
        if n == 1:
            for ant in range(2):
                plt.plot(tx_chan_noise[ant, :].real)
                plt.plot(tx_chan_noise[ant, :].imag)
                plt.title(f'Signal After AWGN for Antenna {ant}')
                plt.show()

        tx_data = tx_chan_noise

if pls is False and legacy is True:
    num_ofdm_symb = 24
    scale_factor_gate = 0.7
    plot_iq = 1
    save_channel_file = 0

    pickle_directory = 'Data/'
    pickle_file = 'tx_data_offline' + '_chan_type_' + channel_profile + '_SNR_' + str(snr) + '.pckl'
    directory_name = 'Output/'
    file_name_cest = 'output_data.pckl'

    block = SynchAndChanEst(num_ofdm_symb, nfft, cp_len,
                            num_synch_bins, synch_data, num_data_bins, channel_profile, snr, scale_factor_gate, directory_name,
                            file_name_cest, plot_iq, true_channel_graphing, perfect_channel_estimation, save_channel_file)
elif pls is False and legacy is False:
    num_ofdm_symb = 24
    scale_factor_gate = 0.7
    plot_iq = 1
    save_channel_file = 0

    pickle_directory = 'Data/'
    pickle_file = 'tx_data_offline' + '_chan_type_' + channel_profile + '_SNR_' + str(snr) + '.pckl'
    directory_name = 'Output/'
    file_name_cest = 'output_data.pckl'

    #block0
    #block1
    #block2
    pass
# elif pls is True:
#     pvt_info_len = 144
#
#     pickle_directory = 'Data/'
#     pickle_file = f'pls_data_offline_SNR_{snr}dB.pckl'
#     directory_name = 'Output/'
#     file_name_cest = 'output_data.pckl'
#
#     block = PhyLayerSecReceiver(1)




# with open(pickle_directory + pickle_file, 'rb') as info:
#     tx_data = pickle.load(info)
# print(tx_data.shape)

# plt.plot(tx_data[0][:].real)
# plt.plot(tx_data[0][:].imag)
# plt.title('TX Data')
# plt.xlabel('Samples(t)')
# plt.ylabel('Amplitude')
# plt.show()
# out_buffer = np.zeros((1, len(tx_data[0][:])), dtype=complex)




