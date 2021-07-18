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
legacy = False
pls = True
snr = 100
channel_profile = 'Fading'

nfft = 64
cp_len = 16
num_synch_bins = nfft - 2


def rx_signal_gen(pls_dictionary_case, buffer_tx_time):
    """
    Generates the time domain rx signal at each receive antenna (Convolution with channel and add noise)
    :param buffer_tx_time: Time domain tx signal streams on each antenna (matrix)
    :return buffer_rx_time: Time domain rx signal at each receive antenna
    """
    max_impulse = 64
    total_symb_len = 960
    pls_profiles = {
        0: {'bandwidth': 960e3,
            'bin_spacing': 15e3,
            'num_ant': 2,
            'bit_codebook': 1,
            'synch_data_pattern': [2, 1]}}
    bandwidth = pls_profiles[pls_dictionary_case]['bandwidth']
    bin_spacing = pls_profiles[pls_dictionary_case]['bin_spacing']
    num_ant = pls_profiles[pls_dictionary_case]['num_ant']
    bit_codebook = pls_profiles[pls_dictionary_case]['bit_codebook']
    synch_data_pattern = pls_profiles[pls_dictionary_case]['synch_data_pattern']

    NFFT = int(np.floor(bandwidth / bin_spacing))
    CP = int(0.25 * NFFT)
    num_data_bins = 4
    subband_size = num_ant

    DC_index = int(NFFT / 2)
    neg_data_bins = list(range(DC_index - int(num_data_bins / 2), DC_index))
    pos_data_bins = list(range(DC_index + 1, DC_index + int(num_data_bins / 2) + 1))
    used_data_bins = np.array(neg_data_bins + pos_data_bins)

    h = np.zeros((num_ant, num_ant), dtype=object)
    h_f = np.zeros((num_ant, num_ant, num_data_bins), dtype=complex)
    channel_time = np.zeros((num_ant, num_ant, max_impulse), dtype=complex)
    channel_freq = np.zeros((num_ant, num_ant, NFFT), dtype=complex)

    h[0, 0] = np.array([1])
    h[0, 1] = np.array([1])
    h[1, 0] = np.array([1])
    h[1, 1] = np.array([1])

    for rx in range(num_ant):
        for tx in range(num_ant):
            channel_time[rx, tx, 0:len(h[rx, tx])] = h[rx, tx] / np.linalg.norm(h[rx, tx])

            channel_freq[rx, tx, :] = np.fft.fft(channel_time[rx, tx, 0:len(h[rx, tx])], NFFT)
            h_f[rx, tx, :] = channel_freq[rx, tx, used_data_bins.astype(int)]

    buffer_rx_time = np.zeros((num_ant, total_symb_len + max_impulse - 1), dtype=complex)
    for rx in range(num_ant):
        rx_sig_ant = 0  # sum rx signal at each antenna
        for tx in range(num_ant):
            chan = channel_time[rx, tx, :]
            tx_sig = buffer_tx_time[tx, :]
            rx_sig_ant += np.convolve(tx_sig, chan)

        buffer_rx_time[rx, :] = rx_sig_ant

    return buffer_rx_time


if pls is True:
    pvt_info_length = 8
    pls_dictionary_case = 0
    block = PhyLayerSecAIO(pvt_info_length, pls_dictionary_case)
    tx_data = np.zeros((2, 960), dtype=complex)
    out_buffer = np.zeros((2, 960), dtype=complex)
    for n in range(5):
        output_buffer = block.work(tx_data, out_buffer)
        tx_chan_noise = rx_signal_gen(0, output_buffer)
        for ant in range(2):
            pass
            # plt.plot(output_buffer[ant].real)
            # plt.plot(output_buffer[ant].imag)
            # plt.show()
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




