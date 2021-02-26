import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy
from synch_and_chan_est import synch_and_chan_est
from TXRX_Parameters import sdr_profile

num_ofdm_symb = 24
nfft = 64
cp_len = 16
num_synch_bins = nfft
num_data_bins = 60
synch_dat = [1, 3]
snr = 50
diagnostics = 1
genie = 1

pickle_directory = '/srv/LTE-GNU-Radio-Code/GNU-Radio-Repositories/TEST/GNU_RADIO_OFFLINE/'
pickle_file = 'tx_data_offline.pckl'
directory_name = '/srv/LTE-GNU-Radio-Code/GNU-Radio-Repositories/TEST/GNU_RADIO_OFFLINE/Output/'
file_name_cest = 'output_data.pckl'

with open(pickle_directory + pickle_file, 'rb') as info:
    tx_data = pickle.load(info)
out_buffer = np.zeros((1, len(tx_data)), dtype=complex)
# print(tx_data)

block = synch_and_chan_est(num_ofdm_symb, nfft, cp_len,
                 num_synch_bins, synch_dat, num_data_bins, snr, directory_name, file_name_cest, diagnostics, genie)
output_buffer = block.work(tx_data, out_buffer)
# block = SynchronizeAndEstimate.SynchronizeAndEstimate(time=time, case=0, diagnostics=True, freq_offset=0, bin_selection=[-4, -3, -2, -1, 1, 2, 3, 4], buffer_on=0, buffer_size=8.192e3,
#                                                           zchu_time=1, split_zchu_across_symbols=False, seed_value=4, Zadoff_Chu_Prime=[23, 41], snr=5, corr_ind_processing=0, synch_data_pattern=[2, 3])
# output_buffer = block.work(tx_data, out_buffer)

# plot_limits = [-3, 3]
# print("Shape of GNU Radio QPSK: ", output_buffer.shape)
# print(output_buffer)
# plt.plot(output_buffer.real, output_buffer.imag, '.')
# plt.xlim(plot_limits)
# plt.ylim(plot_limits)
# plt.xlabel('real axis')
# plt.ylabel('imaginary axis')
# plt.title('GNURadio Constellation Diagram')
# plt.show()
