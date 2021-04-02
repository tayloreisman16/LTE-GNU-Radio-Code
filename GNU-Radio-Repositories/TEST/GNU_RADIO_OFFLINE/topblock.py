import numpy as np
import pickle
import matplotlib.pyplot as plt
from synch_and_chan_est import SynchAndChanEst
# from TXRX_Parameters import sdr_profile

num_ofdm_symb = 24
nfft = 64
cp_len = 16
num_synch_bins = nfft - 2
num_data_bins = 60
synch_dat = [1, 3]
channel = 'IMT16'
snr = 100
scale_factor_gate = 0.60
diagnostics = 1
genie = 1

pickle_directory = '/srv/LTE-Code-Offline/Data/'
pickle_file = 'tx_data_offline' + '_chan_type_' + channel + '_SNR_' + str(snr) + '.pckl'
directory_name = '/srv/LTE-GNU-Radio-Code/GNU-Radio-Repositories/TEST/GNU_RADIO_OFFLINE/Output/'
file_name_cest = 'output_data.pckl'

with open(pickle_directory + pickle_file, 'rb') as info:
    tx_data = pickle.load(info)
print(tx_data.shape)

plt.plot(tx_data[0][:].real)
plt.plot(tx_data[0][:].imag)
plt.title('TX Data')
plt.xlabel('Samples(t)')
plt.ylabel('Amplitude')
plt.show()

out_buffer = np.zeros((1, len(tx_data[0][:])), dtype=complex)

block = SynchAndChanEst(num_ofdm_symb, nfft, cp_len,
                        num_synch_bins, synch_dat, num_data_bins, channel, snr, scale_factor_gate, directory_name, file_name_cest, diagnostics,
                        genie)
output_buffer = block.work(tx_data, out_buffer)
