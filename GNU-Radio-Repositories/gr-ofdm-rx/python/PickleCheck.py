import numpy as np
import pickle
# import SDRParameters
# import MultiAntennaSystem
# import ChannelModel

f = open('/home/utsa/Desktop/softbit0_2018_02_02_11h_06m.pckl', 'rb')
tx_data_mat = pickle.load(f, encoding='latin1')
print(tx_data_mat[:][:])
print(tx_data_mat.shape)