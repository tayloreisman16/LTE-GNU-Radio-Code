import numpy as np


class SynchSignal:
    def __init__(self, len_CP, num_synch_bins, num_ant_txrx, NFFT, synch_data):

        self.synch_state = 0
        self.len_CP = len_CP
        self.num_used_bins = num_synch_bins
        self.num_ant = num_ant_txrx
        self.NFFT = NFFT

        self.used_bins0 = list(range(int(-self.num_used_bins / 2), 0)) + list(range(1, int(self.num_used_bins / 2) + 1))
        self.used_bins = ((self.NFFT + np.array(self.used_bins0)) % self.NFFT)
        # print(self.used_bin_ind[1000:1100].astype(int))

        self.synch_data = synch_data

        # Array - first element is number of synchs in each synch-data pattern.
        # Second element is number of synch bins
        self.M = np.array([self.synch_data[0], self.num_used_bins])
        self.MM = np.product(self.M)
        self.prime = 23

        x0 = np.array(range(0, int(self.MM)))
        x1 = np.array(range(1, int(self.MM) + 1))
        if self.MM % 2 == 0:
            self.ZChu0 = np.exp(-1j * (2 * np.pi / self.MM) * self.prime * (x0**2 / 2))
        else:
            self.ZChu0 = np.exp(-1j * (2 * np.pi / self.MM) * self.prime * (x0 * x1) / 2)

        # print(len(self.ZChu0))

        self.ZChu1 = np.zeros((self.num_ant, int(self.NFFT)), dtype=complex)
        # print(self.ZChu1[0, self.used_bin_ind.astype(int)])
        for ant in range(self.num_ant):
            # print(ant)
            self.ZChu1[0, self.used_bins.astype(int)] = self.ZChu0[0: int(self.M[1])]
