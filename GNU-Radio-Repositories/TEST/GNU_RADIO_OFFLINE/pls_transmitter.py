import numpy as np

class PhyLayerSecTransmitter:
    def __init__(self, bandwidth, bin_spacing, num_ant, bit_codebook, NFFT, CP, num_used_bins):
        gr.sync_block.__init__(self,
                               name="PhyLayerSecTransmitter",
                               in_sig=[complex64],
                               out_sig=[complex64])

        self.message_port_register_in(pmt.intern('msg_in'))
        self.set_msg_handler(pmt.intern('msg_in'), self.handle_msg)

        self.bandwidth = bandwidth
        self.bin_spacing = bin_spacing
        self.num_ant = num_ant
        self.bit_codebook = bit_codebook  # TODO: Add function to calculate codebook!

        self.NFFT = NFFT
        self.CP = CP
        self.OFDMsymb_len = self.NFFT + self.CP
        self.num_used_bins = num_used_bins

        self.used_data_bins = self.NFFT - 2
        self.subband_size = self.num_ant

        self.num_subbands = int(floor(self.num_used_bins/self.subband_size))
        self.num_PMI = self.num_subbands

        self.key_len = self.num_subbands * self.bit_codebook

    def work(self, input_items, output_items):
        in0 = input_items[0]

        rx_state = state
        tx_sig = []

        if tx_node == '' and len(args) == 4:
            ## 1. Alice to Bob #TODO: State 1 (GR SENT)
            print('Entering State 1')
            GA = self.unitary_gen()
            tx_sig = GA

        if tx_node == 'Bob' and len(args) == 5:
            ## 2. Bob to Alice #TODO: State 3 (G1r)
            print('Entering State 3')
            FB = in0


        elif tx_node == 'Alice' and len(args) == 5:
            ## 3. Alice to Bob #TODO: State 5 (G2r/Private Data)
            print('Entering State 5')
            FA = in0
            for sb in range(0, self.num_subbands):
                tx_sig = np.dot(np.conj(UA[sb]), np.conj(FA[sb]).T)


        out[:] = tx_sig
        return (output_items[0])

    def unitary_gen(self):
        """
        Generate random nitary matrices for each sub-band
        :return GA: Unitary matrices in each sub-band at Alice
        """
        GA = zeros(self.num_subbands, dtype=object)
        for sb in range(0, self.num_subbands):
            Q, R = qr(uniform(0, 1, (self.num_ant, self.num_ant))
                      +1j*uniform(0, 1, (self.num_ant, self.num_ant)))

            GA[sb] = dot(Q, diag(diag(R)/abs(diag(R))))
        return GA
