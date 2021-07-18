from numpy import zeros, ones, array, tile, reshape, size, newaxis, delete, concatenate,  \
    sum, prod, divide, sqrt, exp, log10,     \
    floor,   \
    conj, matmul, diag, outer, dot, product, \
    complex64, int16,   \
    pi
from numpy.random import uniform
from numpy.fft import fft, ifft
from numpy.linalg import qr
import pickle as pckl
import pmt

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
        self.n_synch_bins = NFFT - 2
        self.pattern = [1, 3]
        self.synch_mode = 'TDD'
        self.n_patterns = 1
        self.n_symbols = 5
        self.l_symbol = self.NFFT + self.CP
        self.l_buffer = self.n_symbols * self.l_symbols
        self.symb_pattern = concatenate((zeros(self.pattern[0]), ones(self.pattern[1])))
        self.symb_pattern_ext = tile(self.symb_pattern, self.n_patterns)
        self.num_used_bins = num_used_bins

        self.used_data_bins = self.NFFT - 2
        self.subband_size = self.num_ant

        self.num_subbands = int(floor(self.num_used_bins/self.subband_size))
        self.num_PMI = self.num_subbands

        self.key_len = self.num_subbands * self.bit_codebook

        # Receive Messaging Ports Implemented Here
        self.message_port_register_in(pmt.intern("RXState"))
        self.set_msg_handler(pmt.intern('RXState'), self.handle_msg)



    def handle_msg(self, msg):
        self.message_port_pub(pmt.intern('msg_out'), pmt.intern('message received!'))

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        rx_state = state
        data = pckl
        tx_sig = []

        if rx_state == 'Start':
            ## 1. Alice to Bob #TODO: State 1 (GR SENT)
            print('Entering State 1')
            GA = self.unitary_gen()
            tx_sig = GA

        if rx_state == 'S2Fin':
            ## 2. Bob to Alice #TODO: State 3 (G1r)
            print('Entering State 3')
            FB = in0


        elif rx_state == 'S4Fin':
            ## 3. Alice to Bob #TODO: State 5 (G2r/Private Data)
            print('Entering State 5')
            FA = in0
            UA =
            for sb in range(0, self.num_subbands):
                tx_sig = dot(conj(UA[sb]), conj(FA[sb]).T)


        out[:] = tx_sig[:]
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

    def synch_data_mux(self, time_ofdm_data_symbols):
        """
        Multiplexes synch and data symbols according to the symbol pattern
        Takes synch mask from synch class and inserts dats symbols next to the synchs
        :param time_ofdm_data_symbols: NFFT+CP size OFDM data symbols
        :return: time domain tx symbol stream per antenna (contains synch and data symbols) (matrix)
        """
        buffer = zeros((self.num_ant, self.l_buffer)) + 1j * zeros((self.num_ant, self.l_buffer))

        total_symb_count = 0
        synch_symb_count = 0
        data_symb_count = 0
        for symb in self.symb_pattern_ext.tolist():
            symb_start = total_symb_count * self.l_symbol
            symb_end = symb_start + self.l_symbol
            # print(symb_start, symb_end)
            if int(symb) == 0:
                buffer[:, symb_start: symb_end * 2] = cp_block
                synch_symb_count += 2
                total_symb_count += 2
            else:
                # print(symb, symb_start, symb_end)
                data_start = data_symb_count * self.l_symbols
                data_end = data_start + self.l_symbols

                buffer[:, symb_start: symb_end] = time_ofdm_data_symbols[:, data_start: data_end]
                # for ant in range(self.num_ant):

                data_symb_count += 1

                total_symb_count += 1
        return buffer

    def zadoff_chu_gen(self, prime):
        synch_config = array([self.pattern[0], self.n_synch_bins])
        l_chu = product(synch_config)
        primes = [23, 41]

        x0 = array(range(0, int(l_chu)))
        x1 = array(range(1, int(l_chu) + 1))

        zchu = zeros((self.num_ant, l_chu), dtype=complex)
        index = 0
        for prime in primes:
            if l_chu % 2 == 0:
                zchu[index] = exp(-1j * (2 * pi / l_chu) * prime * (x0 ** 2 / 2))
            else:
                zchu[index] = exp(-1j * (2 * pi / l_chu) * prime * (x0 * x1) / 2)
            index += 1

        return zchu
