from numpy import zeros, array, tile, reshape, size, newaxis, delete, real,  \
    sum, prod, divide, sqrt, exp, log10, angle,    \
    floor, argwhere,   \
    conj, matmul, diag, outer, dot,  \
    complex64, int16,   \
    pi
from numpy.fft import fft, ifft
from numpy.random import randint
from numpy.linalg import svd
import pickle as pckl


class PhyLayerSecReceiver:
    def __init__(self, bandwidth, bin_spacing, num_ant, NFFT, CP, num_used_bins):
        gr.sync_block.__init__(self,
                               name="SynchAndChanEst",
                               in_sig=[complex64],
                               out_sig=[complex64])
        self.bandwidth = bandwidth
        self.bin_spacing = bin_spacing
        self.num_ant = num_ant
        self.bit_codebook = self.codebook_gen()  # TODO: Add function to calculate codebook!

        self.NFFT = NFFT
        self.CP = CP
        self.OFDMsymb_len = self.NFFT + self.CP
        self.num_used_bins = num_used_bins

        self.used_data_bins = self.NFFT - 2
        self.subband_size = self.num_ant

        self.num_subbands = int(floor(self.num_used_bins/self.subband_size))
        self.num_PMI = self.num_subbands

        self.key_len = self.num_subbands * self.bit_codebook

        self.message_port_register_out(pmt.intern("RXState"))

    def work(self, input_items, output_items):
        in0 = input_items[0]
        in1 = input_items[1]

        rx_state = state
        rx_in = in0

        [index0, index1] = self.mimo_synchronization(rx_in) # Dual Input
        rx_signal = self.get_signal(rx_in, index0, index1) # Dual Input/Dual Output
        tx_sig = []
        if rx_node == 'Bob' and len(args) == 4:
            # 1. At Bob #TODO: State 2 (G1r calculated)
            # send FB and Channel Information over message path
            print('Entering State 2')
            rx_sigB0 = rx_signal
            UB0 = self.sv_decomp(rx_sigB0)[0]
            bits_subbandB = self.secret_key_gen()
            FB = self.precoder_select(bits_subbandB, self.bit_codebook)
            for sb in range(0, self.num_subbands):
                tx_sig = dot(conj(UB0[sb]), conj(FB[sb].T))
            out[:] = tx_sig
            self.message_port_pub(pmt.intern("RXState"), 'S2Fin')

        if rx_node == 'Bob' and len(args) == 5:
            # 3. At Bob #TODO: State 6 (Private Data recovered)
            print('Entering State 6')
            VB1 = self.sv_decomp(rx_sigB1)[2]
            bits_sb_estimateA = self.PMI_estimate(VB1, codebook)[1]

            #TODO: Add way to recover data:: Last State HERE!
            self.message_port_pub(pmt.intern("RXState"), 'S6Fin')

        elif rx_node == 'Alice' and len(args) == 5:
            # 2. At Alice #TODO: State 4 (G2r calculated)
            print('Entering State 4')
            rx_sigA = rx_signal
            UA, _, VA = self.sv_decomp(rx_sigA)
            bits_sb_estimateB = self.PMI_estimate(VA, self.bit_codebook)[1]

            #Load Secret Data Here
            bits_subbandA = pckl.load(open("data.pckl", 'rb'))
            FA = self.precoder_select(bits_subbandA, self.bit_codebook)
            for sb in range(0, self.num_subbands):
                tx_sig = dot(conj(UA[sb]), conj(FA[sb].T))
            out[:] = FA
            self.message_port_pub(pmt.intern("RXState"), 'S4Fin')

        #TODO: Need data out!

        return len(output_items[0])

    def secret_key_gen(self):
        """
        Generate private info bits in each sub-band
        :return bits_subband: private info bits in each sub-band
        """
        bits_subband = zeros(self.num_subbands, dtype=object)

        secret_key = randint(0, 2, self.key_len)

        # Map secret key to subbands
        for sb in range(self.num_subbands):
            start = sb * self.bit_codebook
            fin = start + self.bit_codebook

            bits_subband[sb] = secret_key[start: fin]

        return bits_subband

    def sv_decomp(self, rx_sig):
        """
        Perform SVD for the matrix in each sub-band
        :param rx_sig: Channel matrix at the receiver in each sub-band
        :return lsv, sval, rsv: Left, Right Singular Vectors and Singular Values for the matrix in each sub-band
        """
        lsv = zeros(self.num_subbands, dtype=object)
        sval = zeros(self.num_subbands, dtype=object)
        rsv = zeros(self.num_subbands, dtype=object)

        for sb in range(0, self.num_subbands):
            U, S, VH = svd(rx_sig[sb])
            V = conj(VH).T
            ph_shift_u = diag(exp(-1j * angle(U[0, :])))
            ph_shift_v = diag(exp(-1j * angle(V[0, :])))
            lsv[sb] = dot(U, ph_shift_u)
            sval[sb] = S
            rsv[sb] = dot(V, ph_shift_v)

        return lsv, sval, rsv

    def dec2binary(x, num_bits):
        """
        Covert decimal number to binary array of ints (1s and 0s)
        :param x: input decimal number
        :param num_bits: Number bits required in the binary format
        :return bits: binary array of ints (1s and 0s)
        """
        bit_str = [char for char in format(x[0, 0], '0' + str(num_bits) + 'b')]
        bits = array([int(char) for char in bit_str])
        # print(x[0, 0], bits)
        return bits

    def PMI_estimate(self, rx_precoder, codebook):
        """
        Apply minumum distance to estimate the transmitted precoder, its index in the codebook and the binary equivalent
        of the index
        :param rx_precoder: observed precoder (RSV of SVD)
        :param codebook: DFT codebook of matrix precoders
        :return PMI_sb_estimate, bits_sb_estimate: Preocder matrix index and bits for each sub-band
        """
        PMI_sb_estimate = zeros(self.num_subbands, dtype=int)
        bits_sb_estimate = zeros(self.num_subbands, dtype=object)

        for sb in range(self.num_subbands):
            dist = zeros(len(codebook), dtype=float)

            for prec in range(len(codebook)):
                diff = rx_precoder[sb] - codebook[prec]
                diff_squared = real(diff*conj(diff))
                dist[prec] = sqrt(diff_squared.sum())
            min_dist = min(dist)
            PMI_estimate = argwhere(dist == min_dist)
            PMI_sb_estimate[sb] = PMI_estimate
            bits_sb_estimate[sb] = self.dec2binary(PMI_estimate, self.bit_codebook)

        return PMI_sb_estimate, bits_sb_estimate

    def codebook_gen(self):
        """
        Generate DFT codebbok of matrix preocders
        :return: matrix of matrix preocders
        """
        num_precoders = 2**self.bit_codebook
        codebook = zeros(num_precoders, dtype=object)

        for p in range(0, num_precoders):
            precoder = zeros((self.num_ant, self.num_ant), dtype=complex)
            for m in range(0, self.num_ant):
                for n in range(0, self.num_ant):
                    w = exp(1j*2*pi*(n/self.num_ant)*(m + p/num_precoders))
                    precoder[n, m] = (1/sqrt(self.num_ant))*w

            codebook[p] = precoder

        return codebook

    def precoder_select(self, bits_subband, codebook):
        """
        selects the DFT precoder from the DFT codebook based. Bits are converted to decimal and used as look up index.
        :param bits_subband: Bits in each sub-band
        :param codebook: DFT codebook of matrix precoders
        :return precoder: Selected DFT preocder from codebook for each sub-band
        """
        precoder = zeros(self.num_subbands, dtype=object)

        for sb in range(self.num_subbands):
            bits = bits_subband[sb]
            start = self.bit_codebook - 1
            bi2dec_wts = 2**(array(range(start, -1, -1)))
            codebook_index = sum(bits*bi2dec_wts)
            precoder[sb] = codebook[codebook_index]

        return precoder

