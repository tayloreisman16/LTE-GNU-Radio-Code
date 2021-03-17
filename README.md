# LTE-GNU-Radio-Code
Our GNU Radio system uses OFDM (Orthogonal Frequency-
Division Multiplexing) to transmit high volumes of binary data. This process maps multiple bits to specific waveforms and sends that data over the air as symbols to the receiver. This technology is the driving force behind many modern communications systems, such as WiFi and LTE systems.
# TO RUN

To enable all the blocks in each of the GRC blocks, use ”source /prefix/default/setup_env.sh” to link GNU Radio and then delete the build directory in each of the GNU Radio OOT modules, such as gr-utsa-ofdm and then rerun the CMake commands after creating a new build/ directory.
The commands for CMake are as follows:
0. sudo rm -R build (if there is already a build folder.)
1. cd build/
2. cmake ../
3. make
4. sudo make install
5. sudo ldconfig
6. gnuradio-companion

After running these commands, all the blocks in that particular OOT module should populate in the GRC diagram.
