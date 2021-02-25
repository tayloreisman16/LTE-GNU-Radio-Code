INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_OFDMRECEIVER OFDMReceiver)

FIND_PATH(
    OFDMRECEIVER_INCLUDE_DIRS
    NAMES OFDMReceiver/api.h
    HINTS $ENV{OFDMRECEIVER_DIR}/include
        ${PC_OFDMRECEIVER_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    OFDMRECEIVER_LIBRARIES
    NAMES gnuradio-OFDMReceiver
    HINTS $ENV{OFDMRECEIVER_DIR}/lib
        ${PC_OFDMRECEIVER_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OFDMRECEIVER DEFAULT_MSG OFDMRECEIVER_LIBRARIES OFDMRECEIVER_INCLUDE_DIRS)
MARK_AS_ADVANCED(OFDMRECEIVER_LIBRARIES OFDMRECEIVER_INCLUDE_DIRS)
