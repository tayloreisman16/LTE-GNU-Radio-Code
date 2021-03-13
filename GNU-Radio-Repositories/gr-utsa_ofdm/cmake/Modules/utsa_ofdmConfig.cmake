INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_UTSA_OFDM utsa_ofdm)

FIND_PATH(
    UTSA_OFDM_INCLUDE_DIRS
    NAMES utsa_ofdm/api.h
    HINTS $ENV{UTSA_OFDM_DIR}/include
        ${PC_UTSA_OFDM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    UTSA_OFDM_LIBRARIES
    NAMES gnuradio-utsa_ofdm
    HINTS $ENV{UTSA_OFDM_DIR}/lib
        ${PC_UTSA_OFDM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/utsa_ofdmTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(UTSA_OFDM DEFAULT_MSG UTSA_OFDM_LIBRARIES UTSA_OFDM_INCLUDE_DIRS)
MARK_AS_ADVANCED(UTSA_OFDM_LIBRARIES UTSA_OFDM_INCLUDE_DIRS)
