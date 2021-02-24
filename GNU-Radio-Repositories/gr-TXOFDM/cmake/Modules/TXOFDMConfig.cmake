INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_TXOFDM TXOFDM)

FIND_PATH(
    TXOFDM_INCLUDE_DIRS
    NAMES TXOFDM/api.h
    HINTS $ENV{TXOFDM_DIR}/include
        ${PC_TXOFDM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    TXOFDM_LIBRARIES
    NAMES gnuradio-TXOFDM
    HINTS $ENV{TXOFDM_DIR}/lib
        ${PC_TXOFDM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/TXOFDMTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TXOFDM DEFAULT_MSG TXOFDM_LIBRARIES TXOFDM_INCLUDE_DIRS)
MARK_AS_ADVANCED(TXOFDM_LIBRARIES TXOFDM_INCLUDE_DIRS)
