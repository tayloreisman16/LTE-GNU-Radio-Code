INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_RXOFDM RXOFDM)

FIND_PATH(
    RXOFDM_INCLUDE_DIRS
    NAMES RXOFDM/api.h
    HINTS $ENV{RXOFDM_DIR}/include
        ${PC_RXOFDM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    RXOFDM_LIBRARIES
    NAMES gnuradio-RXOFDM
    HINTS $ENV{RXOFDM_DIR}/lib
        ${PC_RXOFDM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/RXOFDMTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(RXOFDM DEFAULT_MSG RXOFDM_LIBRARIES RXOFDM_INCLUDE_DIRS)
MARK_AS_ADVANCED(RXOFDM_LIBRARIES RXOFDM_INCLUDE_DIRS)
