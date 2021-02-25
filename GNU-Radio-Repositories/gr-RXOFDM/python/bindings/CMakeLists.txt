# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT RXOFDM_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT RXOFDM_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND RXOFDM_python_files
    python_bindings.cc)

GR_PYBIND_MAKE_OOT(RXOFDM 
   ../..
   gr::RXOFDM
   "${RXOFDM_python_files}")

install(TARGETS RXOFDM_python DESTINATION ${GR_PYTHON_DIR}/RXOFDM COMPONENT pythonapi)
