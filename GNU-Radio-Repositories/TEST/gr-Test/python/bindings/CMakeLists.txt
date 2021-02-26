# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT Test_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT Test_sources)

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

list(APPEND Test_python_files
    python_bindings.cc)

GR_PYBIND_MAKE_OOT(Test 
   ../..
   gr::Test
   "${Test_python_files}")

install(TARGETS Test_python DESTINATION ${GR_PYTHON_DIR}/Test COMPONENT pythonapi)
