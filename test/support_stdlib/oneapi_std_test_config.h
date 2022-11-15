// -*- C++ -*-
//===-- oneapi_std_test_config.h ------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEAPI_STD_TEST_config_H
#define _ONEAPI_STD_TEST_config_H

#include "../support/test_config.h"
#include "../support/utils.h"

#define _ONEAPI_STD_TEST_STRING(X) _ONEAPI_STD_TEST_STRING_AUX(oneapi/dpl/X)
#define _ONEAPI_STD_TEST_STRING_AUX(X) #X
//to support the optional including: <algorithm>, <iterator>, <numeric>, <array> 
#define _ONEAPI_STD_TEST_HEADER(HEADER_ID) _ONEAPI_STD_TEST_STRING(HEADER_ID)
namespace oneapi
{
namespace dpl
{
}
}
namespace oneapi_cpp_ns = oneapi::dpl;
#endif /* _ONEAPI_STD_TEST_config_H */
