// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
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

#ifndef _ONEAPI_STD_TEST_CONFIG_H
#define _ONEAPI_STD_TEST_CONFIG_H

#include "../support/test_config.h"
#include "../support/utils.h"

#ifndef _USE_ONEAPI_STD
#    define _USE_ONEAPI_STD 1
#endif

#define _ONEAPI_STD_TEST_STRING_AUX(X) #X

#if _USE_ONEAPI_STD
#   define _ONEAPI_STD_TEST_STRING(X) _ONEAPI_STD_TEST_STRING_AUX(oneapi/dpl/X)
#   define _ONEAPI_TEST_NAMESPACE oneapi::dpl
#else
#   define _ONEAPI_STD_TEST_STRING(X) _ONEAPI_STD_TEST_STRING_AUX(X)
#   define _ONEAPI_TEST_NAMESPACE std
#endif  // _USE_ONEAPI_STD

//to support the optional including: <algorithm>, <iterator>, <numeric>, <array> 
#define _ONEAPI_STD_TEST_HEADER(HEADER_ID) _ONEAPI_STD_TEST_STRING(HEADER_ID)
namespace oneapi
{
namespace dpl
{
}
}
namespace oneapi_cpp_ns = oneapi::dpl;
#endif // _ONEAPI_STD_TEST_CONFIG_H
