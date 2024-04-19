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

#ifndef _UTILS_CONST_H
#define _UTILS_CONST_H

namespace TestUtils
{
#define _SKIP_RETURN_CODE 77

#define __TEST_MAX_SIZE 30000

// All these offset consts used for indirect testing of calculation an offset parameter
// (as a result dpl::begin(buf) + offset) for further passing within sycl::accessor constructor.
constexpr ::std::size_t inout1_offset = 3;
constexpr ::std::size_t inout2_offset = 5;
constexpr ::std::size_t inout3_offset = 7;
constexpr ::std::size_t inout4_offset = 9;

} /* namespace TestUtils */

#endif // _UTILS_CONST_H
