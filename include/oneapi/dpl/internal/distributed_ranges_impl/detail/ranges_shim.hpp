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

#pragma once

#ifndef DR_USE_RANGES_V3

// see: https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/icpx-std-views-zip-compile-error/m-p/1577247
// needs this workaround: https://github.com/gcc-mirror/gcc/commit/be34a8b538c0f04b11a428bd1a9340eb19dec13f
// llvm issue: https://github.com/llvm/llvm-project/issues/61763
//#include <ranges>
#    include "fixranges"

namespace rng = ::std::ranges;

#    define DR_RANGES_NAMESPACE std::ranges

#else

#    include <range/v3/all.hpp>

namespace rng = ::ranges;

#    define DR_RANGES_NAMESPACE ranges

#endif
