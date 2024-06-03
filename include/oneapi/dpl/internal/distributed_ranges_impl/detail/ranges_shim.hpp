// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// TODO: use libstdc++ 13.0 or greater if available.

#define DR_USE_STD_RANGES

#ifdef DR_USE_STD_RANGES

// workaround of LLVM-61763
// see: https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/icpx-std-views-zip-compile-error/m-p/1577247
// needs this workaround: https://github.com/gcc-mirror/gcc/commit/be34a8b538c0f04b11a428bd1a9340eb19dec13f
// llvm issue: https://github.com/llvm/llvm-project/issues/61763
//#include <ranges>
#include "fixranges"

namespace rng = ::std::ranges;

#define DR_RANGES_NAMESPACE std::ranges

#else

#include <range/v3/all.hpp>

namespace rng = ::ranges;

#define DR_RANGES_NAMESPACE ranges

#endif
