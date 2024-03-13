// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// TODO: use libstdc++ 13.0 or greater if available.

// #define DR_USE_STD_RANGES

#ifdef DR_USE_STD_RANGES

#include <ranges>

namespace rng = ::std::ranges;

#define DR_RANGES_NAMESPACE std::ranges

#else

#include <range/v3/all.hpp>

namespace rng = ::ranges;

#define DR_RANGES_NAMESPACE ranges

#endif
