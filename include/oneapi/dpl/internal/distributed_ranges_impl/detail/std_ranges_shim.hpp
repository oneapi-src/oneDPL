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

#ifndef _ONEDPL_DR_DETAIL_RANGES_SHIM_HPP
#define _ONEDPL_DR_DETAIL_RANGES_SHIM_HPP

#if 1 // ifndef DR_USE_RANGES_V3

#    include <ranges>

namespace stdrng = ::std::ranges;

#    define __ONEDPL_DR_STD_RANGES_NAMESPACE std::ranges

#else

#    include <range/v3/all.hpp>

namespace stdrng = ::ranges;

#    define __ONEDPL_DR_STD_RANGES_NAMESPACE ranges

#endif /* DR_USE_RANGES_V3 */

#endif /* _ONEDPL_DR_DETAIL_RANGES_SHIM_HPP */
