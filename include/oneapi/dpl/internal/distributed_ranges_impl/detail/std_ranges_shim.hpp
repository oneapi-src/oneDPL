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

#ifndef _ONEDPL_DR_STD_RANGES_NAMESPACE

// If not overridden, use std::ranges
#    include <ranges>
#    define _ONEDPL_DR_STD_RANGES_NAMESPACE std::ranges

#endif

namespace stdrng = ::_ONEDPL_DR_STD_RANGES_NAMESPACE;

#endif /* _ONEDPL_DR_DETAIL_RANGES_SHIM_HPP */
