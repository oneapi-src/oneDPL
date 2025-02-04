// -*- C++ -*-
//===-- redefine_windows_minmax.h -----------------------------------------===//
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

#ifndef _ONEDPL_REDEFINE_WINDOWS_MINMAX_H
#define _ONEDPL_REDEFINE_WINDOWS_MINMAX_H

// Windows.h defines min and max macros, which can conflict with std::min and std::max and other APIs
// in oneapi/dpl/internal/undefine_windows_minmax.h we undefine these macros to protect from conflicts.
// Here we redefine these macros if we previously undefined them.
// This header must be included last by public headers.

#if (_MSC_VER)
#if defined(_ONEDPL_UNDEFINED_MIN)
#undef _ONEDPL_UNDEFINED_MIN
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif // defined(_ONEDPL_UNDEFINED_MIN)

#if defined(max)
#undef _ONEDPL_UNDEFINED_MAX
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif // defined(_ONEDPL_UNDEFINED_MAX)
#endif

#endif // _ONEDPL_REDEFINE_WINDOWS_MINMAX_H
