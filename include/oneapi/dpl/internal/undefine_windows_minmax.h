// -*- C++ -*-
//===-- undefine_windows_minmax.h -----------------------------------------===//
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

#ifndef _ONEDPL_UNDEFINE_WINDOWS_MINMAX_H
#define _ONEDPL_UNDEFINE_WINDOWS_MINMAX_H

// Windows.h defines min and max macros, which can conflict with std::min and std::max and other APIs.
// Here we undefine them to protect from conflicts. In oneapi/dpl/internal/redefine_windows_minmax.h we
// redefine them if they are undefined here.
// This header must be included first by public headers, or after windows.h inclusion.

#if (_MSC_VER)
#if defined(min)
#define _ONEDPL_UNDEFINED_MIN 1
#undef min
#endif // defined(min)

#if defined(max)
#define _ONEDPL_UNDEFINED_MAX 1
#undef max
#endif // defined(max)

#endif

#endif // _ONEDPL_UNDEFINE_WINDOWS_MINMAX_H
