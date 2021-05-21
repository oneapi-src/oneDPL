// -*- C++ -*-
//===-- common_config.h ---------------------------------------------------===//
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

#ifndef _ONEDPL_COMMON_CONFIG
#define _ONEDPL_COMMON_CONFIG
// Workarounds for libstdc++9, libstdc++10 when new TBB version is found in the environment
#if __cplusplus >= 201703L
#    if __has_include(<tbb/version.h>)
#        ifndef PSTL_USE_PARALLEL_POLICIES
#            define PSTL_USE_PARALLEL_POLICIES (_GLIBCXX_RELEASE != 9)
#        endif
#        ifndef _GLIBCXX_USE_TBB_PAR_BACKEND
#            define _GLIBCXX_USE_TBB_PAR_BACKEND (_GLIBCXX_RELEASE > 10)
#        endif
#    endif // __has_include(<tbb/version.h>)
#endif     // __cplusplus >= 201703L

#endif
