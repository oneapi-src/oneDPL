// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

#ifndef _DPSTD_CONFIG_H
#define _DPSTD_CONFIG_H

// for deprecation
#if (__cplusplus >= 201402L)
#    define _DPSTD_DEPRECATED [[deprecated]]
#    define _DPSTD_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#elif _MSC_VER
#    define _DPSTD_DEPRECATED __declspec(deprecated)
#    define _DPSTD_DEPRECATED_MSG(msg) __declspec(deprecated(msg))
#elif (__GNUC__ || __clang__)
#    define _DPSTD_DEPRECATED __attribute__((deprecated))
#    define _DPSTD_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
#else
#    define _DPSTD_DEPRECATED
#    define _DPSTD_DEPRECATED_MSG(msg)
#endif

#define _ITERATORS_DEPRECATED _DPSTD_DEPRECATED
#define _POLICY_DEPRECATED _DPSTD_DEPRECATED

#endif /* _DPSTD_config_H */
