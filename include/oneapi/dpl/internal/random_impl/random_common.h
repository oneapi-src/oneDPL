// -*- C++ -*-
//===-- random_common.h ---------------------------------------------------===//
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
//
// Abstract:
//
// Public header file provides common utils for random implementation

#ifndef _ONEDPL_RANDOM_COMMON_H
#define _ONEDPL_RANDOM_COMMON_H

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename _T>
struct type_traits_t
{
    using element_type = _T;
    static constexpr int num_elems = 0;
};

template <typename _T, int _N>
struct type_traits_t<sycl::vec<_T, _N>>
{
    using element_type = _T;
    static constexpr int num_elems = _N;
};

template <typename _T>
using element_type_t = typename type_traits_t<_T>::element_type;

typedef union {
    uint32_t hex[2];
} dp_union_t;

typedef union {
    uint32_t hex[1];
} sp_union_t;

} // namespace internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_RANDOM_COMMON_H
