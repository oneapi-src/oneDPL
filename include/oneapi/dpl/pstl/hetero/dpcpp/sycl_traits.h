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

// This file contains some specialization SYCL traits for some oneDPL types
//
// Include this header before a kernel submit SYCL code

#ifndef _ONEDPL_SYCL_TRAITS_H
#define _ONEDPL_SYCL_TRAITS_H

#define _ONEDPL_DEVICE_COPYABLE(TYPE) \
template<typename... Ts > \
struct sycl::is_device_copyable<TYPE<Ts...>>: ::std::conjunction<sycl::is_device_copyable<Ts>...> {};

using namespace oneapi::dpl::__internal;

_ONEDPL_DEVICE_COPYABLE(__not_pred)
_ONEDPL_DEVICE_COPYABLE(__reorder_pred)
_ONEDPL_DEVICE_COPYABLE(__equal_value_by_pred)
_ONEDPL_DEVICE_COPYABLE(__equal_value)
_ONEDPL_DEVICE_COPYABLE(__not_equal_value)
_ONEDPL_DEVICE_COPYABLE(__transform_functor)
_ONEDPL_DEVICE_COPYABLE(__transform_if_unary_functor)
_ONEDPL_DEVICE_COPYABLE(__transform_if_binary_functor)
_ONEDPL_DEVICE_COPYABLE(__replace_functor)
_ONEDPL_DEVICE_COPYABLE(__replace_copy_functor)
_ONEDPL_DEVICE_COPYABLE(zip_forward_iterator)

using namespace oneapi::dpl;

_ONEDPL_DEVICE_COPYABLE(zip_iterator)
_ONEDPL_DEVICE_COPYABLE(transform_iterator)
_ONEDPL_DEVICE_COPYABLE(permutation_iterator)

#undef _ONEDPL_DEVICE_COPYABLE

#endif // _ONEDPL_SYCL_TRAITS_H
