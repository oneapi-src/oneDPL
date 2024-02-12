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

#if __INTEL_LLVM_COMPILER && (__INTEL_LLVM_COMPILER < 20240100)

#define _ONEDPL_DEVICE_COPYABLE(TYPE) \
template<typename... Ts > \
struct sycl::is_device_copyable<TYPE<Ts...>, ::std::enable_if_t<!std::is_trivially_copyable_v<TYPE<Ts...>>>>: ::std::conjunction<sycl::is_device_copyable<Ts>...> {};

#else

#define _ONEDPL_DEVICE_COPYABLE(TYPE) \
template<typename... Ts > \
struct sycl::is_device_copyable<TYPE<Ts...>>: ::std::conjunction<sycl::is_device_copyable<Ts>...> {};

#endif

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

_ONEDPL_DEVICE_COPYABLE(fill_functor)
_ONEDPL_DEVICE_COPYABLE(generate_functor)
_ONEDPL_DEVICE_COPYABLE(__brick_fill)
_ONEDPL_DEVICE_COPYABLE(__brick_fill_n)
_ONEDPL_DEVICE_COPYABLE(__search_n_unary_predicate)
_ONEDPL_DEVICE_COPYABLE(__is_heap_check)

_ONEDPL_DEVICE_COPYABLE(equal_predicate)
_ONEDPL_DEVICE_COPYABLE(adjacent_find_fn)
_ONEDPL_DEVICE_COPYABLE(__create_mask_unique_copy)

_ONEDPL_DEVICE_COPYABLE(__op_uninitialized_fill)

using namespace oneapi::dpl::__par_backend_hetero;

_ONEDPL_DEVICE_COPYABLE(__early_exit_find_or);

using namespace oneapi::dpl::unseq_backend;

_ONEDPL_DEVICE_COPYABLE(walk_n)
_ONEDPL_DEVICE_COPYABLE(walk_adjacent_difference)
_ONEDPL_DEVICE_COPYABLE(transform_reduce)
_ONEDPL_DEVICE_COPYABLE(reduce_over_group)
_ONEDPL_DEVICE_COPYABLE(single_match_pred_by_idx)
_ONEDPL_DEVICE_COPYABLE(multiple_match_pred)
_ONEDPL_DEVICE_COPYABLE(n_elem_match_pred)
_ONEDPL_DEVICE_COPYABLE(first_match_pred)
_ONEDPL_DEVICE_COPYABLE(__create_mask)
_ONEDPL_DEVICE_COPYABLE(__copy_by_mask)
_ONEDPL_DEVICE_COPYABLE(__partition_by_mask)
_ONEDPL_DEVICE_COPYABLE(__global_scan_functor)
_ONEDPL_DEVICE_COPYABLE(__scan)
_ONEDPL_DEVICE_COPYABLE(__brick_includes)
_ONEDPL_DEVICE_COPYABLE(__brick_set_op)
_ONEDPL_DEVICE_COPYABLE(__brick_reduce_idx)

using namespace oneapi::dpl::internal;

_ONEDPL_DEVICE_COPYABLE(custom_brick)
_ONEDPL_DEVICE_COPYABLE(replace_if_fun)
_ONEDPL_DEVICE_COPYABLE(scan_by_key_fun)
_ONEDPL_DEVICE_COPYABLE(segmented_scan_fun)
_ONEDPL_DEVICE_COPYABLE(scatter_and_accumulate_fun)
_ONEDPL_DEVICE_COPYABLE(transform_if_stencil_fun)

using namespace oneapi::dpl;

_ONEDPL_DEVICE_COPYABLE(zip_iterator)
_ONEDPL_DEVICE_COPYABLE(transform_iterator)
_ONEDPL_DEVICE_COPYABLE(permutation_iterator)

#undef _ONEDPL_DEVICE_COPYABLE

#endif // _ONEDPL_SYCL_TRAITS_H
