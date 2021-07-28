// -*- C++ -*-
//===-- execution_sycl_defs.h ---------------------------------------------===//
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

#ifndef _ONEDPL_utils_sycl_H
#define _ONEDPL_utils_sycl_H

#include <CL/sycl.hpp>

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define __LIBSYCL_VERSION                                                                                          \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define __LIBSYCL_VERSION 0
#endif

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename _T, int _Dim, typename _AllocT>
::std::size_t
__get_buffer_size(const sycl::buffer<_T, _Dim, _AllocT>& __buffer)
{
    return
#if __LIBSYCL_VERSION >= 50300
        __buffer.size();
#else
        __buffer.get_count();
#endif
}

template <typename _T, int _Dim, sycl::access::mode _AccMode, sycl::access::target _AccTarget,
          sycl::access::placeholder _Placeholder>
::std::size_t
__get_accessor_size(const sycl::accessor<_T, _Dim, _AccMode, _AccTarget, _Placeholder>& __accessor)
{
    return
#if __LIBSYCL_VERSION >= 50300
        __accessor.size();
#else
        __accessor.get_count();
#endif
}

template <typename _Item>
void
__group_barrier(_Item _item,
#if __LIBSYCL_VERSION >= 50300
    sycl::memory_scope _fence_scope = sycl::memory_scope::work_group
#else
    sycl::access::fence_space _fence_scope = sycl:access::fence_space::local_space
#endif
)
{
#if __LIBSYCL_VERSION >= 50300
    sycl::group_barrier(_item.get_group(), _fence_scope);
#else
    item.barrier(_fence_scope);
#endif
}

#define _ONEDPL_SYCL2020_COLLECTIVES_PRESENT (__LIBSYCL_VERSION >= 50300)

template <typename _Group, typename _T>
_T __group_broadcast(_Group _g, _T _val, ::std::size_t _local_id)
{
    return
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
        sycl::group_broadcast(
#else
        sycl::ONEAPI::broadcast(
#endif
        _g, _val, _local_id);
}


template <typename _Group, typename _T, typename _BinaryOp>
_T __exclusive_scan_over_group(_Group _g, _T _val, _BinaryOp _binary_op)
{
    return
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
        sycl::exclusive_scan_over_group(
#else
        sycl::ONEAPI::exclusive_scan(
#endif
            _g, _val, _binary_op);
}

template <typename _Group, typename _T, typename _BinaryOp>
_T __inclusive_scan_over_group(_Group _g, _T _val, _BinaryOp _binary_op)
{
    return
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
        sycl::inclusive_scan_over_group(
#else
        sycl::ONEAPI::inclusive_scan(
#endif
            _g, _val, _binary_op);
}

template <typename _Group, typename _T, typename _BinaryOp>
_T __reduce_over_group(_Group _g, _T _val, _BinaryOp _binary_op)
{
    return
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
        sycl::reduce_over_group(
#else
        sycl::ONEAPI::reduce(
#endif
            _g, _val, _binary_op);
}

template <typename _Group, typename _InPtr, typename _OutPtr, typename _T, typename _BinaryOp>
_OutPtr __joint_exclusive_scan(_Group _g, _InPtr _first, _InPtr _last, _OutPtr _result, _T _init, _BinaryOp _binary_op)
{
    return
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
        sycl::joint_exclusive_scan(
#else
        sycl::ONEAPI::exclusive_scan(
#endif
            _g, _first, _last, _result, _init, _binary_op);
}

} // oneapi
} // dpl
} // __par_backend_hetero

#endif /* _ONEDPL_utils_sycl_H */