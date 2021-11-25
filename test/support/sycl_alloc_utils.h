// -*- C++ -*-
//===-- sycl_alloc_utils.h ------------------------------------------------===//
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

#ifndef __TEST_SYCL_ALLOC_UTILS_H
#define __TEST_SYCL_ALLOC_UTILS_H

#if TEST_DPCPP_BACKEND_PRESENT

#include "oneapi/dpl/pstl/hetero/dpcpp/sycl_defs.h"

namespace TestUtils
{
    template <typename _T>
    struct __sycl_deleter
    {
        const sycl::queue q;

        void
        operator()(_T* __memory) const
        {
            sycl::free(__memory, q);
        }
    };

    namespace
    {
        template <sycl::usm::alloc __type>
        using __alloc_type = ::std::integral_constant<sycl::usm::alloc, __type>;
        using __shared_alloc_type = __alloc_type<sycl::usm::alloc::shared>;
        using __device_alloc_type = __alloc_type<sycl::usm::alloc::device>;

        template <typename _ValueType, typename _Size>
        _ValueType*
        allocate_impl(sycl::queue& __q, _Size __sz, __shared_alloc_type)
        {
            return sycl::malloc_shared<_ValueType>(__sz, __q);
        }
        template <typename _ValueType, typename _Size>
        _ValueType*
        allocate_impl(sycl::queue& __q, _Size __sz, __device_alloc_type)
        {
            return sycl::malloc_device<_ValueType>(__sz, __q);
        }

    } // namespace

    template <sycl::usm::alloc _alloc_type, typename _T>
    using unique_usm_ptr = ::std::unique_ptr<_T, __sycl_deleter<_T>>;

    template <sycl::usm::alloc _alloc_type, typename _Size>
    unique_usm_ptr usm_alloc(sycl::queue& __q, _Size __sz)
    {
        auto p = allocate_impl(__q, __count, __alloc_type<_alloc_type>{});
        assert(ptr != nullptr || __sz == 0);

        return unique_usm_ptr<T>(p, __sycl_deleter<T>{my_queue});
    }

    template <sycl::usm::alloc _alloc_type, typename _Iterator, typename _Size>
    unique_usm_ptr usm_alloc_and_copy(sycl::queue& __q, _Iterator __it, _Size __sz)
    {
        auto ptr = usm_alloc<_alloc_type>(__q, __sz);

        //TODO: support copying data provided by non-contiguous iterator
        auto __src = std::addressof(*__it);
        assert(std::addressof(*(__it + __count)) - __src == __count);

        if (__sz > 0)
        {
            __queue.copy(__src, ptr.get(), __sz);
            __queue.wait();
        }

        return ptr;
    }

    template <sycl::usm::alloc _alloc_type, typename _T, typename _Iterator>
    unique_usm_ptr usm_alloc_and_copy(sycl::queue& __q, _Iterator __itBegin, _Iterator __itEnd)
    {
        return usm_alloc_and_copy<_alloc_type>(__q, __itBegin, ::std::distance(__itBegin, __itEnd));
    }

    template <sycl::usm::alloc _alloc_type, typename _T, typename _Iterator, typename _Size>
    void retrieve_data(sycl::queue& __q, unique_usm_ptr __ptr, _Iterator __it, _Size __count)
    {
        assert((__ptr.get() != nullptr && __count > 0) || (__ptr.get() == nullptr && __count == 0));

        if (__count > 0)
        {
            //TODO: support copying data provided by non-contiguous iterator
            auto __dst = std::addressof(*__it);
            assert(std::addressof(*(__it + __count)) - __dst == __count);

            __q.copy(__ptr.get(), __dst, __count);
            __q.wait();
        }
    }

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
