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

#ifdef TEST_DPCPP_BACKEND_PRESENT

#include "oneapi/dpl/pstl/hetero/dpcpp/sycl_defs.h"

namespace TestUtils
{

// RAII service class to allocate shared/device memory (USM)
// Usage model"
// 1. allocate USM memory and copying data to USM:
//    sycl_usm_alloc<alloc_type, _ValueType> alloc(queue, first, count); 
// or just allocate USM memory"
//    sycl_usm_alloc<alloc_type, _ValueType> alloc(queue, count); 
// 2. get a USM pointer by sycl_usm_alloc::get_data() and passed one into a parallel algorithm with dpc++ policy.
// 3. Retrive data back (in case of device allocation type) to the host for further checking result.
//    alloc.retrive_data(dest_host);
template<sycl::usm::alloc _alloc_type, typename _ValueType>
class  sycl_usm_alloc
{
    static_assert(_alloc_type == sycl::usm::alloc::shared || _alloc_type == sycl::usm::alloc::device,
                      "Invalid allocation type for sycl_usm_alloc class");

    using _DifferenceType = typename ::std::iterator_traits<_ValueType*>::difference_type;

    template<sycl::usm::alloc __type>
    using _AllocType = ::std::integral_constant<sycl::usm::alloc, __type>;
    using _SharedType = _AllocType<sycl::usm::alloc::shared>;
    using _DeviceType = _AllocType<sycl::usm::alloc::device>;

    template<typename _Size>
    _ValueType* allocate( _Size __sz, _SharedType)
    {
        return sycl::malloc_shared<_ValueType>(__sz, __queue);
    }
    template<typename _Size>
    _ValueType* allocate( _Size __sz, _DeviceType)
    {
        return sycl::malloc_device<_ValueType>(__sz, __queue);
    }

  public:
    template<typename _Size>
    sycl_usm_alloc(sycl::queue& __q, _Size __sz): __queue(__q), __count(__sz)
    {
        __ptr = allocate(__count, _AllocType<_alloc_type>{});
        assert(__ptr);
    }
    template<typename _Iterator, typename _Size>
    sycl_usm_alloc(sycl::queue& __q, _Iterator __it, _Size __sz): __queue(__q), __count(__sz)
    {
        __ptr = allocate(__count, _AllocType<_alloc_type>{});
        assert(__ptr);

        //TODO: support copying data provided by non-contiguous iterator
        auto __src = std::addressof(*__it);
        assert(std::addressof(*(__it + __count)) - __src == __count);

        __queue.copy(__ptr, __src, __count);
        __queue.wait();
    }
    ~sycl_usm_alloc()
    {
        assert(__ptr);
        assert(__count > 0);

        sycl::free(__ptr, __queue);
    }

    _ValueType* get_data() const
    {
        return __ptr;
    }
    template<typename _Iterator>
    void retrive_data(_Iterator __it)
    {
        assert(__ptr);
        assert(__count > 0);

        //TODO: support copying data provided by non-contiguous iterator
        auto __dst = std::addressof(*__it);
        assert(std::addressof(*(__it + __count)) - __dst == __count);

        __queue.copy(__dst, __ptr, __count);
        __queue.wait();
    }

private:
    _DifferenceType __count;
    _ValueType* __ptr;
    sycl::queue& __queue;
};

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
