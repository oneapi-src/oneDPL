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

// RAII service class to allocate shared/device memory (USM)
// Usage model"
// 1. allocate USM memory and copying data to USM:
//    usm_data_transfer_helper<alloc_type, _ValueType> dtHelper(queue, first, count); 
// or 
//    usm_data_transfer_helper<alloc_type, _ValueType> dtHelper(queue, std::begin(data), std::end(data));
// or just allocate USM memory"
//    usm_data_transfer_helper<alloc_type, _ValueType> dtHelper(queue, count); 
// 2. get a USM pointer by usm_data_transfer_helper::get_data() and passed one into a parallel algorithm with dpc++ policy.
// 3. Retrieve data back (in case of device allocation type) to the host for further checking result.
//    alloc.retrieve_data(dest_host);
template<sycl::usm::alloc _alloc_type, typename _ValueType>
class usm_data_transfer_helper
{
    static_assert(_alloc_type == sycl::usm::alloc::shared || _alloc_type == sycl::usm::alloc::device,
                      "Invalid allocation type for usm_data_transfer_helper class");

    using __difference_type = typename ::std::iterator_traits<_ValueType*>::difference_type;

    template<sycl::usm::alloc __type>
    using __alloc_type = ::std::integral_constant<sycl::usm::alloc, __type>;
    using __shared_alloc_type = __alloc_type<sycl::usm::alloc::shared>;
    using __device_alloc_type = __alloc_type<sycl::usm::alloc::device>;

    template<typename _Size>
    _ValueType* allocate( _Size __sz, __shared_alloc_type)
    {
        return sycl::malloc_shared<_ValueType>(__sz, __queue);
    }
    template<typename _Size>
    _ValueType* allocate( _Size __sz, __device_alloc_type)
    {
        return sycl::malloc_device<_ValueType>(__sz, __queue);
    }

  public:

    template<typename _Size>
    usm_data_transfer_helper(sycl::queue& __q, _Size __sz)
        : __queue(__q), __count(__sz)
    {
        __ptr = allocate(__count, __alloc_type<_alloc_type>{});
        assert(__ptr != nullptr || __sz == 0);
    }

    template<typename _Iterator, typename _Size>
    usm_data_transfer_helper(sycl::queue& __q, _Iterator __it, _Size __sz)
        : usm_data_transfer_helper(__q, __sz)
    {
        //TODO: support copying data provided by non-contiguous iterator
        auto __src = std::addressof(*__it);
        assert(std::addressof(*(__it + __count)) - __src == __count);

        if (__count > 0)
        {
            __queue.copy(__src, __ptr, __count);
            __queue.wait();
        }
    }

    template<typename _Iterator>
    usm_data_transfer_helper(sycl::queue& __q, _Iterator __itBegin, _Iterator __itEnd)
        : usm_data_transfer_helper(__q, __itBegin, ::std::distance(__itBegin, __itEnd))
    {
    }

    ~usm_data_transfer_helper()
    {
        assert((__ptr != nullptr && __count > 0) || (__ptr == nullptr && __count == 0));

        sycl::free(__ptr, __queue);
    }

    _ValueType* get_data() const
    {
        return __ptr;
    }

    template<typename _Iterator>
    void retrieve_data(_Iterator __it)
    {
        assert((__ptr != nullptr && __count > 0) || (__ptr == nullptr && __count == 0));

        if (__count > 0)
        {
            //TODO: support copying data provided by non-contiguous iterator
            auto __dst = std::addressof(*__it);
            assert(std::addressof(*(__it + __count)) - __dst == __count);

            __queue.copy(__ptr, __dst, __count);
            __queue.wait();
        }
    }

private:
    __difference_type __count = 0;
    _ValueType* __ptr = nullptr;
    sycl::queue& __queue;
};

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
