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

#include <algorithm>

#include <CL/sycl.hpp>

namespace TestUtils
{
    template <sycl::usm::alloc alloc_type, typename T>
    class sycl_operations_helper
    {
    public:

        static
        T*
        alloc(const sycl::queue& q, size_t count)
        {
            check_alloc_type();

            if constexpr (alloc_type == sycl::usm::alloc::shared)
            {
                return sycl::malloc_shared<T>(count * sizeof(T), q.get_device(), q.get_context());
            }
            else
            {
                return sycl::malloc_device<T>(count * sizeof(T), q.get_device(), q.get_context());
            }
        }

        static
        void
        copy_from_host(sycl::queue& q, T* dest_ptr, const T* src_ptr, size_t count)
        {
            check_alloc_type();

            if constexpr (alloc_type == sycl::usm::alloc::shared)
            {
                ::std::copy_n(src_ptr, count, dest_ptr);
            }
            else
            {
                q.memcpy(dest_ptr, src_ptr, count * sizeof(T));
                q.wait();
            }
        }

        static
        void
        copy_to_host(sycl::queue& q, T* dest_ptr, const T* src_ptr, size_t count)
        {
            check_alloc_type();

            if constexpr (alloc_type == sycl::usm::alloc::shared)
            {
                ::std::copy_n(src_ptr, count, dest_ptr);
            }
            else
            {
                q.memcpy(dest_ptr, src_ptr, count * sizeof(T));
                q.wait();
            }
        }

    private:

        static
        void
        check_alloc_type()
        {
            static_assert(alloc_type == sycl::usm::alloc::shared || alloc_type == sycl::usm::alloc::device,
                          "Invalid state of alloc_type param in sycl_operations_helper class");
        }
    };

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
