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

#include <CL/sycl.hpp>

namespace TestUtils
{

    template <typename Op, ::std::size_t CallNumber>
    using unique_kernel_name = oneapi::dpl::__par_backend_hetero::__unique_kernel_name<Op, CallNumber>;

    template <typename Policy, int idx>
    using new_kernel_name = oneapi::dpl::__par_backend_hetero::__new_kernel_name<Policy, idx>;

    template <sycl::usm::alloc alloc_type, typename T>
    class Helper
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
        cpy_from_host(sycl::queue& q, T* dest_ptr, const T* src_ptr, size_t count)
        {
            check_alloc_type();

            if constexpr (alloc_type == sycl::usm::alloc::shared)
            {
                q.memcpy(dest_ptr, src_ptr, count * sizeof(T));
                q.wait();
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                    dest_ptr[i] = src_ptr[i];
            }
        }

        static
        void
        cpy_to_host(sycl::queue& q, T* dest_ptr, const T* src_ptr, size_t count)
        {
            check_alloc_type();

            if constexpr (alloc_type == sycl::usm::alloc::shared)
            {
                q.memcpy(dest_ptr, src_ptr, count * sizeof(T));
                q.wait();
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                    dest_ptr[i] = src_ptr[i];
            }
        }

    private:

        static
        void
        check_alloc_type()
        {
            static_assert(alloc_type == sycl::usm::alloc::shared || alloc_type == sycl::usm::alloc::device,
                          "Invalid state of alloc_type param in Helper class");
        }
    };

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
