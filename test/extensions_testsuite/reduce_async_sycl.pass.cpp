// -*- C++ -*-
//===-- reduce_async_sycl.pass.cpp --------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/async"
#include "oneapi/dpl/iterator"

#include <iostream>
#include <iomanip>

#include <CL/sycl.hpp>

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (PSTL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

// comparator implementing operator==
template <typename T>
class my_equal
{
  public:
    using first_argument_type = T;
    using second_argument_type = T;

    explicit my_equal() {}

    template <typename _Xp, typename _Yp>
    bool
    operator()(_Xp&& __x, _Yp&& __y) const
    {
        return std::forward<_Xp>(__x) == std::forward<_Yp>(__y);
    }
};

// binary functor implementing operator+
class my_plus
{
  public:
    explicit my_plus() {}

    template <typename _Xp, typename _Yp>
    bool
    operator()(_Xp&& __x, _Yp&& __y) const
    {
        return std::forward<_Xp>(__x) + std::forward<_Yp>(__y);
    }
};

int
main()
{
    const size_t N = 13;

    // ASYNC REDUCE TEST //

    {
        // create buffers
        cl::sycl::buffer<uint64_t, 1> val_buf{cl::sycl::range<1>(N)};

        {
            auto vals = val_buf.template get_access<cl::sycl::access::mode::read_write>();

            //T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 ... };

            // result = X;

            // Initialize data
            for (int i = 0; i != N - 1; ++i)
            {
                vals[i] = i % 4 + 1;
                if (i > 3)
                {
                    ++i;
                    vals[i] = vals[i - 1];
                }
            }
            vals[N - 1] = 0;
        }

        // create sycl iterators
        auto val_beg = oneapi::dpl::begin(val_buf);
        auto val_end = oneapi::dpl::end(val_buf);

        // create named policy from existing one
        auto new_policy =
            oneapi::dpl::execution::make_device_policy<class AsyncReduce>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm
        auto fut1 = oneapi::dpl::experimental::reduce_async(new_policy, val_beg, val_end, 0);

        auto res1 = fut1.get();

        {
            // check result
            ASSERT_EQUAL(res1, 26);
        }
    }
    std::cout << "done" << std::endl;
    return 0;
}
