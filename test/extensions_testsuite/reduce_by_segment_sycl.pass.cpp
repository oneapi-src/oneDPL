// -*- C++ -*-
//===-- reduce_by_segment_sycl.pass.cpp --------------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#include <iostream>
#include <iomanip>

#include <CL/sycl.hpp>

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

template<typename _T1, typename _T2>
void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS (PSTL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

// comparator implementing operator==
template<typename T>
class my_equal
{
public:
    using first_argument_type = T;
    using second_argument_type = T;

    explicit my_equal() {}

    template <typename _Xp, typename _Yp>
    bool operator()(_Xp&& __x, _Yp&& __y) const {
        return std::forward<_Xp>(__x) == std::forward<_Yp>(__y);
    }
};

// binary functor implementing operator+
class my_plus
{
public:
    explicit my_plus() {}

    template <typename _Xp, typename _Yp>
    bool operator()(_Xp&& __x, _Yp&& __y) const {
        return std::forward<_Xp>(__x) + std::forward<_Yp>(__y);
    }
};

int main() {

    // #6 REDUCE BY SEGMENT TEST //

    {
        // create buffers
        cl::sycl::buffer<uint64_t, 1> key_buf{ cl::sycl::range<1>(13) };
        cl::sycl::buffer<uint64_t, 1> val_buf{ cl::sycl::range<1>(13) };
        cl::sycl::buffer<uint64_t, 1> key_res_buf{ cl::sycl::range<1>(13) };
        cl::sycl::buffer<uint64_t, 1> val_res_buf{ cl::sycl::range<1>(13) };

        {
            auto keys    = key_buf.template get_access<cl::sycl::access::mode::read_write>();
            auto vals    = val_buf.template get_access<cl::sycl::access::mode::read_write>();
            auto keys_res    = key_res_buf.template get_access<cl::sycl::access::mode::read_write>();
            auto vals_res    = val_res_buf.template get_access<cl::sycl::access::mode::read_write>();

            //T keys[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };
            //T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };

            // keys_result = {1, 2, 3, 4, 1, 3, 1, 3, 0};
            // vals_result = {1, 2, 3, 4, 2, 6, 2, 6, 0};

            // Initialize data
            for (int i = 0; i != 12; ++i) {
                keys[i] = i % 4 + 1;
                vals[i] = i % 4 + 1;
                keys_res[i] = 9;
                vals_res[i] = 1;
                if (i > 3) {
                    ++i;
                    keys[i] = keys[i-1];
                    vals[i] = vals[i-1];
                    keys_res[i] = 9;
                    vals_res[i] = 1;
                }
            }
            keys[12] = 0;
            vals[12] = 0;
        }

        // create sycl iterators
        auto key_beg = oneapi::dpl::begin(key_buf);
        auto key_end = oneapi::dpl::end(key_buf);
        auto val_beg = oneapi::dpl::begin(val_buf);
        auto key_res_beg = oneapi::dpl::begin(key_res_buf);
        auto val_res_beg = oneapi::dpl::begin(val_res_buf);

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<class ReduceBySegment>(
            oneapi::dpl::execution::dpcpp_default);

        // call algorithm
        auto res1 = oneapi::dpl::reduce_by_segment(new_policy, key_beg, key_end, val_beg, key_res_beg, val_res_beg);

        {
            // check values
            auto keys_res    = key_res_buf.template get_access<cl::sycl::access::mode::read_write>();
            auto vals_res    = val_res_buf.template get_access<cl::sycl::access::mode::read_write>();
            int n = std::distance(key_res_beg, res1.first);
            for (auto i = 0; i != n; ++i) {
                if (i < 4) {
                    ASSERT_EQUAL(keys_res[i], i+1);
                    ASSERT_EQUAL(vals_res[i], i+1);
                } else if (i == 4 || i == 6) {
                    ASSERT_EQUAL(keys_res[i], 1);
                    ASSERT_EQUAL(vals_res[i], 2);
                } else if (i == 5 || i == 7) {
                    ASSERT_EQUAL(keys_res[i], 3);
                    ASSERT_EQUAL(vals_res[i], 6);
                } else if (i == 8) {
                    ASSERT_EQUAL(keys_res[i], 0);
                    ASSERT_EQUAL(vals_res[i], 0);
                } else {
                    std::cout << "fail: unexpected values in output range\n";
                }
            }

            // reset value_result for test using discard_iterator
            for (auto i = 0; i != n; ++i) {
                keys_res[i] = 0;
                vals_res[i] = 0;
            }
        }
    }
    std::cout << "done" << std::endl;
    return 0;
}
