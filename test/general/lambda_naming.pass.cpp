// -*- C++ -*-
//===-- lambda_naming.pass.cpp --------------------------------------------===//
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

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"

#if _ONEDPL_BACKEND_SYCL
#    include <CL/sycl.hpp>
#endif

using namespace TestUtils;

// This is the simple test for compilation only, to check if lambda naming works correctly
int main() {
    const int n = 1000;
    sycl::buffer<int, 1> buf{ sycl::range<1>(n) };
    sycl::buffer<int, 1> out_buf{ sycl::range<1>(n) };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end = buf_begin + n;

    const auto policy = TestUtils::default_dpcpp_policy;
    auto buf_begin_discard_write =
        oneapi::dpl::begin(buf, sycl::write_only,
#if __cplusplus >= 201703L
            sycl::noinit);
#else
            sycl::property::noinit{});
#endif // __cplusplus >= 201703L

    ::std::fill(policy, buf_begin_discard_write, buf_begin_discard_write + n, 1);
#if __SYCL_UNNAMED_LAMBDA__ && _ONEDPL_BACKEND_SYCL
    ::std::sort(policy, buf_begin, buf_end);
    ::std::for_each(policy, buf_begin, buf_end, [](int& x) { x += 41; });

#if !_ONEDPL_FPGA_DEVICE
    ::std::inplace_merge(policy, buf_begin, buf_begin + n / 2, buf_end);
    auto red_val = ::std::reduce(policy, buf_begin, buf_end, 1);
    EXPECT_TRUE(red_val == 42001, "wrong return value from reduce");
    auto buf_out_begin = oneapi::dpl::begin(out_buf);
    ::std::inclusive_scan(policy, buf_begin, buf_end, buf_out_begin);
    bool is_equal = ::std::equal(policy, buf_begin, buf_end, buf_out_begin);
    EXPECT_TRUE(!is_equal, "wrong return value from equal");
    auto does_1_exist = ::std::find(policy, buf_begin, buf_end, 1);
    EXPECT_TRUE(does_1_exist - buf_begin == 1000, "wrong return value from find");
#endif // !_ONEDPL_FPGA_DEVICE

#elif !__SYCL_UNNAMED_LAMBDA__ && _ONEDPL_BACKEND_SYCL
    // ::std::for_each(policy, buf_begin, buf_end, [](int& x) { x++; }); // It's not allowed. Policy with different name is needed
    ::std::for_each(oneapi::dpl::execution::make_device_policy<class ForEach>(policy), buf_begin, buf_end, [](int& x) { x++; });
    auto red_val = ::std::reduce(policy, buf_begin, buf_end, 1);
    EXPECT_TRUE(red_val == 1002, "wrong return value from reduce");
#endif
    ::std::cout << done() << ::std::endl;
    return 0;
}
