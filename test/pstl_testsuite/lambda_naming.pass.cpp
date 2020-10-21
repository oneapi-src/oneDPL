// -*- C++ -*-
//===-- lambda_naming.pass.cpp --------------------------------------------===//
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

#if _PSTL_BACKEND_SYCL
#    include <CL/sycl.hpp>
#endif
#include "support/pstl_test_config.h"
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(iterator)

using namespace TestUtils;

// This is the simple test for compilation only, to check if lambda naming works correctly
int main() {
#if __SYCL_UNNAMED_LAMBDA__ && _PSTL_BACKEND_SYCL
    const int n = 1000;
    cl::sycl::buffer<int, 1> buf{ cl::sycl::range<1>(n) };
    cl::sycl::buffer<int, 1> out_buf{ cl::sycl::range<1>(n) };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end = buf_begin + n;

    const auto policy = TestUtils::default_dpcpp_policy;
    auto buf_begin_discard_write =
        dpstd::begin(buf, cl::sycl::write_only,
#if __cplusplus >= 201703L
            cl::sycl::noinit);
#else
            cl::sycl::property::noinit{});
#endif
    ::std::fill(policy, buf_begin_discard_write, buf_begin_discard_write + n, 1);
    ::std::sort(policy, buf_begin, buf_end);
    ::std::for_each(policy, buf_begin, buf_end, [](int& x) { x += 41; });
#if !_PSTL_FPGA_DEVICE
    ::std::inplace_merge(policy, buf_begin, buf_begin + n / 2, buf_end);
    auto red_val = ::std::reduce(policy, buf_begin, buf_end, 1);
    auto buf_out_begin = oneapi::dpl::begin(out_buf);
    ::std::inclusive_scan(policy, buf_begin, buf_end, buf_out_begin);
    bool is_equal = ::std::equal(policy, buf_begin, buf_end, buf_out_begin);
    auto does_1_exist = ::std::find(policy, buf_begin, buf_end, 1);
#endif
#endif
    ::std::cout << done() << ::std::endl;
    return 0;
}
