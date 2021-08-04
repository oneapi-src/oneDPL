// -*- C++ -*-
//===-- binary_search_utils.h --------------------------------------------===//
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

using namespace TestUtils;

#if !TEST_DPCPP_BACKEND_PRESENT
// Test buffers
const int max_n = 100000;
const int inout1_offset = 3;
const int inout2_offset = 5;
const int inout3_offset = 7;
const int inout4_offset = 9;
#endif

template <typename Accessor1, typename Accessor2, typename Accessor3, typename Size>
void
initialize_data(Accessor1 data, Accessor2 value, Accessor3 result, Size n)
{
    int num_values = n * .01 > 1 ? n * .01 : 1; // # search values expected to be << n
    for (int i = 0; i < n; i += 2)
    {
        data[i] = i;
        if (i + 1 < n)
        {
            data[i+1] = i;
        }
        if (i < num_values * 2)
        {
            // value = {0, 2, 5, 6, 9, 10, 13...}
            // result will alternate true/false after initial true
            value[i/2] = i + (i != 0 && i % 4 == 0 ? 1 : 0);
        }
        result[i/2] = 0;
    }
}

template <typename T, typename TestName>
void
test_on_host()
{
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        // create buffers
        std::vector<T>   inout1(max_n + inout1_offset);
        std::vector<T> inout2(max_n + inout2_offset);
        std::vector<T>   inout3(max_n + inout3_offset);

        // create iterators
        auto inout1_offset_first = std::begin(inout1) + inout1_offset;
        auto inout2_offset_first = std::begin(inout2) + inout2_offset;
        auto inout3_offset_first = std::begin(inout3) + inout3_offset;

#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_host_policies()(
                TestName(), inout1_offset_first, inout1_offset_first + n, inout2_offset_first, inout2_offset_first + n,
                inout3_offset_first, inout3_offset_first + n, n);
    }
}
