// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include "std_ranges_test.h"

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto copy_checker = [](std::ranges::random_access_range auto&& __r_in,
                           std::ranges::random_access_range auto&& __r_out, auto&&... args)
    {
        auto res = std::ranges::copy(std::forward<decltype(__r_in)>(__r_in), std::ranges::begin(__r_out),
            std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::copy_result<std::ranges::borrowed_iterator_t<decltype(__r_in)>,
            std::ranges::borrowed_iterator_t<decltype(__r_out)>>;
        return ret_type{res.in, res.out};
    };

    test_range_algo<0, int, data_in_out>{}(dpl_ranges::copy,  copy_checker);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
