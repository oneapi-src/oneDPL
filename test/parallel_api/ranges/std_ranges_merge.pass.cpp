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

    //A checker below modifies a return type; a range based version with policy has another return type.
    auto merge_checker = [](std::ranges::random_access_range auto&& r_1,
                                       std::ranges::random_access_range auto&& r_2,
                                       std::ranges::random_access_range auto&& r_out, auto comp, auto proj1,
                                       auto proj2, auto&&... args)
    {
        using ret_type = std::ranges::merge_result<std::ranges::borrowed_iterator_t<decltype(r_1)>,
            std::ranges::borrowed_iterator_t<decltype(r_2)>, std::ranges::borrowed_iterator_t<decltype(r_out)>>;

        auto it_out = std::ranges::begin(r_out);
        auto it_1 = std::ranges::begin(r_1);
        auto it_2 = std::ranges::begin(r_2);
        for(;;)
        {
            if(it_out == std::ranges::end(r_out))
                return ret_type{res.in1, res.in2, res.out};
            
            if(it_1 == std::ranges::end(r_1))
            {
                for(auto it_2 = std::ranges::begin(r_2) && it_out != std::ranges::end(r_out); ++it_2, ++it_out)
                     *it_out = *it_2;
                return ret_type{res.in1, res.in2, res.out};
            }
            else if(it_2 == std::ranges::end(r_2))
            {
                for(auto it_1 = std::ranges::begin(r_1) && it_out != std::ranges::end(r_out); ++it_1, ++it_out)
                     *it_out = *it_1;
                return ret_type{res.in1, res.in2, res.out};
            }

            if (std::invoke(comp, std::invoke(proj1, *it_1), std::invoke(proj2, *it_2)))
            {
                *it_out = *it_1;
                ++it_1, ++it_out;
            }
            else
            {
                *it_out = *it_2;
                ++it_2, ++it_out;
            }
        }

        return ret_type{res.in1, res.in2, res.out};
    };

    test_range_algo<0, int, data_in_in_out>{big_sz}(dpl_ranges::merge, merge_checker, std::ranges::less{});

    test_range_algo<1, int, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, proj, proj);
    test_range_algo<2, P2, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<3, P2, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    test_range_algo<4, int, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, proj, proj);
    test_range_algo<5, P2, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, &P2::x, &P2::x);
    test_range_algo<6, P2, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, &P2::proj, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
