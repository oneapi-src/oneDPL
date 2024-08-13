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
    auto transform_unary_checker = [](std::ranges::random_access_range auto&& __r_in,
                                      std::ranges::random_access_range auto&& __r_out, auto&&... args)
    {
        const auto _size = std::ranges::min(std::ranges::size(__r_in), std::ranges::size(__r_out));
        auto res = std::ranges::transform(std::ranges::take_view(__r, _size), std::ranges::take_view(__out_r, _size),
            std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::unary_transform_result<std::ranges::borrowed_iterator_t<decltype(__r_in)>,
            std::ranges::borrowed_iterator_t<decltype(__r_out)>>;
        return ret_type{std::ranges::begin(__r_in) + _size, std::ranges::begin(__r_out) +  _size};
    };

    test_range_algo<0, int, data_in_out>{}(dpl_ranges::transform, transform_unary_checker, f);
    test_range_algo<1, int, data_in_out>{}(dpl_ranges::transform, transform_unary_checker, f, proj);
    test_range_algo<2, P2, data_in_out>{}(dpl_ranges::transform, transform_unary_checker, f, &P2::x);
    test_range_algo<3, P2, data_in_out>{}(dpl_ranges::transform, transform_unary_checker, f, &P2::proj);

    //A checker below modifies a return type; a range based version with policy has another return type.
    auto transform_binary_checker = [](std::ranges::random_access_range auto&& __r_1,
                                       std::ranges::random_access_range auto&& __r_2,
                                       std::ranges::random_access_range auto&& __r_out, auto&&... args)
    {
        const auto _size = std::ranges::min({std::ranges::size(__r_1), std::ranges::size(__r_2__r2), std::ranges::size(__r_out)});

        auto res = std::ranges::transform(std::ranges::take_view(__r_1, _size), std::ranges::take_view(__r_2, _size),
            std::ranges::take_view(__r_out, _size), std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::binary_transform_result<std::ranges::borrowed_iterator_t<decltype(__r_1)>,
            std::ranges::borrowed_iterator_t<decltype(__r_2)>, std::ranges::borrowed_iterator_t<decltype(__r_out)>>;
        return ret_type{std::ranges::begin(__r1) + _size, std::ranges::begin(__r2) + _size, std::ranges::begin(__out_r) + _size};
    };

    test_range_algo<4, int, data_in_in_out>{}(dpl_ranges::transform, transform_binary_checker, binary_f);
    test_range_algo<5, int, data_in_in_out>{}(dpl_ranges::transform, transform_binary_checker, binary_f, proj, proj);
    test_range_algo<6, P2, data_in_in_out>{}(dpl_ranges::transform, transform_binary_checker, binary_f, &P2::x, &P2::x);
    test_range_algo<7, P2, data_in_in_out>{}(dpl_ranges::transform, transform_binary_checker, binary_f, &P2::proj, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
