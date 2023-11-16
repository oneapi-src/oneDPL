// -*- C++ -*-
//===-- transform_unary.pass.cpp ------------------------------------------===//
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

// Tests for transform

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#if TEST_DPCPP_BACKEND_PRESENT
#    include "oneapi/dpl/async"
#endif // TEST_DPCPP_BACKEND_PRESENT

#include "support/utils.h"

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
template <typename InputIterator, typename OutputIterator>
void
check_and_reset(InputIterator first, InputIterator last, OutputIterator out_first)
{
    typedef typename ::std::iterator_traits<OutputIterator>::value_type Out;
    typename ::std::iterator_traits<OutputIterator>::difference_type k = 0;
    for (; first != last; ++first, ++out_first, ++k)
    {
        // check
        Out expected = 1 - std::get<0>(*first) - std::get<1>(*first);
        Out actual = *out_first;
        EXPECT_EQ(expected, actual, "wrong value in output sequence");
        // reset
        *out_first = k % 7 != 4 ? 7 * k - 5 : 0;
    }
}

template <typename T1, typename T2>
struct test_one_policy
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename UnaryOp>
    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<Policy>
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, UnaryOp op)
    {
        auto result = oneapi::dpl::experimental::transform_async(exec, first, last, out_first, op);
        result.wait();

        check_and_reset(first, last, out_first);
    }

    template <typename Policy, typename InputIterator, typename OutputIterator, typename UnaryOp>
    ::std::enable_if_t<!oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>>, void>
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, UnaryOp op)
    {
        // transform_async implemented now only for hetero policy
    }
};

template <typename TInput, typename TOutput>
class ZipComplement
{
    std::int32_t val;

public:

    ZipComplement(std::size_t v) : val(v) {}

    TOutput operator()(const ::std::tuple<const TInput&, const TInput&>& x) const
    {
        return TOutput(val - std::get<0>(x) - std::get<1>(x));
    }
};

template <typename Tin, typename Tout>
void
test()
{
    // TODO required to start from 0 when it will be implemented in __pattern_walk3_async and all other patterns
    for (size_t n = 1; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<Tin> in_a(n, [](std::int32_t k) { return k % 5 != 1 ? 3 * k - 7 : 0; });
        Sequence<Tin> in_b(n, [](std::int32_t k) { return 0; });

        auto in_zip_begin = dpl::make_zip_iterator(in_a.begin(), in_b.begin());
        auto in_zip_end = dpl::make_zip_iterator(in_a.end(), in_b.end());

        Sequence<Tout> out(n);
        const ZipComplement<Tin, Tout> flip(1);

        invoke_on_all_policies<0>()(test_one_policy<Tin, Tout>(), in_zip_begin, in_zip_end, out.begin(), out.end(), flip);
#if !ONEDPL_FPGA_DEVICE
        {
            auto in_zip_cbegin = dpl::make_zip_iterator(in_a.cbegin(), in_b.cbegin());
            auto in_zip_cend = dpl::make_zip_iterator(in_a.cend(), in_b.cend());

            invoke_on_all_policies<1>()(test_one_policy<Tin, Tout>(), in_zip_cbegin, in_zip_cend, out.begin(), out.end(), flip);
        }
#endif // ONEDPL_FPGA_DEVICE
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test<std::int32_t, std::int32_t>();
    test<std::int32_t, float32_t>();
    test<std::uint16_t, float32_t>();
    test<float32_t, float64_t>();
    test<float64_t, float64_t>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
