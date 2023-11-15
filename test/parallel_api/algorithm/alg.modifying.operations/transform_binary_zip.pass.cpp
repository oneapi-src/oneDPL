// -*- C++ -*-
//===-- transform_binary.pass.cpp -----------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

template <typename... Types>
using typle_t = oneapi::dpl::__internal::tuple<Types...>;
//using typle_t = ::std::tuple<Types...>;

template <typename Out>
class TheOperation
{
    Out val;

  public:

    TheOperation(Out v) : val(v) {}

    template <typename T1A, typename T1B, typename T2A, typename T2B>
    Out
    operator()(const typle_t<T1A, T1B>& x, const typle_t<T2A, T2B>& y) const
    {
        return Out(val + (std::get<0>(x) + std::get<1>(x)) - (std::get<0>(y) + std::get<1>(y)));
    }
};

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
void
check_and_reset(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator out_first)
{
    typedef typename ::std::iterator_traits<OutputIterator>::value_type Out;
    typename ::std::iterator_traits<OutputIterator>::difference_type k = 0;
    for (; first1 != last1; ++first1, ++first2, ++out_first, ++k)
    {
        // check
        Out expected = Out(1.5) + (std::get<0>(*first1) + std::get<1>(*first1)) - (std::get<0>(*first2) + std::get<1>(*first2));
        Out actual = *out_first;
        if (::std::is_floating_point_v<Out>)
        {
            EXPECT_TRUE((expected > actual ? expected - actual : actual - expected) < Out(1e-7),
                        "wrong value in output sequence");
        }
        else
        {
            EXPECT_EQ(expected, actual, "wrong value in output sequence");
        }
        // reset
        *out_first = k % 7 != 4 ? 7 * k + 5 : 0;
    }
}

template <typename T1, typename T2, typename T3>
struct test_one_policy
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
              typename BinaryOp>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 /* last2 */,
               OutputIterator out_first, OutputIterator /* out_last */, BinaryOp op)
    {
        ::std::transform(exec, first1, last1, first2, out_first, op);
        check_and_reset(first1, last1, first2, out_first);
    }
};

template <typename In1, typename In2, typename Out, typename Predicate>
void
test(Predicate pred)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In1> in1_a(n, [](size_t k) { return k % 5 != 1 ? In1(3 * k + 7) : 0; });
        Sequence<In1> in1_b(n, [](size_t k) { return 0; });
        auto in1_zip_begin = dpl::make_zip_iterator(in1_a.begin(), in1_b.begin());
        auto in1_zip_end   = dpl::make_zip_iterator(in1_a.end(),   in1_b.end());

        Sequence<In2> in2_a(n, [](size_t k) { return k % 7 != 2 ? In2(5 * k + 5) : 0; });
        Sequence<In2> in2_b(n, [](size_t k) { return 0; });
        auto in2_zip_begin = dpl::make_zip_iterator(in2_a.begin(), in1_b.begin());
        auto in2_zip_end   = dpl::make_zip_iterator(in2_a.end(),   in2_b.end());

        Sequence<Out> out(n, [](size_t) { return -1; });

        invoke_on_all_policies<0>()(test_one_policy<In1, In2, Out>(),
                                    in1_zip_begin, in1_zip_end,
                                    in2_zip_begin, in2_zip_end,
                                    out.begin(), out.end(),
                                    pred);

#if !ONEDPL_FPGA_DEVICE
        {
            auto in1_zip_cbegin = dpl::make_zip_iterator(in1_a.cbegin(), in1_b.cbegin());
            auto in1_zip_cend   = dpl::make_zip_iterator(in1_a.cend(),   in1_b.cend());

            auto in2_zip_cbegin = dpl::make_zip_iterator(in2_a.cbegin(), in1_b.cbegin());
            auto in2_zip_cend   = dpl::make_zip_iterator(in2_a.cend(),   in2_b.cend());

            invoke_on_all_policies<1>()(test_one_policy<In1, In2, Out>(),
                                        in1_zip_cbegin, in1_zip_cend,
                                        in2_zip_cbegin, in2_zip_cend,
                                        out.begin(), out.end(),
                                        pred);
        }
#endif // ONEDPL_FPGA_DEVICE
    }
}

int
main()
{
    //const operator()
    test<std::int32_t, std::int32_t, std::int32_t>(TheOperation<std::int32_t>(1));
    test<float32_t, float32_t, float32_t>(TheOperation<float32_t>(1.5));

    //non-const operator()
    test<std::int32_t, std::int32_t, std::int32_t>(non_const(TheOperation<std::int32_t>(1)));
    test<float32_t, float32_t, float32_t>(non_const(TheOperation<float32_t>(1.5)));

    // lambda
    test<std::int32_t, std::int32_t, std::int32_t>(
        [](const typle_t<const std::int32_t&, const std::int32_t&>& x, const typle_t<const std::int32_t&, const std::int32_t&>& y)
        {
            return std::int32_t(1 + (std::get<0>(x) + std::get<1>(x)) - (std::get<0>(y) + std::get<1>(y)));
        });

    test<float32_t, float32_t, float32_t>(
        [](const typle_t<const float32_t&, const float32_t&>& x, const typle_t<const float32_t&, const float32_t&>& y)
        {
            return float32_t(1 + (std::get<0>(x) + std::get<1>(x)) - (std::get<0>(y) + std::get<1>(y)));
        });

    return TestUtils::done();
}
