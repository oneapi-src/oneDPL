// -*- C++ -*-
//===-- transform_binary_zip.pass.cpp -------------------------------------===//
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

template <typename T>
constexpr T
get_epsilon();

template <>
constexpr float32_t
get_epsilon<float32_t>()
{
    return 1e-7;
}

template <typename TInput1, typename TInput2, typename TOutput>
class TheOperation
{
    TOutput val;

public:

    TheOperation(TOutput v) : val(v) {}

    TOutput
    operator()(const ::std::tuple<const TInput1&, const TInput1&>& x,
               const ::std::tuple<const TInput2&, const TInput2&>& y) const
    {
        return TOutput(val + (::std::get<0>(x) + ::std::get<1>(x)) - (::std::get<0>(y) + ::std::get<1>(y)));
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
        Out expected = Out(1.5) + (::std::get<0>(*first1) + ::std::get<1>(*first1)) - (::std::get<0>(*first2) + ::std::get<1>(*first2));
        Out actual = *out_first;
        if constexpr (::std::is_floating_point_v<Out>)
        {
            EXPECT_TRUE((expected > actual ? expected - actual : actual - expected) < get_epsilon<Out>(),
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

template <typename TInput1, typename TInput2, typename TOutput, typename Predicate>
void
test(Predicate pred)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<TInput1> in1_a(n, [](size_t k) { return k % 5 != 1 ? TInput1(3 * k + 7) : 0; });
        Sequence<TInput1> in1_b(n, [](size_t k) { return 0; });
        auto in1_zip_begin = dpl::make_zip_iterator(in1_a.begin(), in1_b.begin());
        auto in1_zip_end   = dpl::make_zip_iterator(in1_a.end(),   in1_b.end());

        Sequence<TInput2> in2_a(n, [](size_t k) { return k % 7 != 2 ? TInput2(5 * k + 5) : 0; });
        Sequence<TInput2> in2_b(n, [](size_t k) { return 0; });
        auto in2_zip_begin = dpl::make_zip_iterator(in2_a.begin(), in1_b.begin());
        auto in2_zip_end   = dpl::make_zip_iterator(in2_a.end(),   in2_b.end());

        Sequence<TOutput> out(n, [](size_t) { return -1; });

        invoke_on_all_policies<0>()(test_one_policy<TInput1, TInput2, TOutput>(),
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

            invoke_on_all_policies<1>()(test_one_policy<TInput1, TInput2, TOutput>(),
                                        in1_zip_cbegin, in1_zip_cend,
                                        in2_zip_cbegin, in2_zip_cend,
                                        out.begin(), out.end(),
                                        pred);
        }
#endif // ONEDPL_FPGA_DEVICE
    }
}

template <typename TInput1, typename TInput2, typename TOutput>
void test_with_pred(TOutput initial_value)
{
    TheOperation<TInput1, TInput2, TOutput> pred(initial_value);

    test<TInput1, TInput2, TOutput>(pred);
}

template <typename TInput1, typename TInput2, typename TOutput>
void
test_with_pred_non_const(TOutput initial_value)
{
    TheOperation<TInput1, TInput2, TOutput> pred(initial_value);

    test<TInput1, TInput2, TOutput>(non_const(pred));
}

template <typename TInput1, typename TInput2, typename TOutput>
void
test_with_lambda(TOutput initial_value)
{
    test<TInput1, TInput2, TOutput>(
        [initial_value](const ::std::tuple<const TInput1&, const TInput1&>& x,
                        const ::std::tuple<const TInput2&, const TInput2&>& y)
        {
            return TOutput(initial_value + (::std::get<0>(x) + ::std::get<1>(x)) - (::std::get<0>(y) + ::std::get<1>(y)));
        });
}

int
main()
{
    //const operator()
    test_with_pred<std::int32_t, std::int32_t, std::int32_t>(1);
    test_with_pred<float32_t, float32_t, float32_t>(1.5);

    //non-const operator()
    test_with_pred_non_const<std::int32_t, std::int32_t, std::int32_t>(1);
    test_with_pred_non_const<float32_t, float32_t, float32_t>(1.5);

    // lambda
    test_with_lambda<std::int32_t, std::int32_t, std::int32_t>(1);
    test_with_lambda<float32_t, float32_t, float32_t>(1.5);

    return TestUtils::done();
}
