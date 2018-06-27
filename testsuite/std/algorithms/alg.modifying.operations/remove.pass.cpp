/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

// Test for remove, remove_if
#include "support/pstl_test_config.h"

#include <type_traits>
#ifdef PSTL_STANDALONE_TESTS
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/parallel_utils.h"

using namespace Parallel_TestUtils;

struct run_remove {
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN//dummy specialization by policy type, in case of broken configuration
    template <typename InputIterator, typename OutputIterator, typename Size, typename T>
    void operator()(pstl::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
        OutputIterator expected_first, OutputIterator expected_last, Size n, const T& value) { }
    template <typename InputIterator, typename OutputIterator, typename Size, typename T>
    void operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
        OutputIterator expected_first, OutputIterator expected_last, Size n, const T& value) { }
#endif

    template<typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename T>
    void operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
        OutputIterator expected_first, OutputIterator expected_last, Size n, const T& value) {
            // Cleaning
            std::copy(first, last, expected_first);
            std::copy(first, last, out_first);

            // Run remove
            OutputIterator i = remove(expected_first, expected_last, value);
            OutputIterator k = remove(exec, out_first, out_last, value);
            EXPECT_EQ_N(expected_first, out_first, n, "wrong remove effect");
            EXPECT_TRUE(std::distance(expected_first, i) == std::distance(out_first, k), "wrong return value from remove");
    }
};

struct run_remove_if {
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN//dummy specialization by policy type, in case of broken configuration
    template <typename InputIterator, typename OutputIterator, typename Size, typename Predicate>
    void operator()(pstl::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
        OutputIterator expected_first, OutputIterator expected_last, Size n, Predicate pred) { }
    template <typename InputIterator, typename OutputIterator, typename Size, typename Predicate>
    void operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
        OutputIterator expected_first, OutputIterator expected_last, Size n, Predicate pred) { }
#endif

    template<typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename Predicate>
    void operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
        OutputIterator expected_first, OutputIterator expected_last, Size n, Predicate pred) {
            // Cleaning
            std::copy(first, last, expected_first);
            std::copy(first, last, out_first);

            // Run remove_if
            OutputIterator i = remove_if(expected_first, expected_last, pred);
            OutputIterator k = remove_if(exec, out_first, out_last, pred);
            EXPECT_EQ_N(expected_first, out_first, n, "wrong remove_if effect");
            EXPECT_TRUE(std::distance(expected_first, i) == std::distance(out_first, k), "wrong return value from remove_if");
    }
};

template<typename T, typename Predicate, typename Convert>
void test(T trash, const T& value, Predicate pred, Convert convert) {
    const std::size_t max_size = 100000;
    Sequence<T> out(max_size, [trash](size_t) {return trash; });
    Sequence<T> expected(max_size, [trash](size_t) {return trash; });

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n) ) {
        Sequence<T> data(n, [&](size_t k) -> T {return convert(k); });

        invoke_on_all_policies(run_remove(), data.begin(), data.end(), out.begin(), out.begin() + n,
            expected.begin(), expected.begin() + n, n, value);
        invoke_on_all_policies(run_remove_if(), data.begin(), data.end(), out.begin(), out.begin() + n,
            expected.begin(), expected.begin() + n, n, pred);
    }
}

int32_t main( ) {
#if !__PSTL_ICC_18_TEST_EARLY_EXIT_MONOTONIC_RELEASE_BROKEN
    test<int32_t>(666, 42, [](int32_t val) {return true; },
        [](size_t j) {return j; });
#endif

    test<int32_t>(666, 2001, [](const int32_t& val) {return val != 2001; },
        [](size_t j) {return ((j + 1) % 5 & 2) != 0 ? 2001 : -1 - int32_t(j); });
    test<float64_t>(-666.0, 8.5, [](const float64_t& val) {return val != 8.5; },
        [](size_t j) {return ((j + 1) % 7 & 2) != 0 ? 8.5 : float64_t(j % 32 + j); });

#if !__PSTL_ICC_17_TEST_MAC_RELEASE_32_BROKEN
    test<Number>(Number(-666, OddTag()), Number(42, OddTag()),
        IsMultiple(3, OddTag()),
        [](int32_t j) {return Number(j, OddTag()); });
#endif

    std::cout << done() << std::endl;
    return 0;
}
