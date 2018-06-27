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

// Tests for unique_copy that uses == as equivalence relationship
// Tests for copy_if and remove_copy_if
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

struct run_unique_copy {
    template<typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size, typename Predicate, typename T>
    void operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
        OutputIterator2 expected_first, OutputIterator2 expected_last, Size n, Predicate pred, T trash) {
        // Cleaning
        std::fill_n(expected_first, n, trash);
        std::fill_n(out_first, n, trash);

        // Run unique_copy
        auto i = unique_copy(first, last, expected_first);
        auto k = unique_copy(exec, first, last, out_first);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong unique_copy effect");
        for (size_t j = 0; j < GuardSize; ++j) {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from unique_copy");

        // Cleaning
        std::fill_n(expected_first, n, trash);
        std::fill_n(out_first, n, trash);
        // Run unique_copy with predicate
        i = unique_copy(first, last, expected_first, pred);
        k = unique_copy(exec, first, last, out_first, pred);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong unique_copy with predicate effect");
        for (size_t j = 0; j < GuardSize; ++j) {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from unique_copy with predicate");
        }
    };

template<typename T, typename BinaryPredicate, typename Convert>
void test(T trash, BinaryPredicate pred, Convert convert, bool check_weakness = true) {
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        Sequence<T> in(n, [&](size_t k) -> T {
            return convert(k^n);
        });
        using namespace std;
        size_t count = GuardSize;
        for (size_t k = 0; k<in.size(); ++k)
            count += k == 0 || !pred(in[k], in[k - 1]) ? 1 : 0;
        Sequence<T> out(count, [=](size_t) {return trash; });
        Sequence<T> expected(count, [=](size_t) {return trash; });
        if (check_weakness) {
            auto expected_result = unique_copy(in.begin(), in.end(), expected.begin(), pred);
            size_t m = expected_result - expected.begin();
            EXPECT_TRUE(n / (n<10000 ? 4 : 6) <= m && m <= (3 * n + 1) / 4, "weak test for unique_copy");
        }
        invoke_on_all_policies(run_unique_copy(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(), expected.end(), count, pred, trash);
    }
}

int main( int argc, char* argv[] ) {
    test<Number>( Number(42,OddTag()),
                  std::equal_to<Number>(),
                  [](int32_t j){return Number(3*j/13^(j&8),OddTag());});

    test<float32_t>(float32_t(42),
        std::equal_to<float32_t>(),
        [](int32_t j) {return float32_t(5 * j / 23 ^ (j / 7)); });
#if !__PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN
    test<float32_t>(float32_t(42),
        [](float32_t x, float32_t y) {return false; },
        [](int32_t j) {return float32_t(j); }, false);
#endif
    std::cout << "done" << std::endl;
    return 0;
}
