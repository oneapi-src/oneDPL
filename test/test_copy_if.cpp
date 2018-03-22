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

// Tests for copy_if and remove_copy_if
#include "test/pstl_test_config.h"

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

using namespace TestUtils;

const size_t GuardSize = 5;

struct run_copy_if {
    template<typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size, typename Predicate, typename T>
    void operator()(Policy&& exec, InputIterator first, InputIterator last,  OutputIterator out_first, OutputIterator out_last,
        OutputIterator2 expected_first, Size n, Predicate pred, T trash) {
        // Cleaning
        std::fill_n(expected_first, n, trash);
        std::fill_n(out_first, n, trash);

        // Run copy_if
        auto i = copy_if(first, last, expected_first, pred);
        auto k = copy_if(exec, first, last, out_first, pred);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong copy_if effect");
        for (size_t j = 0; j < GuardSize; ++j) {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from copy_if");

        // Cleaning
        std::fill_n(expected_first, n, trash);
        std::fill_n(out_first, n, trash);
        // Run remove_copy_if
        i = remove_copy_if(first, last, expected_first, [=](const T& x) {return !pred(x); });
        k = remove_copy_if(exec, first, last, out_first, [=](const T& x) {return !pred(x); });
        EXPECT_EQ_N(expected_first, out_first, n, "wrong remove_copy_if effect");
        for (size_t j = 0; j < GuardSize; ++j) {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from remove_copy_if");
    }
};

template<typename T, typename Predicate, typename Convert>
void test(T trash, Predicate pred, Convert convert, bool check_weakness = true) {
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n) ) {
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        size_t count = GuardSize;
        Sequence<T> in(n, [&](size_t k) -> T {
            T val = convert(n^k);
            count += pred(val) ? 1 : 0;
            return val;
        });

        Sequence<T> out(count, [=](size_t){return trash;});
        Sequence<T> expected(count, [=](size_t){return trash;});
        auto expected_result = copy_if( in.cfbegin(), in.cfend(), expected.begin(), pred );
        if (check_weakness) {
            size_t m = expected_result - expected.begin();
            EXPECT_TRUE(n / 4 <= m && m <= 3 * (n + 1) / 4, "weak test for copy_if");
        }
        invoke_on_all_policies(run_copy_if(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(), count, pred, trash);
        invoke_on_all_policies(run_copy_if(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(), count, pred, trash);
    }
}

int32_t main( ) {
    test<float64_t>( -666.0,
                 [](const float64_t& x) {return x*x<=1024;},
                 [](size_t j){return ((j+1)%7&2)!=0? float64_t(j%32) : float64_t(j%33+34);});

    test<int32_t>( -666,
               [](const int32_t& x) {return x!=42;},
               [](size_t j){return ((j+1)%5&2)!=0? int32_t(j+1) : 42;});

#if !__PSTL_TEST_ICC_17_IA32_RELEASE_MAC_BROKEN
    test<Number>( Number(42,OddTag()),
                  IsMultiple(3,OddTag()),
                  [](int32_t j){return Number(j,OddTag());});
#endif

#if !__PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN
    test<int32_t>( -666,
               [](const int32_t& x) {return true;},
               [](size_t j){return j;}, false);
#endif
    std::cout << "done" << std::endl;
    return 0;
}
