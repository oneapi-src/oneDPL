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
#include "support/pstl_test_config.h"

#include <type_traits>
#include <functional>
#ifdef PSTL_STANDALONE_TESTS
#include <algorithm>
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/parallel_utils.h"

using namespace Parallel_TestUtils;

struct test_merge {
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
    void operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
        OutputIterator out_first, OutputIterator out_last, Compare comp) {
        using namespace std;
        {
            const auto res = merge(exec, first1, last1, first2, last2, out_first, comp);
            EXPECT_TRUE(res == out_last, "wrong return result from merge with predicate");
            EXPECT_TRUE(is_sorted(out_first, res, comp), "wrong result from merge with predicate");
            EXPECT_TRUE(includes(out_first, res, first1, last1, comp), "first sequence is not a part of result");
            EXPECT_TRUE(includes(out_first, res, first2, last2, comp), "second sequence is not a part of result");
        }
        {
            const auto res = merge(exec, first1, last1, first2, last2, out_first);
            EXPECT_TRUE(res == out_last, "wrong return result from merge");
            EXPECT_TRUE(is_sorted(out_first, res), "wrong result from merge");
        }
    }

    // for reverse iterators
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
    void operator()(Policy&& exec, std::reverse_iterator<InputIterator1> first1, std::reverse_iterator<InputIterator1> last1,
        std::reverse_iterator<InputIterator2> first2, std::reverse_iterator<InputIterator2> last2,
        std::reverse_iterator<OutputIterator> out_first, std::reverse_iterator<OutputIterator> out_last, Compare comp) {
        using namespace std;
        typedef typename std::iterator_traits<std::reverse_iterator<InputIterator1>>::value_type T;
        const auto res = merge(exec, first1, last1, first2, last2, out_first, std::greater<T>());

        EXPECT_TRUE(res == out_last, "wrong return result from merge with predicate");
        EXPECT_TRUE(is_sorted(out_first, res, std::greater<T>()), "wrong result from merge with predicate");
        EXPECT_TRUE(includes(out_first, res, first1, last1, std::greater<T>()), "first sequence is not a part of result");
        EXPECT_TRUE(includes(out_first, res, first2, last2, std::greater<T>()), "second sequence is not a part of result");
    }
};

template <typename T, typename Generator1, typename Generator2>
void test_merge_by_type(Generator1 generator1, Generator2 generator2) {
    using namespace std;
    size_t max_size = 100000;
    Sequence<T> in1(max_size, generator1);
    Sequence<T> in2(max_size / 2, generator2);
    Sequence<T> out(in1.size() + in2.size());
    std::sort(in1.begin(), in1.end());
    std::sort(in2.begin(), in2.end());

    for (size_t size = 0; size <= max_size; size = size <= 16 ? size + 1 : size_t(3.1415 * size)) {
        invoke_on_all_policies(test_merge(),  in1.cbegin(), in1.cbegin() + size,  in2.data(), in2.data() + size / 2, out.begin(), out.begin() + 1.5 * size, std::less<T>());
        invoke_on_all_policies(test_merge(), in1.data(), in1.data() + size, in2.cbegin(), in2.cbegin() + size / 2, out.begin(), out.begin() + 3 * size/2, std::less<T>());
    }
}


int32_t main( ) {
    test_merge_by_type<int32_t>([](size_t v) { return (v % 2 == 0 ? v : -v) * 3; },
        [](size_t v) { return v * 2; });
    test_merge_by_type<float64_t>([](size_t v) { return float64_t(v); },
        [](size_t v) { return float64_t(v - 100); });

#if !__PSTL_ICC_16_17_TEST_64_TIMEOUT
    test_merge_by_type<Wrapper<int16_t>>([](size_t v) { return Wrapper<int16_t>(v % 100); },
        [](size_t v) { return Wrapper<int16_t>(v % 10); });
#endif

    std::cout << done() << std::endl;
    return 0;
}
