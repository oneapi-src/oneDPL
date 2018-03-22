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

#include <algorithm>
#include <functional>
#include "pstl/execution"
#include "pstl/algorithm"
#include "pstl/numeric"
#include "pstl/memory"

#include "test/utils.h"

using namespace TestUtils;

struct test_merge {
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Compare>
    void operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2,
        Iterator3 first3, Compare comp) {
        using namespace std;
        const auto dist1 = distance(first1, last1);
        const auto dist2 = distance(first2, last2);
        const auto expected = std::next(first3, (dist1 + dist2));
        {
            const auto res = merge(exec, first1, last1, first2, last2, first3, comp);
            EXPECT_TRUE(res == expected, "wrong return result from merge with predicate");
            EXPECT_TRUE(is_sorted(first3, res, comp), "wrong result from merge with predicate");
            EXPECT_TRUE(includes(first3, res, first1, last1, comp), "first sequence is not a part of result");
            EXPECT_TRUE(includes(first3, res, first2, last2, comp), "second sequence is not a part of result");
        }
        {
            const auto res = merge(exec, first1, last1, first2, last2, first3);
            EXPECT_TRUE(res == expected, "wrong return result from merge");
            EXPECT_TRUE(is_sorted(first3, res), "wrong result from merge");
        }
    }
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Compare>
    void operator()(Policy&& exec, std::reverse_iterator<Iterator1> first1, std::reverse_iterator<Iterator1> last1, std::reverse_iterator<Iterator2> first2, std::reverse_iterator<Iterator2> last2, std::reverse_iterator<Iterator3> first3, Compare comp) { }
};

template <typename T, typename Generator1, typename Generator2>
void test_merge_by_type(Generator1 generator1, Generator2 generator2) {
    using namespace std;
    size_t max_size = 100000;
    Sequence<T> in1(max_size, generator1);
    Sequence<T> in2(max_size / 2, generator2);
    Sequence<T> in3(in1.size() + in2.size());
    std::sort(in1.begin(), in1.end());
    std::sort(in2.begin(), in2.end());

    for (size_t size = 0; size <= max_size; size = size <= 16 ? size + 1 : size_t(3.1415 * size)) {
        invoke_on_all_policies(test_merge(),  in1.begin(),  in1.begin() + size,  in2.begin(),  in2.begin() + size / 2, in3.begin(), std::less<T>());
        invoke_on_all_policies(test_merge(), in1.cbegin(), in1.cbegin() + size, in2.cbegin(), in2.cbegin() + size / 2, in3.begin(), std::less<T>());
    }

}


int32_t main( ) {

    test_merge_by_type<int32_t>([](size_t v) { return (v % 2 == 0 ? v : -v) * 3; },
        [](size_t v) { return v * 2; });
    test_merge_by_type<float64_t>([](size_t v) { return float64_t(v); },
        [](size_t v) { return float64_t(v - 100); });
    test_merge_by_type<Wrapper<int16_t>>([](size_t v) { return Wrapper<int16_t>(v % 100); },
        [](size_t v) { return Wrapper<int16_t>(v % 10); });

    std::cout << done() << std::endl;
    return 0;
}
