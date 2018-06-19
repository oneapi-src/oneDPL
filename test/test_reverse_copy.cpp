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

#include "test/pstl_test_config.h"

#include <iterator>

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

using namespace TestUtils;

template <typename T>
struct wrapper {
    T t;
    wrapper() {}
    explicit wrapper(T t_) : t(t_) {}
    void operator=(const T& t_) {
        t = t_;
    }
};

template <typename T>
bool eq(const wrapper<T>& a, const wrapper<T>& b) {
    return a.t == b.t;
}

template <typename T>
bool eq(const T& a, const T& b) {
    return a == b;
}

// we need to save state here, because we need to test with different types of iterators
// due to the caller invoke_on_all_policies does forcing modification passed iterator type to cover additional usage cases.
template <typename Iterator>
struct test_one_policy {
    Iterator data_b;
    Iterator data_e;
    test_one_policy(Iterator b, Iterator e): data_b(b), data_e(e) {}

#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN // dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
        operator()(pstl::execution::unsequenced_policy, Iterator1 actual_b, Iterator1 actual_e) { }
    template <typename Iterator1>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
        operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 actual_b, Iterator1 actual_e) { }
#endif

    template <typename ExecutionPolicy, typename Iterator1>
    void operator()(ExecutionPolicy&& exec, Iterator1 actual_b, Iterator1 actual_e) {
        using namespace std;
        using T = typename iterator_traits<Iterator1>::value_type;

        fill(actual_b, actual_e, T(-123));
        Iterator1 actual_return = reverse_copy(exec, data_b, data_e, actual_b);

        EXPECT_TRUE(actual_return == actual_e, "wrong result of reverse_copy");

        if (actual_b != actual_e) {
            T temp = *actual_b;

            // check effect of reverse_copy
            while (actual_b != actual_e) {
                --data_e;
                temp = *data_e;
                if (!eq(temp, *actual_b)) {
                    EXPECT_TRUE(false, "wrong effect of reverse_copy");
                    break;
                }
                ++actual_b;
            }
        }
    }
};

template <typename T1, typename T2>
void test() {
    typedef typename Sequence<T1>::iterator iterator_type;
    typedef typename Sequence<T1>::const_bidirectional_iterator cbi_iterator_type;

    const std::size_t max_len = 100000;

    Sequence<T2> actual(max_len);

    Sequence<T1> data(max_len, [](std::size_t i) {
        return T1(i);
    });

    for (std::size_t len = 0; len < max_len; len = len <= 16 ? len + 1 : std::size_t(3.1415 * len)) {
        invoke_on_all_policies(test_one_policy<iterator_type>(data.begin(), data.begin() + len),
            actual.begin(), actual.begin() + len);
        invoke_on_all_policies(test_one_policy<cbi_iterator_type>(data.cbibegin(), std::next(data.cbibegin(), len)),
            actual.begin(), actual.begin() + len);
    }
}

int32_t main() {
    test<int32_t, int8_t>();
    test<uint16_t, float32_t>();
    test<float64_t, int64_t>();
    test<wrapper<float64_t>, wrapper<float64_t>>();

    std::cout << done() << std::endl;
    return 0;
}
