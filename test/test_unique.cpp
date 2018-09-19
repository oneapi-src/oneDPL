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

// Test for unique
#include "pstl_test_config.h"

#include "pstl/execution"
#include "pstl/algorithm"
#include "utils.h"

using namespace TestUtils;

struct run_unique {
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN//dummy specialization by policy type, in case of broken configuration
    template<typename ForwardIt, typename Generator>
    void operator()(pstl::execution::unsequenced_policy, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, Generator generator) {}

    template<typename ForwardIt, typename Generator>
    void operator()(pstl::execution::parallel_unsequenced_policy, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, Generator generator) {}
    
    template<typename ForwardIt, typename BinaryPred, typename Generator>
    void operator()(pstl::execution::unsequenced_policy, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, BinaryPred pred, Generator generator) {}

    template<typename ForwardIt, typename BinaryPred, typename Generator>
    void operator()(pstl::execution::parallel_unsequenced_policy, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, BinaryPred pred, Generator generator) {}
#endif

    template<typename Policy, typename ForwardIt, typename Generator>
    void operator()(Policy&& exec, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, Generator generator) {
        using namespace std;

        // Preparation
        fill_data(first1, last1, generator);
        fill_data(first2, last2, generator);

        ForwardIt i = unique(first1, last1);
        ForwardIt k = unique(exec, first2, last2);

        auto n = std::distance(first1, i);
        EXPECT_TRUE(std::distance(first2, k) == n, "wrong return value from unique without predicate");
        EXPECT_EQ_N(first1, first2, n, "wrong effect from unique without predicate");
    }

    template<typename Policy, typename ForwardIt, typename BinaryPred, typename Generator>
    void operator()(Policy&& exec, ForwardIt first1, ForwardIt last1,
        ForwardIt first2, ForwardIt last2, BinaryPred pred, Generator generator) {
        using namespace std;

        // Preparation
        fill_data(first1, last1, generator);
        fill_data(first2, last2, generator);

        ForwardIt i = unique(first1, last1, pred);
        ForwardIt k = unique(exec, first2, last2, pred);

        auto n = std::distance(first1, i);
        EXPECT_TRUE(std::distance(first2, k) == n, "wrong return value from unique with predicate");
        EXPECT_EQ_N(first1, first2, n, "wrong effect from unique with predicate");
    }
};

template<typename T, typename Generator, typename Predicate>
void test(Generator generator, Predicate pred) {
    const std::size_t max_size = 1000000;
    Sequence<T> in(max_size, [](size_t v) {return T(v); });
    Sequence<T> exp(max_size, [](size_t v) {return T(v); });

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n) ) {
        invoke_on_all_policies(run_unique(), exp.begin(), exp.begin() + n, in.begin(), in.begin() + n,       generator);
        invoke_on_all_policies(run_unique(), exp.begin(), exp.begin() + n, in.begin(), in.begin() + n, pred, generator);
    }
}

template<typename T>
struct LocalWrapper {
    T my_val;

    explicit LocalWrapper(T k): my_val(k) { }
    LocalWrapper(LocalWrapper&& input): my_val(std::move(input.my_val)) { }
    LocalWrapper& operator=(LocalWrapper&& input) {
        my_val = std::move(input.my_val);
        return *this;
    }
    friend bool operator==(const LocalWrapper<T>& x, const LocalWrapper<T>& y) {
        return x.my_val == y.my_val;
    }
};

int32_t main( ) {
#if !__PSTL_ICC_16_17_18_TEST_UNIQUE_MASK_RELEASE_BROKEN
    test<int32_t>([](size_t j) {return j / 3; },
        [](const int32_t& val1, const int32_t& val2) {return val1 * val1 == val2 * val2; });
    test<float64_t>([](size_t) {return float64_t(1); },
        [](const float64_t& val1, const float64_t& val2) {return val1 != val2; });
#endif
    test<LocalWrapper<uint32_t>>([](size_t j) {return LocalWrapper<uint32_t>(j); },
        [](const LocalWrapper<uint32_t>& val1, const LocalWrapper<uint32_t>& val2) {return val1.my_val != val2.my_val; });

    std::cout << done() << std::endl;
    return 0;
}
