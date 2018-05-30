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

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

using namespace TestUtils;

struct test_one_policy {
    //dummy specialization by policy type, in case of broken configuration
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN

    template <typename Iterator1, typename Predicate>
    void operator()(pstl::execution::unsequenced_policy, Iterator1 begin1, Iterator1 end1, Predicate pred) {}
    template <typename Iterator1, typename Predicate>
    void operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 begin1, Iterator1 end1, Predicate pred) {}
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Predicate>
    void operator()(ExecutionPolicy&& exec, Iterator1 begin1, Iterator1 end1, Predicate pred) {
        const bool expected = std::is_partitioned(begin1, end1, pred);
        const bool actual   = std::is_partitioned(exec, begin1, end1, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from is_partitioned");
    }
};

template <typename T, typename Predicate>
void test(Predicate pred) {

    const std::size_t max_n = 1000000;
    Sequence<T> in(max_n, [](std::size_t k) {return T(k); });

    for (std::size_t n1 = 0; n1 <= max_n; n1 = n1 <= 16 ? n1 + 1 : std::size_t(3.1415 * n1)) {
        invoke_on_all_policies(test_one_policy(),  in.begin(),  in.begin() + n1, pred);
        std::partition(in.begin(), in.begin() + n1, pred);
        invoke_on_all_policies(test_one_policy(), in.cbegin(), in.cbegin() + n1, pred);
    }
}

template<typename T>
struct LocalWrapper {
    explicit LocalWrapper(std::size_t k) : my_val(k) { }
private:
    T my_val;
};

int32_t main( ) {
    test<float64_t>([](const float64_t x) {return x < 0; });
    test<int32_t>([](const int32_t x) {return x > 1000; });
    test<uint16_t>([](const uint16_t x) {return x % 5 < 3; });
#if !__PSTL_ICC_18_TEST_EARLY_EXIT_MONOTONIC_RELEASE_BROKEN
    test<LocalWrapper<float64_t>>(
        [](const LocalWrapper<float64_t>& x) {return true; }
    );
#endif

    std::cout << done() << std::endl;
    return 0;
}
