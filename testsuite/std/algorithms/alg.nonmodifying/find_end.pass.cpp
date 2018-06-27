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

#include "support/pstl_test_config.h"

#ifdef PSTL_STANDALONE_TESTS
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/parallel_utils.h"

using namespace Parallel_TestUtils;

struct test_one_policy {
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN//dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void operator()(pstl::execution::unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred) { }
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred) { }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred) {
        using namespace std;
        // For find_end
        {
            auto expected = find_end(b, e, bsub, esub, pred);
            auto actual = find_end(exec, b, e, bsub, esub);
            EXPECT_TRUE(actual == expected, "wrong return result from find_end");

            actual = find_end(exec, b, e, bsub, esub, pred);
            EXPECT_TRUE(actual == expected, "wrong return result from find_end with a predicate");
        }

        // For search
        {
            auto expected = search(b, e, bsub, esub, pred);
            auto actual = search(exec, b, e, bsub, esub);
            EXPECT_TRUE(actual == expected, "wrong return result from search");

            actual = search(exec, b, e, bsub, esub, pred);
            EXPECT_TRUE(actual == expected, "wrong return result from search with a predicate");
        }
    }
};

template <typename T>
void test(const std::size_t bits) {

    const std::size_t max_n1 = 1000;
    const std::size_t max_n2 = (max_n1 * 10) / 8;
    Sequence<T> in(max_n1, [max_n1, bits](std::size_t k) {return T(2 * HashBits(max_n1, bits - 1) ^ 1); });
    Sequence<T> sub(max_n2, [max_n1, bits](std::size_t k) {return T(2 * HashBits(max_n1, bits - 1)); });
    for (std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1)) {
        std::size_t sub_n[] = { 0, 1, 3, n1, (n1 * 10) / 8 };
        std::size_t res[] = { 0, 1, n1 / 2, n1 };
        for(auto n2 : sub_n) {
            for(auto r : res) {
                std::size_t i = r, isub = 0;
                for(; i < n1 & isub < n2; ++i, ++isub)
                    in[i] = sub[isub];
                invoke_on_all_policies(test_one_policy(), in.begin(), in.begin() + n1, sub.begin(), sub.begin() + n2, std::equal_to<T>());
                invoke_on_all_policies(test_one_policy(), in.cbegin(), in.cbegin() + n1, sub.cbegin(), sub.cbegin() + n2, std::equal_to<T>());
            }
        }
    }
}

int32_t main( ) {
    test<int32_t>(8*sizeof(int32_t));
    test<uint16_t>(8*sizeof(uint16_t));
    test<float64_t>(53);
#if !__PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN
    test<bool>(1);
#endif

    std::cout << done() << std::endl;
    return 0;
}
