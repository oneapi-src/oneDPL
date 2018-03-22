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

// Tests for the memory algorithms; approach to check compiling

#include "test/pstl_test_config.h"

#include "pstl/execution"
#include "pstl/memory"

#include "test/utils.h"

using namespace TestUtils;

struct test_memory {

#if __PSTL_TEST_PAR_TBB_RT_ICC_16_VC14_RELEASE_64_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename InputIterator, typename OutputIterator>
    void operator()(pstl::execution::parallel_policy, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin, OutputIterator outEnd) {}
    template <typename InputIterator, typename OutputIterator>
    void operator()(pstl::execution::parallel_unsequenced_policy, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin, OutputIterator outEnd) {}
#endif

#if __PSTL_TEST_SIMD_LAMBDA_ICC_17_VC141_DEBUG_32_BROKEN || __PSTL_TEST_SIMD_LAMBDA_ICC_16_VC14_DEBUG_32_BROKEN//dummy specialization by policy type, in case of broken configuration
    template <typename InputIterator, typename OutputIterator>
    void operator()(pstl::execution::unsequenced_policy, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin, OutputIterator outEnd) {}
    template <typename InputIterator, typename OutputIterator>
    void operator()(pstl::execution::parallel_unsequenced_policy, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin, OutputIterator outEnd) {}
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator>
    void operator()(Policy&& exec, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin, OutputIterator outEnd) {
        typedef typename std::iterator_traits<OutputIterator>::value_type T;
        using namespace std;
        const auto n = std::distance(outBegin, outEnd);

        //uninitialized_copy
        uninitialized_copy(exec, inBegin, inEnd, outBegin);
        //destroy (for cleaning memory, for uninitializing memory)
        destroy(exec, outBegin, outEnd);

        //uninitialized_move
        uninitialized_move(exec, inBegin, inEnd, outBegin);
        destroy(exec, outBegin, outEnd);

        //uninitialized_default_construct
        uninitialized_default_construct(exec, outBegin, outEnd);
        destroy(exec, outBegin, outEnd);

        //uninitialized_value_construct
        uninitialized_value_construct(exec, outBegin, outEnd);
        destroy(exec, outBegin, outEnd);

        //uninitialized_fill
        uninitialized_fill(exec, outBegin, outEnd, T());
        destroy(exec, outBegin, outEnd);

        //uninitialized_copy_n
        uninitialized_copy_n(exec, inBegin, n, outBegin);
        destroy_n(exec, outBegin, n);

        //uninitialized_move_n
        uninitialized_move_n(exec, inBegin, n, outBegin);
        destroy_n(exec, outBegin, n);

        //uninitialized_default_construct_n
        uninitialized_default_construct_n(exec, outBegin, n);
        destroy_n(exec, outBegin, n);

        //uninitialized_value_construct_n
        uninitialized_value_construct_n(exec, outBegin, n);
        destroy_n(exec, outBegin, n);

        //uninitialized_fill_n
        uninitialized_fill_n(exec, outBegin, n, T());
        destroy_n(exec, outBegin, n);
    }
};

template <typename T>
void test_mem_by_type() {
    size_t N = 10000;
    Sequence<T> in(N);

    for (size_t n = 0; n < N; n = n < 16 ? n + 1 : size_t(3.1415 * n)) {
        std::unique_ptr<T[]> p(new T[n]);

        auto end = in.begin();
        std::advance(end, n);
        invoke_on_all_policies(test_memory(), in.begin(), end, p.get(), p.get() + n);
    }
}

int32_t main() {

    test_mem_by_type<int32_t>();
    test_mem_by_type<float64_t>();

    std::cout<<done()<<std::endl;
    return 0;
}

