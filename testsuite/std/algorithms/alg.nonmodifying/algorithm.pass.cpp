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

// Tests for the rest algorithms; temporary approach to check compiling
// TODO break into individual test suites

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

auto is_even = [](double v) { unsigned int i = (unsigned int)v;  return i % 2 == 0; };

template<typename Policy, typename F>
static void invoke_if(Policy&& p, F f) {
    #if __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN
        pstl::internal::invoke_if_not(pstl::internal::allow_unsequenced<Policy>(), f);
    #else
        f();
    #endif
}

//Testing with random-access iterator only
struct run_rnd {
    template <typename Policy, typename Iterator>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value, void>::type
        operator()(Policy&& exec, Iterator b1, Iterator e1, Iterator b2, Iterator e2) {
        typedef typename std::iterator_traits<Iterator>::value_type T;
        using namespace std;

        //usage of "non_const" adapter - we pass empty container due to just compilation checks

        invoke_if(exec, [&]() {
            //is_heap
            is_heap(exec, b1, e1);
            is_heap(exec, b1, e1, std::less<T>());
            is_heap(exec, b1, b1, non_const(std::less<T>()));

            //is_heap_until
            is_heap_until(exec, b1, e1);
            is_heap_until(exec, b1, e1, std::less<T>());
            is_heap_until(exec, b1, b1, non_const(std::less<T>()));
        });

        //nth_element
        auto middle = b1 + (e1 - b1) / 2;
        nth_element(exec, b1, middle, e1);
        nth_element(exec, b1, middle, e1, std::less<T>());
        nth_element(exec, b1, b1, b1, non_const(std::less<T>()));

        //partial_sort
        partial_sort(exec, b1, middle, e1);
        partial_sort(exec, b1, middle, e1, std::less<T>());
        partial_sort(exec, b1, b1, b1, non_const(std::less<T>()));

        //partial_sort_copy
        partial_sort_copy(exec, b1, e1, b2, e2);
        partial_sort_copy(exec, b1, e1, b2, e2, std::less<T>());
        partial_sort_copy(exec, b1, b1, b2, b2, non_const(std::less<T>()));

        //sort
        sort(exec, b1, e1);
        sort(exec, b1, e1, std::less<T>());
        sort(exec, b1, b1, non_const(std::less<T>()));

        //stable_sort
        stable_sort(exec, b1, e1);
        stable_sort(exec, b1, e1, std::less<T>());
        stable_sort(exec, b1, b1, non_const(std::less<T>()));
    }

    template <typename Policy, typename Iterator>
    typename std::enable_if<!is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value, void>::type
        operator()(Policy&& exec, Iterator b1, Iterator e1, Iterator b2, Iterator e2) { }

};

//Testing with random-access iterator and bidirectional iterator
struct run_rnd_bi {
    template <typename Policy, typename Iterator>
    typename std::enable_if<!is_same_iterator_category<Iterator, std::forward_iterator_tag>::value, void>::type
        operator()(Policy&& exec, Iterator b1, Iterator e1, Iterator b2, Iterator e2) {
        typedef typename std::iterator_traits<Iterator>::value_type T;
        using namespace std;

        //usage of "non_const" adapter - we pass empty container due to just compilation checks

        //inplace_merge
        auto middle = next(b1, distance(b1, e1) / 2);
        inplace_merge(exec, b1, middle, e1);
        inplace_merge(exec, b1, middle, e1, std::less<T>());
        inplace_merge(exec, b1, b1, b1, non_const(std::less<T>()));

        //reverse
        reverse(exec, b2, e2);

        //reverse_copy
        reverse_copy(exec, b1, e1, b2);

        //stable_partition
        stable_partition(exec, b1, e1, is_even);
        stable_partition(exec, b1, b1, non_const(is_even));
    }

    template <typename Policy, typename Iterator>
    typename std::enable_if<is_same_iterator_category<Iterator, std::forward_iterator_tag>::value, void>::type
        operator()(Policy&& exec, Iterator b1, Iterator e1, Iterator b2, Iterator e2) {}
};

//Testing with random-access iterator and forward iterator
struct run_rnd_fw {
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator b1, Iterator e1, Iterator b2, Iterator e2) {
        typedef typename std::iterator_traits<Iterator>::value_type T;
        using namespace std;

        Iterator out = b1;
        auto n = distance(b1, e1);
        Iterator cmiddle = b1;
        std::advance(cmiddle, n/2);
        //usage of "non_const" adapter - we pass empty container due to just compilation checks

        //all_of
        all_of(exec, b1, e1, is_even);
        all_of(exec, b1, b1, non_const(is_even));

        //adjacent_find
        adjacent_find(exec, b1, e1);
        adjacent_find(exec, b1, e1, std::equal_to<T>());
        adjacent_find(exec, b1, b1, non_const(std::equal_to<T>()));

        //any_of
        any_of(exec, b1, e1, is_even);
        any_of(exec, b1, b1, non_const(is_even));

        invoke_if(exec, [&]() {
            //copy
            copy(exec, b1, e1, out);
            copy_n(exec, b1, n, out);
        });

        //copy_if
        copy_if(exec, b1, e1, out, is_even);
        copy_if(exec, b1, b1, out, non_const(is_even));

        invoke_if(exec, [&]() {
            //count
            count(exec, b1, e1, T(0));
        });

        //count_if
        count_if(exec, b1, e1, is_even);
        count_if(exec, b1, b1, non_const(is_even));

        //equal
        equal(exec, b1, e1, b2);
        equal(exec, b1, e1, b2, std::equal_to<T>());
        equal(exec, b1, e1, b2, e2);
        equal(exec, b1, e1, b2, e2, std::equal_to<T>());
        equal(exec, b1, b1, b2, b2, non_const(std::equal_to<T>()));

        //fill
        fill(exec, b1, e1, T(0));

        //fill_n
        fill_n(exec, b1, n, T(0));

        invoke_if(exec, [&]() {
            //find
            find(exec, b1, e1, T(0));

            //find_end
            find_end(exec, b1, e1, b2, e2);
            find_end(exec, b1, e1, b2, e2, std::equal_to<T>());
            find_end(exec, b1, b1, b2, b2, non_const(std::equal_to<T>()));

            //find_first_of
            find_first_of(exec, b1, e1, b2, e2);
            find_first_of(exec, b1, e1, b2, e2, std::equal_to<T>());
            find_first_of(exec, b1, b1, b2, b2, non_const(std::equal_to<T>()));

            //find_if
            find_if(exec, b1, e1, is_even);
            find_if(exec, b1, b1, non_const(is_even));

            //find_if_not
            find_if_not(exec, b1, e1, is_even);
            find_if_not(exec, b1, b1, non_const(is_even));

            //for_each
            auto f = [](typename iterator_traits<Iterator>::reference x) { x = x+1; };
            for_each(exec, b1, e1, f);
            for_each(exec, b1, b1, non_const(f));

            //for_each_n
            for_each_n(exec, b1, n, f);
            for_each_n(exec, b1, 0, non_const(f));
        });

        //generate
        auto gen = [](){return T(0);};
        generate(exec, b1, e1, gen);
        generate(exec, b1, b1, non_const(gen));

        //generate_n
        generate_n(exec, b1, n, gen);
        generate_n(exec, b1, 0, non_const(gen));

        //includes
        includes(exec, b1, e1, b2, e2);
        includes(exec, b1, e1, b2, e2, std::less<T>());
        includes(exec, b1, b1, b2, b2, non_const(std::less<T>()));

        //is_partitioned
        is_partitioned(exec, b1, e1, is_even);
        is_partitioned(exec, b1, b1, non_const(is_even));

        //is_sorted
        is_sorted(exec, b1, e1);
        is_sorted(exec, b1, e1, std::less<T>());
        is_sorted(exec, b1, b1, non_const(std::less<T>()));

        //is_sorted_until
        is_sorted_until(exec, b1, e1);
        is_sorted_until(exec, b1, e1, std::less<T>());
        is_sorted_until(exec, b1, b1, non_const(std::less<T>()));

        //lexicographical_compare
        lexicographical_compare(exec, b1, e1, b2, e2);
        lexicographical_compare(exec, b1, e1, b2, e2, std::less<T>());
        lexicographical_compare(exec, b1, b1, b2, b2, non_const(std::less<T>()));

        //max_element
        max_element(exec, b1, e1);
        max_element(exec, b1, e1, std::less<T>());
        max_element(exec, b1, b1, non_const(std::less<T>()));

        //merge
        merge(exec, b1, cmiddle, cmiddle, e1, out);
        merge(exec, b1, cmiddle, cmiddle, e1, out, std::less<T>());
        merge(exec, b1, b1, b1, b1, out, non_const(std::less<T>()));

        //mismatch
        mismatch(exec, b1, e1, b2);
        mismatch(exec, b1, e1, b2, std::equal_to<T>());
        mismatch(exec, b1, e1, b2, e2);
        mismatch(exec, b1, e1, b2, e2, std::equal_to<T>());
        mismatch(exec, b1, b1, b2, b2, non_const(std::equal_to<T>()));

        //min_element
        min_element(exec, b1, e1);
        min_element(exec, b1, e1, std::less<T>());
        min_element(exec, b1, b1, non_const(std::less<T>()));

        //minmax_element
        minmax_element(exec, b1, e1);
        minmax_element(exec, b1, e1, std::less<T>());
        minmax_element(exec, b1, b1, non_const(std::less<T>()));

        invoke_if(exec, [&]() {
            //move
            move(exec, b1, e1, out);
        });

        //none_of
        none_of(exec, b1, e1, is_even);
        none_of(exec, b1, b1, non_const(is_even));

        //partition
        partition(exec, b1, e1, is_even);
        partition(exec, b1, b1, non_const(is_even));

        //partition_copy
        partition_copy(exec, b1, e1, out, out, is_even);
        partition_copy(exec, b1, b1, out, out, non_const(is_even));
        invoke_if(exec, [&]() {
            //remove
            remove(exec, b1, e1, T(0));

            //remove_copy
            remove_copy(exec, b1, e1, out, T(0));

            //remove_copy_if
            remove_copy_if(exec, b1, e1, out, is_even);
            remove_copy_if(exec, b1, b1, out, non_const(is_even));

            //remove_if
            remove_if(exec, b1, e1, is_even);
            remove_if(exec, b1, b1, non_const(is_even));

            //replace
            replace(exec, b1, e1, T(1), T(0));

            //replace_copy
            replace_copy(exec, b1, e1, out, T(1), T(0));

            //replace_copy_if
            replace_copy_if(exec, b1, e1, out, is_even, T(0));
            replace_copy_if(exec, b1, b1, out, non_const(is_even), T(0));

            //replace_if
            replace_if(exec, b1, e1, is_even, T(0));
            replace_if(exec, b1, b1, non_const(is_even), T(0));
        });

        //rotate
        rotate(exec, b1, b1, e1);

        //rotate_copy
        rotate_copy(exec, b1, b1, e1, out);

        invoke_if(exec, [&]() {
            //search
            search(exec, b1, e1, b2, e2);
            search(exec, b1, e1, b2, e2, std::equal_to<T>());
            search(exec, b1, b1, b2, b2, non_const(std::equal_to<T>()));

            //search_n
            search_n(exec, b1, e1, 2, T(0));
            search_n(exec, b1, e1, 2, T(0), std::equal_to<T>());
            search_n(exec, b1, b1, 0, T(0), non_const(std::equal_to<T>()));
        });

        //set_difference
        set_difference(exec, b1, cmiddle, cmiddle, e1, out);
        set_difference(exec, b1, cmiddle, cmiddle, e1, out, std::less<T>());
        set_difference(exec, b1, b1, b1, b1, out, non_const(std::less<T>()));

        //set_intersection
        set_intersection(exec, b1, cmiddle, cmiddle, e1, out);
        set_intersection(exec, b1, cmiddle, cmiddle, e1, out, std::less<T>());
        set_intersection(exec, b1, b1, b1, b1, out, non_const(std::less<T>()));

        //set_symmetric_difference
        set_symmetric_difference(exec, b1, cmiddle, cmiddle, e1, out);
        set_symmetric_difference(exec, b1, cmiddle, cmiddle, e1, out, std::less<T>());
        set_symmetric_difference(exec, b1, b1, b1, b1, out, non_const(std::less<T>()));

        //set_union
        set_union(exec, b1, cmiddle, cmiddle, e1, out);
        set_union(exec, b1, cmiddle, cmiddle, e1, out, std::less<T>());
        set_union(exec, b1, b1, b1, b1, out, non_const(std::less<T>()));

        //swap_ranges
        swap_ranges(exec, b1, e1, b2);

        invoke_if(exec, [&]() {
            //transform
            transform(exec, b1, e1, out, std::negate<T>());
            transform(exec, b1, b1, out, non_const(std::negate<T>()));

            transform(exec, b1, e1, b2, out, std::plus<T>());
            transform(exec, b1, b1, b2, out, non_const(std::plus<T>()));
        });

        //unique
        unique(exec, b1, e1);
        unique(exec, b1, e1, std::equal_to<T>());
        unique(exec, b1, b1, non_const(std::equal_to<T>()));

        //unique_copy
        unique_copy(exec, b1, e1, out);
        unique_copy(exec, b1, e1, out, std::equal_to<T>());
        unique_copy(exec, b1, b1, out, non_const(std::equal_to<T>()));
    }
};

template <typename T>
void test_algo_by_type() {
    size_t N = 10000;
    for (size_t n = 0; n < N; n = n < 10 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<T> in(n, [](size_t v)->T { return T(v); }); //fill 0..n
        Sequence<T> out(n, [](size_t v)->T { return T(v); }); //fill 0..n

        invoke_on_all_policies(run_rnd_fw(), in.begin(), in.end(), out.begin(), out.end());
        invoke_on_all_policies(run_rnd(), in.begin(), in.end(), out.begin(), out.end());
        invoke_on_all_policies(run_rnd_bi(), in.begin(), in.end(), out.begin(), out.end());
    }
}

int32_t main() {

    test_algo_by_type<int32_t>();
#if !__PSTL_CLANG_TEST_BIG_OBJ_DEBUG_32_BROKEN
    test_algo_by_type<float64_t>();
    test_algo_by_type<bool>();
#endif

    std::cout<<done()<<std::endl;
    return 0;
}
