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

#ifndef __PSTL_glue_algorithm_impl_H
#define __PSTL_glue_algorithm_impl_H

#include <functional>

#include "execution_defs.h"
#include "utils.h"
#include "algorithm_impl.h"
#include "numeric_impl.h"  /* count and count_if use pattern_transform_reduce */

namespace std {

// [alg.any_of]

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
any_of(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred) {
    using namespace pstl::internal;
    return pattern_any_of( first, last, pred,
        is_vectorization_preferred<ExecutionPolicy,ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator>(exec));
}

// [alg.all_of]

template<class ExecutionPolicy, class ForwardIterator, class Pred>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
all_of(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Pred pred) {
    return !any_of(std::forward<ExecutionPolicy>(exec), first, last, pstl::internal::not_pred<Pred>(pred));
}

// [alg.none_of]

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
none_of(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred) {
    return !any_of( std::forward<ExecutionPolicy>(exec), first, last, pred );
}

// [alg.foreach]

template<class ExecutionPolicy, class ForwardIterator, class Function>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
for_each(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Function f) {
    using namespace pstl::internal;
    pattern_walk1(
        first, last, f,
        is_vectorization_preferred<ExecutionPolicy,ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class Size, class Function>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
for_each_n(ExecutionPolicy&& exec, ForwardIterator first, Size n, Function f) {
    using namespace pstl::internal;
    return pattern_walk1_n(first, n, f,
        is_vectorization_preferred<ExecutionPolicy,ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator>(exec));
}

// [alg.find]

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
find_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred) {
    using namespace pstl::internal;
    return pattern_find_if( first, last, pred,
        is_vectorization_preferred<ExecutionPolicy,ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
find_if_not(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last,
Predicate pred) {
    return find_if(std::forward<ExecutionPolicy>(exec), first, last, pstl::internal::not_pred<Predicate>(pred));
}

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
find(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last,
const T& value) {
    return find_if(std::forward<ExecutionPolicy>(exec), first, last, pstl::internal::equal_value<T>(value));
}

// [alg.find.end]
template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
find_end(ExecutionPolicy &&exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_find_end(first, last, s_first, s_last, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
find_end(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last) {
    return find_end(std::forward<ExecutionPolicy>(exec), first, last, s_first, s_last, pstl::internal::pstl_equal());
}

// [alg.find_first_of]
template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
find_first_of(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_find_first_of(first, last, s_first, s_last, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
find_first_of(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last) {
    return find_first_of(std::forward<ExecutionPolicy>(exec), first, last, s_first, s_last, pstl::internal::pstl_equal());
}

// [alg.adjacent_find]
template< class ExecutionPolicy, class ForwardIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
adjacent_find(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    using namespace pstl::internal;
    return pattern_adjacent_find(first, last, pstl::internal::pstl_equal(),
       is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec),
       is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec), /*first_semantic*/ false);
}

template< class ExecutionPolicy, class ForwardIterator, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
adjacent_find(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_adjacent_find(first, last, pred,
       is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec),
       is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec), /*first_semantic*/ false);
}

// [alg.count]

// Implementation note: count and count_if call the pattern directly instead of calling std::transform_reduce
// so that we do not have to include <numeric>.

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,typename iterator_traits<ForwardIterator>::difference_type>
count(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value) {
    typedef typename iterator_traits<ForwardIterator>::reference value_type;
    using namespace pstl::internal;
    return pattern_count(first, last, [&value](const value_type x) {return value==x;},
       is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec),
       is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,typename iterator_traits<ForwardIterator>::difference_type>
count_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred) {
    using namespace pstl::internal;
    return pattern_count(first, last, pred,
       is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec),
       is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

// [alg.search]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
search(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_search(first, last, s_first, s_last, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
search(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last) {
    return search(std::forward<ExecutionPolicy>(exec), first, last, s_first, s_last, pstl::internal::pstl_equal());
}

template<class ExecutionPolicy, class ForwardIterator, class Size, class T, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
search_n(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Size count, const T& value, BinaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_search_n(first, last, count, value, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class Size, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
search_n(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Size count, const T& value) {
    return search_n(std::forward<ExecutionPolicy>(exec), first, last, count, value, pstl::internal::pstl_equal());
}

// [alg.copy]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result) {
    using namespace pstl::internal;
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec);

    return pattern_walk2_brick(first, last, result, [is_vector](ForwardIterator1 begin, ForwardIterator1 end, ForwardIterator2 res){
        return brick_copy(begin, end, res, is_vector);
    }, is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class Size, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
copy_n(ExecutionPolicy&& exec, ForwardIterator1 first, Size n, ForwardIterator2 result) {
    using namespace pstl::internal;
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec);

    return pattern_walk2_brick_n(first, n, result, [is_vector](ForwardIterator1 begin, Size sz, ForwardIterator2 res){
        return brick_copy_n(begin, sz, res, is_vector);
    }, is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
copy_if(ExecutionPolicy&& exec,
        ForwardIterator1 first, ForwardIterator1 last,
        ForwardIterator2 result, Predicate pred) {
    using namespace pstl::internal;
    return pattern_copy_if(
        first, last, result, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

// [alg.swap]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
swap_ranges(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2) {
    using namespace pstl::internal;
    typedef typename iterator_traits<ForwardIterator1>::reference reference_type1;
    typedef typename iterator_traits<ForwardIterator2>::reference reference_type2;
    return pattern_walk2(first1, last1, first2,
        [](reference_type1 x, reference_type2 y) {
            using std::swap;
            swap(x, y);
        },
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

// [alg.transform]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class UnaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
transform( ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, UnaryOperation op ) {
    typedef typename iterator_traits<ForwardIterator1>::reference input_type;
    typedef typename iterator_traits<ForwardIterator2>::reference output_type;
    using namespace pstl::internal;
    return pattern_walk2(first, last, result,
        [op](input_type x, output_type y ) mutable { y = op(x);},
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator>
transform( ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator result, BinaryOperation op ) {
    typedef typename iterator_traits<ForwardIterator1>::reference input1_type;
    typedef typename iterator_traits<ForwardIterator2>::reference input2_type;
    typedef typename iterator_traits<ForwardIterator>::reference output_type;
    using namespace pstl::internal;
    return pattern_walk3(first1, last1, first2, result, [op](input1_type x, input2_type y, output_type z) mutable {z = op(x,y);},
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2,ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2,ForwardIterator>(exec));
}

// [alg.replace]

template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
replace_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred, const T& new_value) {
    using namespace pstl::internal;
    typedef typename iterator_traits<ForwardIterator>::reference element_type;
    pattern_walk1(first, last, [&pred, &new_value] (element_type elem) {
            if (pred(elem)) {
                elem = new_value;
            }
        },
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
replace(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& old_value, const T& new_value) {
    replace_if(std::forward<ExecutionPolicy>(exec), first, last, pstl::internal::equal_value<T>(old_value), new_value);
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class UnaryPredicate, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
replace_copy_if(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, UnaryPredicate pred, const T& new_value) {
    typedef typename iterator_traits<ForwardIterator1>::reference input_type;
    typedef typename iterator_traits<ForwardIterator2>::reference output_type;
    using namespace pstl::internal;
    return pattern_walk2(
        first, last, result,
        [pred, &new_value](input_type x, output_type y) mutable { y = pred(x) ? new_value : x; },
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
replace_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, const T& old_value, const T& new_value) {
    return replace_copy_if(std::forward<ExecutionPolicy>(exec), first, last, result, pstl::internal::equal_value<T>(old_value), new_value);
}

// [alg.fill]

template <class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
fill( ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value ) {
    using namespace pstl::internal;
    pattern_fill(first, last, value,
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator>(exec),
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template< class ExecutionPolicy, class ForwardIterator, class Size, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
fill_n( ExecutionPolicy&& exec, ForwardIterator first, Size count, const T& value ) {
    if(count <= 0)
        return first;

    using namespace pstl::internal;
    return pattern_fill_n(first, count, value,
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

// [alg.generate]
template< class ExecutionPolicy, class ForwardIterator, class Generator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
generate( ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Generator g ) {
    using namespace pstl::internal;
    pattern_generate(first, last, g,
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator>(exec),
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template< class ExecutionPolicy, class ForwardIterator, class Size, class Generator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
generate_n( ExecutionPolicy&& exec, ForwardIterator first, Size count, Generator g ) {
    if(count <= 0)
        return first;

    using namespace pstl::internal;
    return pattern_generate_n(first, count, g,
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

// [alg.remove]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
remove_copy_if(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, Predicate pred) {
    return copy_if( std::forward<ExecutionPolicy>(exec), first, last, result, pstl::internal::not_pred<Predicate>(pred));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
remove_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, const T& value) {
    return copy_if( std::forward<ExecutionPolicy>(exec), first, last, result, pstl::internal::not_equal_value<T>(value));
}

template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
remove_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_remove_if(first, last, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
remove(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value) {
    return remove_if(std::forward<ExecutionPolicy>(exec), first, last, pstl::internal::equal_value<T>(value));
}

// [alg.unique]

template<class ExecutionPolicy, class ForwardIterator, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
unique(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_unique(first, last, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
unique(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    return unique(std::forward<ExecutionPolicy>(exec), first, last, pstl::internal::pstl_equal());
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
unique_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_unique_copy(first, last, result, pred,
        is_vectorization_preferred<ExecutionPolicy,ForwardIterator1,ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator1,ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
unique_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result) {
    return unique_copy(std::forward<ExecutionPolicy>(exec), first, last, result, pstl::internal::pstl_equal() );
}

// [alg.reverse]

template<class ExecutionPolicy, class BidirectionalIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
reverse(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator last) {
    using namespace pstl::internal;
    pattern_reverse(first, last,
        is_vectorization_preferred<ExecutionPolicy, BidirectionalIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, BidirectionalIterator>(exec));
}

template<class ExecutionPolicy, class BidirectionalIterator, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
reverse_copy(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator last, ForwardIterator d_first) {
    using namespace pstl::internal;
    return pattern_reverse_copy(first, last, d_first,
        is_vectorization_preferred<ExecutionPolicy, BidirectionalIterator, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, BidirectionalIterator, ForwardIterator>(exec));
}

// [alg.rotate]

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
rotate(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator middle, ForwardIterator last) {
    using namespace pstl::internal;
    return pattern_rotate(first, middle, last,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
rotate_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 middle, ForwardIterator1 last, ForwardIterator2 result) {
    using namespace pstl::internal;
    return pattern_rotate_copy(first, middle, last, result,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

// [alg.partitions]

template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_partitioned(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_is_partitioned(first, last, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
partition(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_partition(first, last, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class BidirectionalIterator, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, BidirectionalIterator>
stable_partition(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator last, UnaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_stable_partition(first, last, pred,
        is_vectorization_preferred<ExecutionPolicy, BidirectionalIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, BidirectionalIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class ForwardIterator1, class ForwardIterator2, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
partition_copy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, ForwardIterator1 out_true, ForwardIterator2 out_false, UnaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_partition_copy(first, last, out_true, out_false, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator, ForwardIterator1, ForwardIterator2>(exec));
}

// [alg.sort]

template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
    typedef typename iterator_traits<RandomAccessIterator>::value_type input_type;
    using namespace pstl::internal;
    return pattern_sort(first, last, comp,
        is_vectorization_preferred<ExecutionPolicy,RandomAccessIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy,RandomAccessIterator>(exec),
        typename std::is_move_constructible<input_type>::type());
}

template<class ExecutionPolicy, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last) {
    typedef typename iterator_traits<RandomAccessIterator>::value_type input_type;
    sort(std::forward<ExecutionPolicy>(exec), first, last, std::less<input_type>());
}

// [stable.sort]

template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
stable_sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
    using namespace pstl::internal;
    return pattern_stable_sort(first, last, comp,
        is_vectorization_preferred<ExecutionPolicy,RandomAccessIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy,RandomAccessIterator>(exec));
}

template<class ExecutionPolicy, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
stable_sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last) {
    typedef typename iterator_traits<RandomAccessIterator>::value_type input_type;
    stable_sort(std::forward<ExecutionPolicy>(exec), first, last, std::less<input_type>());
}

// [mismatch]

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
mismatch(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, BinaryPredicate pred) {
    using namespace pstl::internal;
    return pattern_mismatch(first1, last1, first2, last2, pred,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
mismatch(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, BinaryPredicate pred) {
    return mismatch(std::forward<ExecutionPolicy>(exec), first1, last1, first2, std::next(first2, std::distance(first1, last1)), pred);
}

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2 >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
mismatch(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2) {
    return mismatch(std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, pstl::internal::pstl_equal());
}

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2 >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
mismatch(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2) {
    return mismatch(std::forward<ExecutionPolicy>(exec), first1, last1, first2, std::next(first2, std::distance(first1, last1)));
}

// [alg.equal]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
equal(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, BinaryPredicate p) {
    using namespace pstl::internal;
    return pattern_equal(first1, last1, first2, p,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1>(exec)
        );
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
equal(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2) {
    return equal(std::forward<ExecutionPolicy>(exec), first1, last1, first2, pstl::internal::pstl_equal());
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
equal(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, BinaryPredicate p) {
    if ( std::distance(first1, last1) == std::distance(first2, last2) )
        return std::equal(std::forward<ExecutionPolicy>(exec), first1, last1, first2, p);
    else
        return false;
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
equal(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2) {
    return equal(std::forward<ExecutionPolicy>(exec), first1, last1, first2, pstl::internal::pstl_equal());
}

// [alg.move]
template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2 >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
move(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first) {
    using namespace pstl::internal;
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec);

    return pattern_walk2_brick(first, last, d_first, [is_vector](ForwardIterator1 begin, ForwardIterator1 end, ForwardIterator2 res) {
        return brick_move(begin, end, res, is_vector);
    }, is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

// [partial.sort]

template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
partial_sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare comp) {
    using namespace pstl::internal;
    pattern_partial_sort(first, middle, last, comp,
        is_vectorization_preferred<ExecutionPolicy, RandomAccessIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, RandomAccessIterator>(exec));
}

template<class ExecutionPolicy, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
partial_sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last) {
    typedef typename iterator_traits<RandomAccessIterator>::value_type input_type;
    partial_sort(exec, first, middle, last, std::less<input_type>());
}

// [partial.sort.copy]

template<class ExecutionPolicy, class ForwardIterator, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, RandomAccessIterator>
partial_sort_copy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, RandomAccessIterator d_first, RandomAccessIterator d_last, Compare comp) {
    using namespace pstl::internal;
    return pattern_partial_sort_copy(first, last, d_first, d_last, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator, RandomAccessIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator, RandomAccessIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, RandomAccessIterator>
partial_sort_copy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, RandomAccessIterator d_first, RandomAccessIterator d_last) {
    return partial_sort_copy(std::forward<ExecutionPolicy>(exec), first, last, d_first, d_last, pstl::internal::pstl_less());
}

// [is.sorted]
template<class ExecutionPolicy, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
is_sorted_until(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp) {
    using namespace pstl::internal;
    const ForwardIterator res = pattern_adjacent_find(first, last, pstl::internal::reorder_pred<Compare>(comp),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec), /*first_semantic*/ false);
    return res==last ? last : std::next(res);
}

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
is_sorted_until(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type input_type;
    return is_sorted_until(exec, first, last, std::less<input_type>());
}

template<class ExecutionPolicy, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_sorted(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp) {
    using namespace pstl::internal;
    return pattern_adjacent_find(first, last, reorder_pred<Compare>(comp),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec), /*or_semantic*/ true)==last;
}

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_sorted(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type input_type;
    return is_sorted(exec, first, last, std::less<input_type>());
}

// [alg.nth.element]

template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
nth_element(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last, Compare comp) {
    using namespace pstl::internal;
    pattern_nth_element(first, nth, last, comp,
        is_vectorization_preferred<ExecutionPolicy, RandomAccessIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, RandomAccessIterator>(exec));
}

template<class ExecutionPolicy, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
nth_element(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last) {
    typedef typename iterator_traits<RandomAccessIterator>::value_type input_type;
    nth_element(exec, first, nth, last, std::less<input_type>());
}

// [alg.merge]
template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
merge(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator d_first, Compare comp) {
    using namespace pstl::internal;
    return pattern_merge(first1, last1, first2, last2, d_first, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec));
}

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
merge(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator d_first) {
    return merge(std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, d_first, pstl::internal::pstl_less());
}

template< class ExecutionPolicy, class BidirectionalIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
inplace_merge(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, Compare comp) {
    using namespace pstl::internal;
    pattern_inplace_merge(first, middle, last, comp,
        is_vectorization_preferred<ExecutionPolicy, BidirectionalIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, BidirectionalIterator>(exec));
}

template< class ExecutionPolicy, class BidirectionalIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
inplace_merge(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last) {
    typedef typename iterator_traits<BidirectionalIterator>::value_type input_type;
    inplace_merge(exec, first, middle, last, std::less<input_type>());
}

// [includes]

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
includes(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, Compare comp) {
    using namespace pstl::internal;
    return pattern_includes(first1, last1, first2, last2, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
includes(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2) {
    return includes(std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, pstl::internal::pstl_less());
}

// [set.union]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_union(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator result, Compare comp) {
    using namespace pstl::internal;
    return pattern_set_union(first1, last1, first2, last2, result, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_union(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2,
    ForwardIterator2 last2, ForwardIterator result) {
    return set_union(std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, result, pstl::internal::pstl_less());
}

// [set.intersection]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_intersection(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator result, Compare comp) {
    using namespace pstl::internal;
    return pattern_set_intersection(first1, last1, first2, last2, result, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_intersection(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator result) {
    return set_intersection(std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, result, pstl::internal::pstl_less());
}

// [set.difference]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_difference(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator result, Compare comp) {
    using namespace pstl::internal;
    return pattern_set_difference(first1, last1, first2, last2, result, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_difference(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator result) {
    return set_difference(std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, result, pstl::internal::pstl_less());
}

// [set.symmetric.difference]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_symmetric_difference(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator result, Compare comp) {
    using namespace pstl::internal;
    return pattern_set_symmetric_difference(first1, last1, first2, last2, result, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2, ForwardIterator>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_symmetric_difference(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, ForwardIterator result) {
    return set_symmetric_difference(std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, result, pstl::internal::pstl_less());
}

// [is.heap]
template< class ExecutionPolicy, class RandomAccessIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, RandomAccessIterator>
is_heap_until(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
    using namespace pstl::internal;
    return pattern_is_heap_until(first, last, comp,
        is_vectorization_preferred<ExecutionPolicy, RandomAccessIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, RandomAccessIterator>(exec));
}

template< class ExecutionPolicy, class RandomAccessIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, RandomAccessIterator>
is_heap_until(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last) {
    typedef typename iterator_traits<RandomAccessIterator>::value_type input_type;
    return is_heap_until(std::forward<ExecutionPolicy>(exec), first, last, std::less<input_type>());
}

template< class ExecutionPolicy, class RandomAccessIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_heap(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
    return is_heap_until(std::forward<ExecutionPolicy>(exec), first, last, comp) == last;
}

template< class ExecutionPolicy, class RandomAccessIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_heap(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last) {
    typedef typename iterator_traits<RandomAccessIterator>::value_type input_type;
    return is_heap(std::forward<ExecutionPolicy>(exec), first, last, std::less<input_type>());
}

// [alg.min.max]

template< class ExecutionPolicy, class ForwardIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
min_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp) {
    using namespace pstl::internal;
    return pattern_min_element(first, last, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template< class ExecutionPolicy, class ForwardIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
min_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type input_type;
    return min_element(std::forward<ExecutionPolicy>(exec), first, last, std::less<input_type>());
}

template< class ExecutionPolicy, class ForwardIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
max_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp) {
    using namespace pstl::internal;
    return pattern_min_element(first, last, pstl::internal::reorder_pred<Compare>(comp),
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template< class ExecutionPolicy, class ForwardIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
max_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type input_type;
    return min_element(std::forward<ExecutionPolicy>(exec), first, last, pstl::internal::reorder_pred<std::less<input_type> >(std::less<input_type>()));
}

template< class ExecutionPolicy, class ForwardIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator, ForwardIterator>>
minmax_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp) {
    using namespace pstl::internal;
    return pattern_minmax_element(first, last, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec));
}

template< class ExecutionPolicy, class ForwardIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator, ForwardIterator>>
minmax_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    return minmax_element(std::forward<ExecutionPolicy>(exec), first, last, std::less<value_type>());
}

// [alg.lex.comparison]

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
lexicographical_compare(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, Compare comp) {
    using namespace pstl::internal;
    return pattern_lexicographical_compare(first1, last1, first2, last2, comp,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2 >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
lexicographical_compare(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2) {
    return lexicographical_compare(std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, pstl::internal::pstl_less());
}

} // namespace std

#endif /* __PSTL_glue_algorithm_impl_H */
