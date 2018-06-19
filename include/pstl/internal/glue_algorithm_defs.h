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

#ifndef __PSTL_glue_algorithm_defs_H
#define __PSTL_glue_algorithm_defs_H

#include <functional>

#include "execution_defs.h"

namespace std {

// [alg.any_of]

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
any_of(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

// [alg.all_of]

template<class ExecutionPolicy, class ForwardIterator, class Pred>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
all_of(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Pred pred);

// [alg.none_of]

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
none_of(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

// [alg.foreach]

template<class ExecutionPolicy, class ForwardIterator, class Function>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
for_each(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Function f);

template<class ExecutionPolicy, class ForwardIterator, class Size, class Function>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
for_each_n(ExecutionPolicy&& exec, ForwardIterator first, Size n, Function f);

// [alg.find]

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
find_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
find_if_not(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
find(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value);

// [alg.find.end]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
find_end(ExecutionPolicy &&exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last,
         BinaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
find_end(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last);

// [alg.find_first_of]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
find_first_of(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last,
              BinaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
find_first_of(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last);

// [alg.adjacent_find]

template< class ExecutionPolicy, class ForwardIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
adjacent_find(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template< class ExecutionPolicy, class ForwardIterator, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
adjacent_find(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate pred);

// [alg.count]

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,typename iterator_traits<ForwardIterator>::difference_type>
count(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value);

template<class ExecutionPolicy, class ForwardIterator, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,typename iterator_traits<ForwardIterator>::difference_type>
count_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

// [alg.search]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
search(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last,
       BinaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator1>
search(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last);

template<class ExecutionPolicy, class ForwardIterator, class Size, class T, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
search_n(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Size count, const T& value, BinaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator, class Size, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
search_n(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Size count, const T& value);

// [alg.copy]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result);

template<class ExecutionPolicy, class ForwardIterator1, class Size, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
copy_n(ExecutionPolicy&& exec, ForwardIterator1 first, Size n, ForwardIterator2 result);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
copy_if(ExecutionPolicy&& exec,
        ForwardIterator1 first, ForwardIterator1 last,
        ForwardIterator2 result, Predicate pred);

// [alg.swap]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
swap_ranges(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2);

// [alg.transform]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class UnaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
transform( ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, UnaryOperation op );

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator>
transform(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator result,
          BinaryOperation op);

// [alg.replace]

template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
replace_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred, const T& new_value);

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
replace(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& old_value, const T& new_value);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class UnaryPredicate, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
replace_copy_if(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, UnaryPredicate pred,
                const T& new_value);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
replace_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, const T& old_value,
             const T& new_value);

// [alg.fill]

template <class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
fill( ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value );

template< class ExecutionPolicy, class ForwardIterator, class Size, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
fill_n( ExecutionPolicy&& exec, ForwardIterator first, Size count, const T& value );

// [alg.generate]
template< class ExecutionPolicy, class ForwardIterator, class Generator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
generate( ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Generator g );

template< class ExecutionPolicy, class ForwardIterator, class Size, class Generator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
generate_n( ExecutionPolicy&& exec, ForwardIterator first, Size count, Generator g );

// [alg.remove]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Predicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
remove_copy_if(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, Predicate pred);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
remove_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, const T& value);

template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
remove_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
remove(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value);

// [alg.unique]

template<class ExecutionPolicy, class ForwardIterator, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
unique(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
unique(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
unique_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
unique_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result);

// [alg.reverse]

template<class ExecutionPolicy, class BidirectionalIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
reverse(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator last);

template<class ExecutionPolicy, class BidirectionalIterator, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
reverse_copy(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator last, ForwardIterator d_first);

// [alg.rotate]

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
rotate(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator middle, ForwardIterator last);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
rotate_copy(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 middle, ForwardIterator1 last, ForwardIterator2 result);

// [alg.partitions]

template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_partitioned(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
partition(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred);

template<class ExecutionPolicy, class BidirectionalIterator, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, BidirectionalIterator>
stable_partition(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator last, UnaryPredicate pred);

template<class ExecutionPolicy, class ForwardIterator, class ForwardIterator1, class ForwardIterator2, class UnaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
partition_copy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, ForwardIterator1 out_true, ForwardIterator2 out_false,
               UnaryPredicate pred);

// [alg.sort]

template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp);

template<class ExecutionPolicy, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last);

// [stable.sort]

template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
stable_sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp);

template<class ExecutionPolicy, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
stable_sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last);

// [mismatch]

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
mismatch(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
         BinaryPredicate pred);

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
mismatch(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, BinaryPredicate pred);

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2 >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
mismatch(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2);

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2 >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator1, ForwardIterator2>>
mismatch(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2);

// [alg.equal]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
equal(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, BinaryPredicate p);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
equal(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
equal(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, BinaryPredicate p);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
equal(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2);

// [alg.move]
template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2 >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
move(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first);

// [partial.sort]

template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
partial_sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare comp);

template<class ExecutionPolicy, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
partial_sort(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last);

// [partial.sort.copy]

template<class ExecutionPolicy, class ForwardIterator, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, RandomAccessIterator>
partial_sort_copy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, RandomAccessIterator d_first, RandomAccessIterator d_last,
                  Compare comp);

template<class ExecutionPolicy, class ForwardIterator, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, RandomAccessIterator>
partial_sort_copy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, RandomAccessIterator d_first, RandomAccessIterator d_last);

// [is.sorted]
template<class ExecutionPolicy, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
is_sorted_until(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp);

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
is_sorted_until(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template<class ExecutionPolicy, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_sorted(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp);

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_sorted(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

// [alg.nth.element]

template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
nth_element(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last, Compare comp);

template<class ExecutionPolicy, class RandomAccessIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
nth_element(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last);

// [alg.merge]
template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
merge(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
      ForwardIterator d_first, Compare comp);

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
merge(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
      ForwardIterator d_first);

template< class ExecutionPolicy, class BidirectionalIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
inplace_merge(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, Compare comp);

template< class ExecutionPolicy, class BidirectionalIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
inplace_merge(ExecutionPolicy&& exec, BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last);

// [includes]

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
includes(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, Compare comp);

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
includes(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2);

// [set.union]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_union(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
          ForwardIterator result, Compare comp);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_union(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2,
          ForwardIterator2 last2, ForwardIterator result);

// [set.intersection]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_intersection(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
                 ForwardIterator result, Compare comp);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_intersection(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
                 ForwardIterator result);

// [set.difference]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_difference(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
               ForwardIterator result, Compare comp);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_difference(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
               ForwardIterator result);

// [set.symmetric.difference]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator, class Compare>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_symmetric_difference(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
                         ForwardIterator result, Compare comp);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
set_symmetric_difference(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
                         ForwardIterator result);

// [is.heap]
template< class ExecutionPolicy, class RandomAccessIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, RandomAccessIterator>
is_heap_until(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp);

template< class ExecutionPolicy, class RandomAccessIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, RandomAccessIterator>
is_heap_until(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last);

template< class ExecutionPolicy, class RandomAccessIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_heap(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp);

template< class ExecutionPolicy, class RandomAccessIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
is_heap(ExecutionPolicy&& exec, RandomAccessIterator first, RandomAccessIterator last);

// [alg.min.max]

template< class ExecutionPolicy, class ForwardIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
min_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp);

template< class ExecutionPolicy, class ForwardIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
min_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template< class ExecutionPolicy, class ForwardIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
max_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp);

template< class ExecutionPolicy, class ForwardIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
max_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template< class ExecutionPolicy, class ForwardIterator, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator, ForwardIterator>>
minmax_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, Compare comp);

template< class ExecutionPolicy, class ForwardIterator >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, std::pair<ForwardIterator, ForwardIterator>>
minmax_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

// [alg.lex.comparison]

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Compare >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
lexicographical_compare(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
                        Compare comp);

template< class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2 >
pstl::internal::enable_if_execution_policy<ExecutionPolicy, bool>
lexicographical_compare(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2);

} // namespace std

#endif /* __PSTL_glue_algorithm_defs_H */
