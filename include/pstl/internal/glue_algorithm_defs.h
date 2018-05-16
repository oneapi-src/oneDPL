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

#ifndef __PSTL_glue_algorithm_defs_H_
#define __PSTL_glue_algorithm_defs_H_

#include <functional>

#include "execution_defs.h"


namespace std {

// [alg.any_of]

template<class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
any_of(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred);

// [alg.all_of]

template<class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
all_of(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred);

// [alg.none_of]

template<class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
none_of(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred);

// [alg.foreach]

template<class _ExecutionPolicy, class _ForwardIterator, class _Function>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
for_each(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f);

template<class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Function>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
for_each_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, _Function __f);

// [alg.find]

template<class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
find_if(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred);

template<class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
find_if_not(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred);

template<class _ExecutionPolicy, class _ForwardIterator, class _Tp>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
find(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value);

// [alg.find.end]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
find_end(_ExecutionPolicy &&__exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last,
         _BinaryPredicate __pred);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
find_end(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last);

// [alg.find_first_of]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
find_first_of(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last,
              _BinaryPredicate __pred);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
find_first_of(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last);

// [alg.adjacent_find]

template< class _ExecutionPolicy, class _ForwardIterator >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
adjacent_find(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template< class _ExecutionPolicy, class _ForwardIterator, class _BinaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
adjacent_find(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred);

// [alg.count]

template<class _ExecutionPolicy, class _ForwardIterator, class _Tp>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy,typename iterator_traits<_ForwardIterator>::difference_type>
count(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value);

template<class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy,typename iterator_traits<_ForwardIterator>::difference_type>
count_if(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred);

// [alg.search]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
search(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last,
       _BinaryPredicate __pred);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
search(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last);

template<class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
search_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value, _BinaryPredicate __pred);

template<class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
search_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value);

// [alg.copy]

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator2>
copy(_executionpolicy&& __exec, _forwarditerator1 __first, _forwarditerator1 __last, _forwarditerator2 __result);

template<class _executionpolicy, class _forwarditerator1, class _size, class _forwarditerator2>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator2>
copy_n(_executionpolicy&& __exec, _forwarditerator1 __first, _size __n, _forwarditerator2 __result);

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2, class _predicate>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator2>
copy_if(_executionpolicy&& __exec,
        _forwarditerator1 __first, _forwarditerator1 __last,
        _forwarditerator2 result, _predicate __pred);

// [alg.swap]

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator2>
swap_ranges(_executionpolicy&& __exec, _forwarditerator1 __first1, _forwarditerator1 __last1, _forwarditerator2 __first2);

// [alg.transform]

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2, class _unaryoperation>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator2>
transform( _executionpolicy&& __exec, _forwarditerator1 __first, _forwarditerator1 __last, _forwarditerator2 __result, _unaryoperation __op );

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2, class _forwarditerator, class _binaryoperation>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator>
transform(_executionpolicy&& __exec, _forwarditerator1 __first1, _forwarditerator1 __last1, _forwarditerator2 __first2, _forwarditerator __result,
          _binaryoperation __op);

// [alg.replace]

template<class _executionpolicy, class _forwarditerator, class _unarypredicate, class _tp>
__pstl::internal::enable_if_execution_policy<_executionpolicy, void>
replace_if(_executionpolicy&& __exec, _forwarditerator __first, _forwarditerator __last, _unarypredicate __pred, const _tp& __new_value);

template<class _executionpolicy, class _forwarditerator, class _tp>
__pstl::internal::enable_if_execution_policy<_executionpolicy, void>
replace(_executionpolicy&& __exec, _forwarditerator __first, _forwarditerator __last, const _tp& __old_value, const _tp& __new_value);

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2, class _unarypredicate, class _tp>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator2>
replace_copy_if(_executionpolicy&& __exec, _forwarditerator1 __first, _forwarditerator1 __last, _forwarditerator2 __result, _unarypredicate __pred,
                const _tp& __new_value);

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2, class _tp>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator2>
replace_copy(_executionpolicy&& __exec, _forwarditerator1 __first, _forwarditerator1 __last, _forwarditerator2 __result, const _tp& __old_value,
             const _tp& __new_value);

// [alg.fill]

template <class _executionpolicy, class _forwarditerator, class _tp>
__pstl::internal::enable_if_execution_policy<_executionpolicy, void>
fill( _executionpolicy&& __exec, _forwarditerator __first, _forwarditerator __last, const _tp& __value );

template< class _executionpolicy, class _forwarditerator, class _size, class _tp>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator>
fill_n( _executionpolicy&& __exec, _forwarditerator __first, _size __count, const _tp& __value );

// [alg.generate]
template< class _executionpolicy, class _forwarditerator, class _generator>
__pstl::internal::enable_if_execution_policy<_executionpolicy, void>
generate( _executionpolicy&& __exec, _forwarditerator __first, _forwarditerator __last, _generator __g );

template< class _executionpolicy, class _forwarditerator, class _size, class _generator>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator>
generate_n( _executionpolicy&& __exec, _forwarditerator __first, _size count, _generator __g );

// [alg.remove]

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2, class _predicate>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator2>
remove_copy_if(_executionpolicy&& __exec, _forwarditerator1 __first, _forwarditerator1 __last, _forwarditerator2 __result, _predicate __pred);

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2, class _tp>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator2>
remove_copy(_executionpolicy&& __exec, _forwarditerator1 __first, _forwarditerator1 __last, _forwarditerator2 __result, const _tp& __value);

template<class _executionpolicy, class _forwarditerator, class _unarypredicate>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator>
remove_if(_executionpolicy&& __exec, _forwarditerator __first, _forwarditerator __last, _unarypredicate __pred);

template<class _executionpolicy, class _forwarditerator, class _tp>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator>
remove(_executionpolicy&& __exec, _forwarditerator __first, _forwarditerator __last, const _tp& __value);

// [alg.unique]

template<class _executionpolicy, class _forwarditerator, class _binarypredicate>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator>
unique(_executionpolicy&& __exec, _forwarditerator __first, _forwarditerator __last, _binarypredicate __pred);

template<class _executionpolicy, class _forwarditerator>
__pstl::internal::enable_if_execution_policy<_executionpolicy, _forwarditerator>
unique(_executionpolicy&& __exec, _forwarditerator __first, _forwarditerator __last);

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2, class _binarypredicate>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator2>
unique_copy(_executionpolicy&& __exec, _forwarditerator1 __first, _forwarditerator1 __last, _forwarditerator2 __result, _binarypredicate __pred);

template<class _executionpolicy, class _forwarditerator1, class _forwarditerator2>
__pstl::internal::enable_if_execution_policy<_executionpolicy,_forwarditerator2>
unique_copy(_executionpolicy&& __exec, _forwarditerator1 __first, _forwarditerator1 __last, _forwarditerator2 __result);

// [alg.reverse]

template<class _executionpolicy, class _bidirectionaliterator>
__pstl::internal::enable_if_execution_policy<_executionpolicy, void>
reverse(_executionpolicy&& __exec, _bidirectionaliterator __first, _bidirectionaliterator __last);

template<class _ExecutionPolicy, class _BidirectionalIterator, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
reverse_copy(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __last, _ForwardIterator __d_first);

// [alg.rotate]

template<class _ExecutionPolicy, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
rotate(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
rotate_copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __middle, _ForwardIterator1 __last, _ForwardIterator2 __result);

// [alg.partitions]

template<class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
is_partitioned(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred);

template<class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
partition(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred);

template<class _ExecutionPolicy, class _BidirectionalIterator, class _UnaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _BidirectionalIterator>
stable_partition(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __last, _UnaryPredicate __pred);

template<class _ExecutionPolicy, class _ForwardIterator, class _ForwardIterator1, class _ForwardIterator2, class _UnaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, std::pair<_ForwardIterator1, _ForwardIterator2>>
partition_copy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _ForwardIterator1 __out_true, _ForwardIterator2 __out_false,
               _UnaryPredicate __pred);

// [alg.sort]

template<class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp);

template<class _ExecutionPolicy, class _RandomAccessIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last);

// [stable.sort]

template<class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
stable_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp);

template<class _ExecutionPolicy, class _RandomAccessIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
stable_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last);

// [mismatch]

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, std::pair<_ForwardIterator1, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
         _BinaryPredicate pred);

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, std::pair<_ForwardIterator1, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _BinaryPredicate __pred);

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2 >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, std::pair<_ForwardIterator1, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2);

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2 >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, std::pair<_ForwardIterator1, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2);

// [alg.equal]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _BinaryPredicate __p);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _BinaryPredicate __p);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2);

// [alg.move]
template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2 >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
move(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __d_first);

// [partial.sort]

template<class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
partial_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator middle, _RandomAccessIterator __last, _Compare __comp);

template<class _ExecutionPolicy, class _RandomAccessIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
partial_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator middle, _RandomAccessIterator __last);

// [partial.sort.copy]

template<class _ExecutionPolicy, class _ForwardIterator, class _RandomAccessIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
partial_sort_copy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator d_first, _RandomAccessIterator d_last,
                  _Compare __comp);

template<class _ExecutionPolicy, class _ForwardIterator, class _RandomAccessIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
partial_sort_copy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator d_first, _RandomAccessIterator d_last);

// [is.sorted]
template<class _ExecutionPolicy, class _ForwardIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
is_sorted_until(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp);

template<class _ExecutionPolicy, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
is_sorted_until(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template<class _ExecutionPolicy, class _ForwardIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp);

template<class _ExecutionPolicy, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

// [alg.nth.element]

template<class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
nth_element(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp);

template<class _ExecutionPolicy, class _RandomAccessIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
nth_element(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last);

// [alg.merge]
template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
merge(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
      _ForwardIterator d_first, _Compare __comp);

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
merge(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
      _ForwardIterator d_first);

template< class _ExecutionPolicy, class _BidirectionalIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
inplace_merge(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __middle, _BidirectionalIterator __last, _Compare __comp);

template< class _ExecutionPolicy, class _BidirectionalIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
inplace_merge(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __middle, _BidirectionalIterator __last);

// [includes]

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
includes(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp);

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
includes(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2);

// [set.union]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_union(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
          _ForwardIterator __result, _Compare __comp);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_union(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
          _ForwardIterator2 __last2, _ForwardIterator __result);

// [set.intersection]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_intersection(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                 _ForwardIterator __result, _Compare __comp);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_intersection(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                 _ForwardIterator __result);

// [set.difference]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
               _ForwardIterator __result, _Compare __comp);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
               _ForwardIterator __result);

// [set.symmetric.difference]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_symmetric_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                         _ForwardIterator result, _Compare __comp);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_symmetric_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                         _ForwardIterator __result);

// [is.heap]
template< class _ExecutionPolicy, class _RandomAccessIterator, class _Compare >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
is_heap_until(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp);

template< class _ExecutionPolicy, class _RandomAccessIterator >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
is_heap_until(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last);

template< class _ExecutionPolicy, class _RandomAccessIterator, class _Compare >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
is_heap(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp);

template< class _ExecutionPolicy, class _RandomAccessIterator >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
is_heap(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last);

// [alg.min.max]

template< class _ExecutionPolicy, class _ForwardIterator, class _Compare >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
min_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp);

template< class _ExecutionPolicy, class _ForwardIterator >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
min_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template< class _ExecutionPolicy, class _ForwardIterator, class _Compare >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
max_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp);

template< class _ExecutionPolicy, class _ForwardIterator >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
max_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template< class _ExecutionPolicy, class _ForwardIterator, class _Compare >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, std::pair<_ForwardIterator, _ForwardIterator>>
minmax_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp);

template< class _ExecutionPolicy, class _ForwardIterator >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, std::pair<_ForwardIterator, _ForwardIterator>>
minmax_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

// [alg.lex.comparison]

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Compare >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
lexicographical_compare(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                        _Compare __comp);

template< class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2 >
__pstl::internal::enable_if_execution_policy<_ExecutionPolicy, bool>
lexicographical_compare(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2);

} // namespace std
#endif /* __PSTL_glue_algorithm_defs_H_ */
