// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_PSTL_OFFLOAD__
#error "-fsycl-pstl-offload option should be passed to the compiler to run this test"
#endif

// WARNING: don't include unguarded standard and oneDPL headers
// to guarantee that intercepting overloads of oneapi::dpl:: algorithms
// appears before corresponding functions in oneDPL headers

// The idea of the test is to check that for each algorithm
// the call to std::algorithm_name(std::par_unseq, ...)
// results in a low-level call to oneapi::dpl::algorithm_name(device_policy<KernelName>, ...)
// and hence the offload to the correct device happends

// To check that special overloads for oneapi::dpl:: algorithms added
// BEFORE including the oneDPL and any standard header, because in __SYCL_PSTL_OFFLOAD mode
// each standard header inclusion results in inclusion of oneDPL

// Define guard to include only standard part of the header
// without additional pstl offload part
#define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#include <type_traits>
#include <utility>
#include <iterator>
#undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL

namespace oneapi::dpl::execution {
inline namespace v1 {

template <typename T>
struct is_execution_policy;

} // inline namespace v1
} // namespace oneapi::dpl::execution

struct not_iterator {
    using difference_type = int;
    using value_type = int;
    using pointer = int*;
    using reference = int&;
    using iterator_category = std::random_access_iterator_tag;
};

template <typename ExecutionPolicy, typename T>
using test_enable_if_execution_policy = std::enable_if_t<
    oneapi::dpl::execution::is_execution_policy<std::decay_t<ExecutionPolicy>>::value, T>;

enum class algorithm_id {
    EMPTY_ID,
    ANY_OF,
    ALL_OF,
    NONE_OF,
    FOR_EACH,
    FOR_EACH_N,
    FIND_IF,
    FIND_IF_NOT,
    FIND,
    FIND_END_PREDICATE,
    FIND_END,
    FIND_FIRST_OF_PREDICATE,
    FIND_FIRST_OF,
    ADJACENT_FIND_PREDICATE,
    ADJACENT_FIND,
    COUNT,
    COUNT_IF,
    SEARCH_PREDICATE,
    SEARCH,
    SEARCH_N_PREDICATE,
    SEARCH_N,
    COPY,
    COPY_N,
    COPY_IF,
    SWAP_RANGES,
    TRANSFORM_BINARY,
    TRANSFORM_UNARY,
    REPLACE_IF,
    REPLACE,
    REPLACE_COPY_IF,
    REPLACE_COPY,
    FILL,
    FILL_N,
    GENERATE,
    GENERATE_N,
    REMOVE_COPY_IF,
    REMOVE_COPY,
    REMOVE_IF,
    REMOVE,
    UNIQUE_PREDICATE,
    UNIQUE,
    UNIQUE_COPY_PREDICATE,
    UNIQUE_COPY,
    REVERSE,
    REVERSE_COPY,
    ROTATE,
    ROTATE_COPY,
    IS_PARTITIONED,
    PARTITION,
    STABLE_PARTITION,
    PARTITION_COPY,
    SORT_COMPARE,
    SORT,
    STABLE_SORT_COMPARE,
    STABLE_SORT,
    MISMATCH_4ITERS_PREDICATE,
    MISMATCH_4ITERS,
    MISMATCH_3ITERS_PREDICATE,
    MISMATCH_3ITERS,
    EQUAL_3ITERS_PREDICATE,
    EQUAL_3ITERS,
    EQUAL_4ITERS_PREDICATE,
    EQUAL_4ITERS,
    MOVE,
    PARTIAL_SORT_COMPARE,
    PARTIAL_SORT,
    PARTIAL_SORT_COPY_COMPARE,
    PARTIAL_SORT_COPY,
    IS_SORTED_UNTIL_COMPARE,
    IS_SORTED_UNTIL,
    IS_SORTED_COMPARE,
    IS_SORTED,
    MERGE_COMPARE,
    MERGE,
    INPLACE_MERGE_COMPARE,
    INPLACE_MERGE,
    INCLUDES_COMPARE,
    INCLUDES,
    SET_UNION_COMPARE,
    SET_UNION,
    SET_INTERSECTION_COMPARE,
    SET_INTERSECTION,
    SET_DIFFERENCE_COMPARE,
    SET_DIFFERENCE,
    SET_SYMMETRIC_DIFFERENCE_COMPARE,
    SET_SYMMETRIC_DIFFERENCE,
    IS_HEAP_UNTIL_COMPARE,
    IS_HEAP_UNTIL,
    IS_HEAP_COMPARE,
    IS_HEAP,
    MIN_ELEMENT_COMPARE,
    MIN_ELEMENT,
    MAX_ELEMENT_COMPARE,
    MAX_ELEMENT,
    MINMAX_ELEMENT_COMPARE,
    MINMAX_ELEMENT,
    NTH_ELEMENT_COMPARE,
    NTH_ELEMENT,
    LEXICOGRAPHICAL_COMPARE_COMPARE,
    LEXICOGRAPHICAL_COMPARE,
    SHIFT_LEFT,
    SHIFT_RIGHT,
    UNINITIALIZED_COPY,
    UNINITIALIZED_COPY_N,
    UNINITIALIZED_MOVE,
    UNINITIALIZED_MOVE_N,
    UNINITIALIZED_FILL,
    UNINITIALIZED_FILL_N,
    DESTROY,
    DESTROY_N,
    UNINITIALIZED_DEFAULT_CONSTRUCT,
    UNINITIALIZED_DEFAULT_CONSTRUCT_N,
    UNINITIALIZED_VALUE_CONSTRUCT,
    UNINITIALIZED_VALUE_CONSTRUCT_N,
    REDUCE,
    REDUCE_INIT,
    REDUCE_INIT_BINARYOP,
    TRANSFORM_REDUCE,
    TRANSFORM_REDUCE_BINARY_BINARY,
    TRANSFORM_REDUCE_BINARY_UNARY,
    EXCLUSIVE_SCAN_3ITERS_INIT_BINARYOP,
    EXCLUSIVE_SCAN_3ITERS_INIT,
    INCLUSIVE_SCAN_3ITERS,
    INCLUSIVE_SCAN_3ITERS_BINARYOP,
    INCLUSIVE_SCAN_3ITERS_BINARYOP_INIT,
    TRANSFORM_EXCLUSIVE_SCAN,
    TRANSFORM_INCLUSIVE_SCAN_INIT,
    TRANSFORM_INCLUSIVE_SCAN,
    ADJACENT_DIFFERENCE,
    ADJACENT_DIFFERENCE_BINARYOP
};

static algorithm_id algorithm_id_state = algorithm_id::EMPTY_ID;

template <typename _ExecutionPolicy>
void check_policy(const _ExecutionPolicy& exec);

void store_id(algorithm_id);

namespace oneapi::dpl {

template <typename _ExecutionPolicy, typename _Predicate>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
any_of(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Predicate __pred)
{
    store_id(algorithm_id::ANY_OF);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _Predicate>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
all_of(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Predicate __pred)
{
    store_id(algorithm_id::ALL_OF);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _Predicate>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
none_of(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Predicate __pred)
{
    store_id(algorithm_id::NONE_OF);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _Function>
test_enable_if_execution_policy<_ExecutionPolicy, void>
for_each(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Function __f)
{
    store_id(algorithm_id::FOR_EACH);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size, typename _Function>
test_enable_if_execution_policy<_ExecutionPolicy, void>
for_each_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __n, _Function __f)
{
    store_id(algorithm_id::FOR_EACH_N);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Predicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
find_if(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Predicate __pred)
{
    store_id(algorithm_id::FIND_IF);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Predicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
find_if_not(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Predicate __pred)
{
    store_id(algorithm_id::FIND_IF_NOT);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
find(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, const _Tp& __value)
{
    store_id(algorithm_id::FIND);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
find_end(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __s_first,
         _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    store_id(algorithm_id::FIND_END_PREDICATE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
find_end(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __s_first,
         _ForwardIterator2 __s_last)
{
    store_id(algorithm_id::FIND_END);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
find_first_of(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
              _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    store_id(algorithm_id::FIND_FIRST_OF_PREDICATE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
find_first_of(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
              _ForwardIterator2 __s_first, _ForwardIterator2 __s_last)
{
    store_id(algorithm_id::FIND_FIRST_OF);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
adjacent_find(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::ADJACENT_FIND);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
adjacent_find(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _BinaryPredicate __pred)
{
    store_id(algorithm_id::ADJACENT_FIND_PREDICATE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, typename std::iterator_traits<not_iterator>::difference_type>
count(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, const _Tp& __value)
{
    store_id(algorithm_id::COUNT);
    check_policy(__exec);
    return 0;
}

template <typename _ExecutionPolicy, typename _Predicate>
test_enable_if_execution_policy<_ExecutionPolicy, typename std::iterator_traits<not_iterator>::difference_type>
count_if(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Predicate __pred)
{
    store_id(algorithm_id::COUNT_IF);
    check_policy(__exec);
    return 0;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
search(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
       _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    store_id(algorithm_id::SEARCH_PREDICATE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
search(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
       _ForwardIterator2 __s_first, _ForwardIterator2 __s_last)
{
    store_id(algorithm_id::SEARCH);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Size, typename _Tp, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
search_n(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Size __count,
         const _Tp& __value, _BinaryPredicate __pred)
{
    store_id(algorithm_id::SEARCH_N_PREDICATE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Size, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
search_n(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Size __count,
         const _Tp& __value)
{
    store_id(algorithm_id::SEARCH_N);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __result)
{
    store_id(algorithm_id::COPY);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _Size, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
copy_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __n, _ForwardIterator2 __result)
{
    store_id(algorithm_id::COPY_N);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Predicate>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
copy_if(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __result,
        _Predicate __pred)
{
    store_id(algorithm_id::COPY_IF);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
swap_ranges(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1, _ForwardIterator2 __first2)
{
    store_id(algorithm_id::SWAP_RANGES);
    check_policy(__exec);
    return __first2;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
transform(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
          not_iterator __result, _BinaryOperation __op)
{
    store_id(algorithm_id::TRANSFORM_BINARY);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _UnaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __result,
          _UnaryOperation __op)
{
    store_id(algorithm_id::TRANSFORM_UNARY);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _UnaryPredicate, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, void>
replace_if(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _UnaryPredicate __pred,
           const _Tp& __new_value)
{
    store_id(algorithm_id::REPLACE_IF);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, void>
replace(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, const _Tp& __old_value,
        const _Tp& __new_value)
{
    store_id(algorithm_id::REPLACE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _UnaryPredicate, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
replace_copy_if(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
                _ForwardIterator2 __result, _UnaryPredicate __pred, const _Tp& __new_value)
{
    store_id(algorithm_id::REPLACE_COPY_IF);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
replace_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
             _ForwardIterator2 __result, const _Tp& __old_value, const _Tp& __new_value)
{
    store_id(algorithm_id::REPLACE_COPY);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, void>
fill(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, const _Tp& __value)
{
    store_id(algorithm_id::FILL);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
fill_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __count, const _Tp& __value)
{
    store_id(algorithm_id::FILL_N);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Generator>
test_enable_if_execution_policy<_ExecutionPolicy, void>
generate(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Generator __g)
{
    store_id(algorithm_id::GENERATE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size, typename _Generator>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
generate_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __count, _Generator __g)
{
    store_id(algorithm_id::GENERATE_N);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Predicate>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
remove_copy_if(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __result, _Predicate __pred)
{
    store_id(algorithm_id::REMOVE_COPY_IF);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
remove_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
            _ForwardIterator2 __result, const _Tp& __value)
{
    store_id(algorithm_id::REMOVE_COPY);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _UnaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
remove_if(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _UnaryPredicate __pred)
{
    store_id(algorithm_id::REMOVE_IF);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
remove(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, const _Tp& __value)
{
    store_id(algorithm_id::REMOVE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
unique(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _BinaryPredicate __pred)
{
    store_id(algorithm_id::UNIQUE_PREDICATE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
unique(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::UNIQUE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
unique_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __result,
            _BinaryPredicate __pred)
{
    store_id(algorithm_id::UNIQUE_COPY_PREDICATE);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
unique_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __result)
{
    store_id(algorithm_id::UNIQUE_COPY);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
reverse(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::REVERSE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
reverse_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
             not_iterator __d_first)
{
    store_id(algorithm_id::REVERSE_COPY);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
rotate(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __middle, not_iterator __last)
{
    store_id(algorithm_id::ROTATE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
rotate_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __middle, not_iterator __last,
            _ForwardIterator2 __result)
{
    store_id(algorithm_id::ROTATE_COPY);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _UnaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
is_partitioned(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _UnaryPredicate __pred)
{
    store_id(algorithm_id::IS_PARTITIONED);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _UnaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
partition(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _UnaryPredicate __pred)
{
    store_id(algorithm_id::PARTITION);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _UnaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
stable_partition(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _UnaryPredicate __pred)
{
    store_id(algorithm_id::STABLE_PARTITION);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _UnaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, std::pair<_ForwardIterator1, _ForwardIterator2>>
partition_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator1 __out_true, _ForwardIterator2 __out_false, _UnaryPredicate __pred)
{
    store_id(algorithm_id::PARTITION_COPY);
    check_policy(__exec);
    return std::make_pair(__first, __first);
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, void>
sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::SORT_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::SORT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, void>
stable_sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::STABLE_SORT_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
stable_sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::STABLE_SORT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, std::pair<not_iterator, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
         _ForwardIterator2 __first2, _ForwardIterator2 __last2, _BinaryPredicate __pred)
{
    store_id(algorithm_id::MISMATCH_4ITERS_PREDICATE);
    check_policy(__exec);
    return std::make_pair(__first1, __first2);
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, std::pair<not_iterator, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
         _ForwardIterator2 __first2, _BinaryPredicate __pred)
{
    store_id(algorithm_id::MISMATCH_3ITERS_PREDICATE);
    check_policy(__exec);
    return std::make_pair(__first1, __first2);
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, std::pair<not_iterator, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
         _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    store_id(algorithm_id::MISMATCH_4ITERS);
    check_policy(__exec);
    return std::make_pair(__first1, __first2);
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, std::pair<not_iterator, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1, _ForwardIterator2 __first2)
{
    store_id(algorithm_id::MISMATCH_3ITERS);
    check_policy(__exec);
    return std::make_pair(__first1, __first2);
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
      _ForwardIterator2 __first2, _BinaryPredicate __pred)
{
    store_id(algorithm_id::EQUAL_3ITERS_PREDICATE);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1, _ForwardIterator2 __first2)
{
    store_id(algorithm_id::EQUAL_3ITERS);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryPredicate>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
      _ForwardIterator2 __first2, _ForwardIterator2 __last2, _BinaryPredicate __pred)
{
    store_id(algorithm_id::EQUAL_4ITERS_PREDICATE);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    store_id(algorithm_id::EQUAL_4ITERS);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
move(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __d_first)
{
    store_id(algorithm_id::MOVE);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, void>
partial_sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __middle, not_iterator __last,
             _Compare __comp)
{
    store_id(algorithm_id::PARTIAL_SORT_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
partial_sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __middle, not_iterator __last)
{
    store_id(algorithm_id::PARTIAL_SORT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
partial_sort_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
                  not_iterator __d_first, not_iterator __d_last, _Compare __comp)
{
    store_id(algorithm_id::PARTIAL_SORT_COPY_COMPARE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
partial_sort_copy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
                  not_iterator __d_first, not_iterator __d_last)
{
    store_id(algorithm_id::PARTIAL_SORT_COPY);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
is_sorted_until(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::IS_SORTED_UNTIL_COMPARE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
is_sorted_until(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::IS_SORTED_UNTIL);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::IS_SORTED_COMPARE);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::IS_SORTED);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
merge(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
      _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __d_first, _Compare __comp)
{
    store_id(algorithm_id::MERGE_COMPARE);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
merge(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
      _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __d_first)
{
    store_id(algorithm_id::MERGE);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, void>
inplace_merge(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __middle,
              not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::INPLACE_MERGE_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
inplace_merge(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __middle,
              not_iterator __last)
{
    store_id(algorithm_id::INPLACE_MERGE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
includes(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
         _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp)
{
    store_id(algorithm_id::INCLUDES_COMPARE);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
includes(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
         _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    store_id(algorithm_id::INCLUDES);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
set_union(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
          _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __result, _Compare __comp)
{
    store_id(algorithm_id::SET_UNION_COMPARE);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
set_union(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
          _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __result)
{
    store_id(algorithm_id::SET_UNION);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
set_intersection(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __result, _Compare __comp)
{
    store_id(algorithm_id::SET_INTERSECTION_COMPARE);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
set_intersection(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __result)
{
    store_id(algorithm_id::SET_INTERSECTION);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
set_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
               _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __result, _Compare __comp)
{
    store_id(algorithm_id::SET_DIFFERENCE_COMPARE);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
set_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
               _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __result)
{
    store_id(algorithm_id::SET_DIFFERENCE);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
set_symmetric_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                         _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __result, _Compare __comp)
{
    store_id(algorithm_id::SET_SYMMETRIC_DIFFERENCE_COMPARE);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
set_symmetric_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                         _ForwardIterator2 __first2, _ForwardIterator2 __last2, not_iterator __result)
{
    store_id(algorithm_id::SET_SYMMETRIC_DIFFERENCE);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
is_heap_until(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::IS_HEAP_UNTIL_COMPARE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
is_heap_until(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::IS_HEAP_UNTIL);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
is_heap(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::IS_HEAP_COMPARE);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
is_heap(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::IS_HEAP);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
min_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::MIN_ELEMENT_COMPARE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
min_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::MIN_ELEMENT);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
max_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::MAX_ELEMENT_COMPARE);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
max_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::MAX_ELEMENT);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, std::pair<not_iterator, not_iterator>>
minmax_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::MINMAX_ELEMENT_COMPARE);
    check_policy(__exec);
    return std::make_pair(__first, __first);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, std::pair<not_iterator, not_iterator>>
minmax_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::MINMAX_ELEMENT);
    check_policy(__exec);
    return std::make_pair(__first, __first);
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, void>
nth_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __nth, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::NTH_ELEMENT_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
nth_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __nth, not_iterator __last)
{
    store_id(algorithm_id::NTH_ELEMENT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
lexicographical_compare(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
                        _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp)
{
    store_id(algorithm_id::LEXICOGRAPHICAL_COMPARE_COMPARE);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, bool>
lexicographical_compare(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
                        _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    store_id(algorithm_id::LEXICOGRAPHICAL_COMPARE);
    check_policy(__exec);
    return true;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
shift_left(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
           typename std::iterator_traits<not_iterator>::difference_type __n)
{
    store_id(algorithm_id::SHIFT_LEFT);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
shift_right(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
            typename std::iterator_traits<not_iterator>::difference_type __n)
{
    store_id(algorithm_id::SHIFT_RIGHT);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy, typename _InputIterator>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
uninitialized_copy(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last,
                   not_iterator __result)
{
    store_id(algorithm_id::UNINITIALIZED_COPY);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _InputIterator, typename _Size>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
uninitialized_copy_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, not_iterator __result)
{
    store_id(algorithm_id::UNINITIALIZED_COPY_N);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _InputIterator>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
uninitialized_move(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last,
                   not_iterator __result)
{
    store_id(algorithm_id::UNINITIALIZED_MOVE);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _InputIterator, typename _Size>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
uninitialized_move_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, not_iterator __result)
{
    store_id(algorithm_id::UNINITIALIZED_MOVE_N);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_fill(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, const _Tp& __value)
{
    store_id(algorithm_id::UNINITIALIZED_FILL);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, not_iterator>
uninitialized_fill_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __n, const _Tp& __value)
{
    store_id(algorithm_id::UNINITIALIZED_FILL_N);
    check_policy(__exec);
    return __first;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
destroy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::DESTROY);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size>
test_enable_if_execution_policy<_ExecutionPolicy, void>
destroy_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __n)
{
    store_id(algorithm_id::DESTROY_N);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_default_construct(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::UNINITIALIZED_DEFAULT_CONSTRUCT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size>
test_enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_default_construct_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __n)
{
    store_id(algorithm_id::UNINITIALIZED_DEFAULT_CONSTRUCT_N);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_value_construct(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::UNINITIALIZED_VALUE_CONSTRUCT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size>
test_enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_value_construct_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __n)
{
    store_id(algorithm_id::UNINITIALIZED_VALUE_CONSTRUCT_N);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Tp, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Tp __init, _BinaryOperation __binary_op)
{
    store_id(algorithm_id::REDUCE_INIT_BINARYOP);
    check_policy(__exec);
    return __init;
}

template <typename _ExecutionPolicy, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Tp __init)
{
    store_id(algorithm_id::REDUCE_INIT);
    check_policy(__exec);
    return __init;
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy, typename std::iterator_traits<not_iterator>::value_type>
reduce(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::REDUCE);
    check_policy(__exec);
    return 0;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1,
                 _ForwardIterator2 __first2, _Tp __init)
{
    store_id(algorithm_id::TRANSFORM_REDUCE);
    check_policy(__exec);
    return __init;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp, typename _BinaryOperation1, typename _BinaryOperation2>
test_enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, not_iterator __first1, not_iterator __last1, _ForwardIterator2 __first2,
                 _Tp __init, _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    store_id(algorithm_id::TRANSFORM_REDUCE_BINARY_BINARY);
    check_policy(__exec);
    return __init;
}

template <typename _ExecutionPolicy, typename _Tp, typename _BinaryOperation, typename _UnaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Tp __init,
                 _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    store_id(algorithm_id::TRANSFORM_REDUCE_BINARY_UNARY);
    check_policy(__exec);
    return __init;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __d_first, _Tp __init)
{
    store_id(algorithm_id::EXCLUSIVE_SCAN_3ITERS_INIT);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __d_first, _Tp __init, _BinaryOperation __binary_op)
{
    store_id(algorithm_id::EXCLUSIVE_SCAN_3ITERS_INIT_BINARYOP);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __result)
{
    store_id(algorithm_id::INCLUSIVE_SCAN_3ITERS);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op)
{
    store_id(algorithm_id::INCLUSIVE_SCAN_3ITERS_BINARYOP);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op, _Tp __init)
{
    store_id(algorithm_id::INCLUSIVE_SCAN_3ITERS_BINARYOP_INIT);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp, typename _BinaryOperation, typename _UnaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_exclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
                         _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    store_id(algorithm_id::TRANSFORM_EXCLUSIVE_SCAN);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryOperation, typename _UnaryOperation, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_inclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
                         _ForwardIterator2 __result, _BinaryOperation __binary_op, _UnaryOperation __unary_op, _Tp __init)
{
    store_id(algorithm_id::TRANSFORM_INCLUSIVE_SCAN_INIT);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _UnaryOperation, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_inclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
                         _ForwardIterator2 __result, _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    store_id(algorithm_id::TRANSFORM_INCLUSIVE_SCAN);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
adjacent_difference(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
                    _ForwardIterator2 __d_first, _BinaryOperation __op)
{
    store_id(algorithm_id::ADJACENT_DIFFERENCE_BINARYOP);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
adjacent_difference(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
                    _ForwardIterator2 __d_first)
{
    store_id(algorithm_id::ADJACENT_DIFFERENCE);
    check_policy(__exec);
    return __d_first;
}

} // namespace oneapi::dpl

#include <oneapi/dpl/execution>

#include <algorithm>
#include <memory>
#include <numeric>
#include <execution>

#include <support/utils.h>
#include <support/utils_invoke.h>

void store_id(algorithm_id __id) {
    EXPECT_TRUE(algorithm_id_state == algorithm_id::EMPTY_ID, "algorithm_id contains non empty value");
    algorithm_id_state = __id;
}

template <typename _T>
struct is_device_policy : std::false_type {};

template <typename _KernelName>
struct is_device_policy<oneapi::dpl::execution::device_policy<_KernelName>> : std::true_type {};

template <typename _ExecutionPolicy>
void check_policy(const _ExecutionPolicy& __policy) {
    static_assert(is_device_policy<std::decay_t<_ExecutionPolicy>>::value, "Algorithm was not offloaded to the device");
    EXPECT_TRUE(__policy.queue().get_device() == TestUtils::get_pstl_offload_device(), "Algorithm was offloaded to the wrong device");
}

template <algorithm_id _AlgorithmId, typename _RunAlgorithmBody, typename... _AlgorithmArgs>
void test_algorithm(_RunAlgorithmBody __run_algorithm, _AlgorithmArgs&&... __args) {
    EXPECT_TRUE(algorithm_id_state == algorithm_id::EMPTY_ID, "algorithm_id was not reset");
    __run_algorithm(std::execution::par_unseq, std::forward<_AlgorithmArgs>(__args)...);

    EXPECT_TRUE(algorithm_id_state == _AlgorithmId, "Algorithm was not redirected to the device version or incorrect low-level algorithm was called");
    algorithm_id_state = algorithm_id::EMPTY_ID;
}

#define RUN_LAMBDA(ALGORITHM_NAME) [](const auto& policy, auto... args) { std::ALGORITHM_NAME(policy, args...); }

int main() {
    not_iterator iter;
    auto binary_predicate = [](int, int) { return true; };
    auto unary_predicate = [](int) { return true; };
    auto function = [](int) {};
    auto rng = []() { return 1; };

    // Testing <algorithm>
    test_algorithm<algorithm_id::ANY_OF>(RUN_LAMBDA(any_of), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::ALL_OF>(RUN_LAMBDA(all_of), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::NONE_OF>(RUN_LAMBDA(none_of), iter, iter, binary_predicate);

    test_algorithm<algorithm_id::FOR_EACH>(RUN_LAMBDA(for_each), iter, iter, function);
    test_algorithm<algorithm_id::FOR_EACH_N>(RUN_LAMBDA(for_each_n), iter, 5, function);

    test_algorithm<algorithm_id::FIND_IF>(RUN_LAMBDA(find_if), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::FIND_IF_NOT>(RUN_LAMBDA(find_if_not), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::FIND>(RUN_LAMBDA(find), iter, iter, 1);
    test_algorithm<algorithm_id::FIND_END_PREDICATE>(RUN_LAMBDA(find_end), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::FIND_END>(RUN_LAMBDA(find_end), iter, iter, iter, iter);
    test_algorithm<algorithm_id::FIND_FIRST_OF_PREDICATE>(RUN_LAMBDA(find_first_of), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::FIND_FIRST_OF>(RUN_LAMBDA(find_first_of), iter, iter, iter, iter);
    test_algorithm<algorithm_id::ADJACENT_FIND_PREDICATE>(RUN_LAMBDA(adjacent_find), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::ADJACENT_FIND>(RUN_LAMBDA(adjacent_find), iter, iter);

    test_algorithm<algorithm_id::COUNT>(RUN_LAMBDA(count), iter, iter, 0);
    test_algorithm<algorithm_id::COUNT_IF>(RUN_LAMBDA(count_if), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::SEARCH_PREDICATE>(RUN_LAMBDA(search), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SEARCH>(RUN_LAMBDA(search), iter, iter, iter, iter);
    test_algorithm<algorithm_id::SEARCH_N_PREDICATE>(RUN_LAMBDA(search_n), iter, iter, 5, 0, binary_predicate);
    test_algorithm<algorithm_id::SEARCH_N>(RUN_LAMBDA(search_n), iter, iter, 5, 0);

    test_algorithm<algorithm_id::COPY>(RUN_LAMBDA(copy), iter, iter, iter);
    test_algorithm<algorithm_id::COPY_N>(RUN_LAMBDA(copy_n), iter, 5, iter);
    test_algorithm<algorithm_id::COPY_IF>(RUN_LAMBDA(copy_if), iter, iter, iter, unary_predicate);
    test_algorithm<algorithm_id::SWAP_RANGES>(RUN_LAMBDA(swap_ranges), iter, iter, iter);

    test_algorithm<algorithm_id::TRANSFORM_BINARY>(RUN_LAMBDA(transform), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::TRANSFORM_UNARY>(RUN_LAMBDA(transform), iter, iter, iter, unary_predicate);

    test_algorithm<algorithm_id::REPLACE_IF>(RUN_LAMBDA(replace_if), iter, iter, unary_predicate, 5);
    test_algorithm<algorithm_id::REPLACE>(RUN_LAMBDA(replace), iter, iter, 5, 0);
    test_algorithm<algorithm_id::REPLACE_COPY_IF>(RUN_LAMBDA(replace_copy_if), iter, iter, iter, unary_predicate, 5);
    test_algorithm<algorithm_id::REPLACE_COPY>(RUN_LAMBDA(replace_copy), iter, iter, iter, 5, 0);

    test_algorithm<algorithm_id::FILL>(RUN_LAMBDA(fill), iter, iter, 0);
    test_algorithm<algorithm_id::FILL_N>(RUN_LAMBDA(fill_n), iter, 5, 0);

    test_algorithm<algorithm_id::GENERATE>(RUN_LAMBDA(generate), iter, iter, rng);
    test_algorithm<algorithm_id::GENERATE_N>(RUN_LAMBDA(generate_n), iter, 5, rng);

    test_algorithm<algorithm_id::REMOVE_COPY_IF>(RUN_LAMBDA(remove_copy_if), iter, iter, iter, unary_predicate);
    test_algorithm<algorithm_id::REMOVE_COPY>(RUN_LAMBDA(remove_copy), iter, iter, iter, 0);
    test_algorithm<algorithm_id::REMOVE_IF>(RUN_LAMBDA(remove_if), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::REMOVE>(RUN_LAMBDA(remove), iter, iter, 5);

    test_algorithm<algorithm_id::UNIQUE_PREDICATE>(RUN_LAMBDA(unique), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::UNIQUE>(RUN_LAMBDA(unique), iter, iter);
    test_algorithm<algorithm_id::UNIQUE_COPY_PREDICATE>(RUN_LAMBDA(unique_copy), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::UNIQUE_COPY>(RUN_LAMBDA(unique_copy), iter, iter, iter);

    test_algorithm<algorithm_id::REVERSE>(RUN_LAMBDA(reverse), iter, iter);
    test_algorithm<algorithm_id::REVERSE_COPY>(RUN_LAMBDA(reverse_copy), iter, iter, iter);
    test_algorithm<algorithm_id::ROTATE>(RUN_LAMBDA(rotate), iter, iter, iter);
    test_algorithm<algorithm_id::ROTATE_COPY>(RUN_LAMBDA(rotate_copy), iter, iter, iter, iter);

    test_algorithm<algorithm_id::IS_PARTITIONED>(RUN_LAMBDA(is_partitioned), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::PARTITION>(RUN_LAMBDA(partition), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::STABLE_PARTITION>(RUN_LAMBDA(stable_partition), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::PARTITION_COPY>(RUN_LAMBDA(partition_copy), iter, iter, iter, iter, unary_predicate);

    test_algorithm<algorithm_id::SORT_COMPARE>(RUN_LAMBDA(sort), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SORT>(RUN_LAMBDA(sort), iter, iter);
    test_algorithm<algorithm_id::STABLE_SORT_COMPARE>(RUN_LAMBDA(stable_sort), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::STABLE_SORT>(RUN_LAMBDA(stable_sort), iter, iter);

    test_algorithm<algorithm_id::MISMATCH_4ITERS_PREDICATE>(RUN_LAMBDA(mismatch), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MISMATCH_3ITERS_PREDICATE>(RUN_LAMBDA(mismatch), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MISMATCH_4ITERS>(RUN_LAMBDA(mismatch), iter, iter, iter, iter);
    test_algorithm<algorithm_id::MISMATCH_3ITERS>(RUN_LAMBDA(mismatch), iter, iter, iter);

    test_algorithm<algorithm_id::EQUAL_4ITERS_PREDICATE>(RUN_LAMBDA(equal), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::EQUAL_3ITERS_PREDICATE>(RUN_LAMBDA(equal), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::EQUAL_4ITERS>(RUN_LAMBDA(equal), iter, iter, iter, iter);
    test_algorithm<algorithm_id::EQUAL_3ITERS>(RUN_LAMBDA(equal), iter, iter, iter);

    test_algorithm<algorithm_id::MOVE>(RUN_LAMBDA(move), iter, iter, iter);

    test_algorithm<algorithm_id::PARTIAL_SORT_COMPARE>(RUN_LAMBDA(partial_sort), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::PARTIAL_SORT>(RUN_LAMBDA(partial_sort), iter, iter, iter);
    test_algorithm<algorithm_id::PARTIAL_SORT_COPY_COMPARE>(RUN_LAMBDA(partial_sort_copy), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::PARTIAL_SORT_COPY>(RUN_LAMBDA(partial_sort_copy), iter, iter, iter, iter);

    test_algorithm<algorithm_id::IS_SORTED_UNTIL_COMPARE>(RUN_LAMBDA(is_sorted_until), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::IS_SORTED_UNTIL>(RUN_LAMBDA(is_sorted_until), iter, iter);
    test_algorithm<algorithm_id::IS_SORTED_COMPARE>(RUN_LAMBDA(is_sorted), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::IS_SORTED>(RUN_LAMBDA(is_sorted), iter, iter);

    test_algorithm<algorithm_id::MERGE_COMPARE>(RUN_LAMBDA(merge), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MERGE>(RUN_LAMBDA(merge), iter, iter, iter, iter, iter);
    test_algorithm<algorithm_id::INPLACE_MERGE_COMPARE>(RUN_LAMBDA(inplace_merge), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::INPLACE_MERGE>(RUN_LAMBDA(inplace_merge), iter, iter, iter);

    test_algorithm<algorithm_id::INCLUDES_COMPARE>(RUN_LAMBDA(includes), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::INCLUDES>(RUN_LAMBDA(includes), iter, iter, iter, iter);

    test_algorithm<algorithm_id::SET_UNION_COMPARE>(RUN_LAMBDA(set_union), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SET_UNION>(RUN_LAMBDA(set_union), iter, iter, iter, iter, iter);
    test_algorithm<algorithm_id::SET_INTERSECTION_COMPARE>(RUN_LAMBDA(set_intersection), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SET_INTERSECTION>(RUN_LAMBDA(set_intersection), iter, iter, iter, iter, iter);
    test_algorithm<algorithm_id::SET_DIFFERENCE_COMPARE>(RUN_LAMBDA(set_difference), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SET_DIFFERENCE>(RUN_LAMBDA(set_difference), iter, iter, iter, iter, iter);
    test_algorithm<algorithm_id::SET_SYMMETRIC_DIFFERENCE_COMPARE>(RUN_LAMBDA(set_symmetric_difference), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SET_SYMMETRIC_DIFFERENCE>(RUN_LAMBDA(set_symmetric_difference), iter, iter, iter, iter, iter);

    test_algorithm<algorithm_id::IS_HEAP_UNTIL_COMPARE>(RUN_LAMBDA(is_heap_until), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::IS_HEAP_UNTIL>(RUN_LAMBDA(is_heap_until), iter, iter);
    test_algorithm<algorithm_id::IS_HEAP_COMPARE>(RUN_LAMBDA(is_heap), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::IS_HEAP>(RUN_LAMBDA(is_heap), iter, iter);

    test_algorithm<algorithm_id::MIN_ELEMENT_COMPARE>(RUN_LAMBDA(min_element), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MIN_ELEMENT>(RUN_LAMBDA(min_element), iter, iter);
    test_algorithm<algorithm_id::MAX_ELEMENT_COMPARE>(RUN_LAMBDA(max_element), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MAX_ELEMENT>(RUN_LAMBDA(max_element), iter, iter);
    test_algorithm<algorithm_id::MINMAX_ELEMENT_COMPARE>(RUN_LAMBDA(minmax_element), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MINMAX_ELEMENT>(RUN_LAMBDA(minmax_element), iter, iter);
    test_algorithm<algorithm_id::NTH_ELEMENT_COMPARE>(RUN_LAMBDA(nth_element), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::NTH_ELEMENT>(RUN_LAMBDA(nth_element), iter, iter, iter);

    test_algorithm<algorithm_id::LEXICOGRAPHICAL_COMPARE_COMPARE>(RUN_LAMBDA(lexicographical_compare), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::LEXICOGRAPHICAL_COMPARE>(RUN_LAMBDA(lexicographical_compare), iter, iter, iter, iter);

    test_algorithm<algorithm_id::SHIFT_LEFT>(RUN_LAMBDA(shift_left), iter, iter, 0);
    test_algorithm<algorithm_id::SHIFT_RIGHT>(RUN_LAMBDA(shift_right), iter, iter, 0);

    // // Testing <memory>
    test_algorithm<algorithm_id::UNINITIALIZED_COPY>(RUN_LAMBDA(uninitialized_copy), iter, iter, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_COPY_N>(RUN_LAMBDA(uninitialized_copy_n), iter, 0, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_MOVE>(RUN_LAMBDA(uninitialized_move), iter, iter, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_MOVE_N>(RUN_LAMBDA(uninitialized_move_n), iter, 0, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_FILL>(RUN_LAMBDA(uninitialized_fill), iter, iter, 0);
    test_algorithm<algorithm_id::UNINITIALIZED_FILL_N>(RUN_LAMBDA(uninitialized_fill_n), iter, 0, 0);

    test_algorithm<algorithm_id::DESTROY>(RUN_LAMBDA(destroy), iter, iter);
    test_algorithm<algorithm_id::DESTROY_N>(RUN_LAMBDA(destroy_n), iter, 0);

    test_algorithm<algorithm_id::UNINITIALIZED_DEFAULT_CONSTRUCT>(RUN_LAMBDA(uninitialized_default_construct), iter, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_DEFAULT_CONSTRUCT_N>(RUN_LAMBDA(uninitialized_default_construct_n), iter, 0);
    test_algorithm<algorithm_id::UNINITIALIZED_VALUE_CONSTRUCT>(RUN_LAMBDA(uninitialized_value_construct), iter, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_VALUE_CONSTRUCT_N>(RUN_LAMBDA(uninitialized_value_construct_n), iter, 0);

    // // Testing <numeric>
    test_algorithm<algorithm_id::REDUCE_INIT_BINARYOP>(RUN_LAMBDA(reduce), iter, iter, 0, binary_predicate);
    test_algorithm<algorithm_id::REDUCE_INIT>(RUN_LAMBDA(reduce), iter, iter, 0);
    test_algorithm<algorithm_id::REDUCE>(RUN_LAMBDA(reduce), iter, iter);

    test_algorithm<algorithm_id::TRANSFORM_REDUCE>(RUN_LAMBDA(transform_reduce), iter, iter, iter, 0);
    test_algorithm<algorithm_id::TRANSFORM_REDUCE_BINARY_BINARY>(RUN_LAMBDA(transform_reduce), iter, iter, iter, 0, binary_predicate, binary_predicate);
    test_algorithm<algorithm_id::TRANSFORM_REDUCE_BINARY_UNARY>(RUN_LAMBDA(transform_reduce), iter, iter, 0, binary_predicate, unary_predicate);

    test_algorithm<algorithm_id::EXCLUSIVE_SCAN_3ITERS_INIT>(RUN_LAMBDA(exclusive_scan), iter, iter, iter, 0);
    test_algorithm<algorithm_id::EXCLUSIVE_SCAN_3ITERS_INIT_BINARYOP>(RUN_LAMBDA(exclusive_scan), iter, iter, iter, 0, binary_predicate);

    test_algorithm<algorithm_id::INCLUSIVE_SCAN_3ITERS>(RUN_LAMBDA(inclusive_scan), iter, iter, iter);
    test_algorithm<algorithm_id::INCLUSIVE_SCAN_3ITERS_BINARYOP>(RUN_LAMBDA(inclusive_scan), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::INCLUSIVE_SCAN_3ITERS_BINARYOP_INIT>(RUN_LAMBDA(inclusive_scan), iter, iter, iter, binary_predicate, 0);

    test_algorithm<algorithm_id::TRANSFORM_EXCLUSIVE_SCAN>(RUN_LAMBDA(transform_exclusive_scan), iter, iter, iter, 0, binary_predicate, unary_predicate);
    test_algorithm<algorithm_id::TRANSFORM_INCLUSIVE_SCAN>(RUN_LAMBDA(transform_inclusive_scan), iter, iter, iter, binary_predicate, unary_predicate);
    test_algorithm<algorithm_id::TRANSFORM_INCLUSIVE_SCAN_INIT>(RUN_LAMBDA(transform_inclusive_scan), iter, iter, iter, binary_predicate, unary_predicate, 0);

    test_algorithm<algorithm_id::ADJACENT_DIFFERENCE_BINARYOP>(RUN_LAMBDA(adjacent_difference), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::ADJACENT_DIFFERENCE>(RUN_LAMBDA(adjacent_difference), iter, iter, iter);

    return TestUtils::done();
}
