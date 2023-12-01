// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !__SYCL_PSTL_OFFLOAD__
#error "PSTL offload compiler mode should be enabled to run this test"
#endif

// ATTENTION: don't include oneDPL and unguarded standard headers
// to guarantee that intercepting overloads of oneapi::dpl:: algorithms
// appear before corresponding functions in oneDPL headers

// The idea of the test is to check that each standard parallel algorithm
// call with std::execution::par_unseq policy results in
// oneapi::dpl::algorithm_name(device_policy<KernelName>, ...) call
// and hence the offload to the correct device would happen

// To check that, special overloads for oneapi::dpl:: algorithms added
// BEFORE including the oneDPL

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

template <typename ExecutionPolicy, typename T = void>
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
    TRANSFORM_BINARYOP,
    TRANSFORM_UNARYOP,
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
    TRANSFORM_REDUCE_BINARYOP_BINARYOP,
    TRANSFORM_REDUCE_BINARYOP_UNARYOP,
    EXCLUSIVE_SCAN_INIT_BINARYOP,
    EXCLUSIVE_SCAN_INIT,
    INCLUSIVE_SCAN,
    INCLUSIVE_SCAN_BINARYOP,
    INCLUSIVE_SCAN_BINARYOP_INIT,
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
test_enable_if_execution_policy<_ExecutionPolicy>
for_each(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Function __f)
{
    store_id(algorithm_id::FOR_EACH);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size, typename _Function>
test_enable_if_execution_policy<_ExecutionPolicy>
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
    store_id(algorithm_id::TRANSFORM_BINARYOP);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _UnaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _ForwardIterator2 __result,
          _UnaryOperation __op)
{
    store_id(algorithm_id::TRANSFORM_UNARYOP);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _UnaryPredicate, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy>
replace_if(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _UnaryPredicate __pred,
           const _Tp& __new_value)
{
    store_id(algorithm_id::REPLACE_IF);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::SORT_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy>
sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::SORT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Compare>
test_enable_if_execution_policy<_ExecutionPolicy>
stable_sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::STABLE_SORT_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
partial_sort(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __middle, not_iterator __last,
             _Compare __comp)
{
    store_id(algorithm_id::PARTIAL_SORT_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
inplace_merge(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __middle,
              not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::INPLACE_MERGE_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
nth_element(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __nth, not_iterator __last, _Compare __comp)
{
    store_id(algorithm_id::NTH_ELEMENT_COMPARE);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
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
test_enable_if_execution_policy<_ExecutionPolicy>
destroy(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::DESTROY);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size>
test_enable_if_execution_policy<_ExecutionPolicy>
destroy_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __n)
{
    store_id(algorithm_id::DESTROY_N);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy>
uninitialized_default_construct(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::UNINITIALIZED_DEFAULT_CONSTRUCT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size>
test_enable_if_execution_policy<_ExecutionPolicy>
uninitialized_default_construct_n(_ExecutionPolicy&& __exec, not_iterator __first, _Size __n)
{
    store_id(algorithm_id::UNINITIALIZED_DEFAULT_CONSTRUCT_N);
    check_policy(__exec);
}

template <typename _ExecutionPolicy>
test_enable_if_execution_policy<_ExecutionPolicy>
uninitialized_value_construct(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last)
{
    store_id(algorithm_id::UNINITIALIZED_VALUE_CONSTRUCT);
    check_policy(__exec);
}

template <typename _ExecutionPolicy, typename _Size>
test_enable_if_execution_policy<_ExecutionPolicy>
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
    store_id(algorithm_id::TRANSFORM_REDUCE_BINARYOP_BINARYOP);
    check_policy(__exec);
    return __init;
}

template <typename _ExecutionPolicy, typename _Tp, typename _BinaryOperation, typename _UnaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last, _Tp __init,
                 _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    store_id(algorithm_id::TRANSFORM_REDUCE_BINARYOP_UNARYOP);
    check_policy(__exec);
    return __init;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __d_first, _Tp __init)
{
    store_id(algorithm_id::EXCLUSIVE_SCAN_INIT);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __d_first, _Tp __init, _BinaryOperation __binary_op)
{
    store_id(algorithm_id::EXCLUSIVE_SCAN_INIT_BINARYOP);
    check_policy(__exec);
    return __d_first;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __result)
{
    store_id(algorithm_id::INCLUSIVE_SCAN);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op)
{
    store_id(algorithm_id::INCLUSIVE_SCAN_BINARYOP);
    check_policy(__exec);
    return __result;
}

template <typename _ExecutionPolicy, typename _ForwardIterator2, typename _Tp, typename _BinaryOperation>
test_enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, not_iterator __first, not_iterator __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op, _Tp __init)
{
    store_id(algorithm_id::INCLUSIVE_SCAN_BINARYOP_INIT);
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
struct is_device_policy_impl : std::false_type {};

template <typename _KernelName>
struct is_device_policy_impl<oneapi::dpl::execution::device_policy<_KernelName>> : std::true_type {};

template <typename T>
struct is_device_policy : is_device_policy_impl<std::decay_t<T>> {};

template <typename _ExecutionPolicy>
void check_policy(const _ExecutionPolicy& __policy) {
    static_assert(is_device_policy<std::decay_t<_ExecutionPolicy>>::value, "Algorithm was redirected with unexpected policy type");
    EXPECT_TRUE(__policy.queue().get_device() == TestUtils::get_pstl_offload_device(), "The passed policy is associated with the wrong device");
}

template <algorithm_id _AlgorithmId, typename _RunAlgorithmBody, typename... _AlgorithmArgs>
void test_algorithm(_RunAlgorithmBody __run_algorithm, _AlgorithmArgs&&... __args) {
    EXPECT_TRUE(algorithm_id_state == algorithm_id::EMPTY_ID, "algorithm_id was not reset");
    __run_algorithm(std::execution::par_unseq, std::forward<_AlgorithmArgs>(__args)...);

    EXPECT_TRUE(algorithm_id_state != algorithm_id::EMPTY_ID, "Algorithm was not redirected");
    EXPECT_TRUE(algorithm_id_state == _AlgorithmId, "Algorithm was redirected to the wrong oneDPL algorithm");
    algorithm_id_state = algorithm_id::EMPTY_ID;
}

#define ALGORITHM_WRAPPER(ALGORITHM_NAME) [](auto&&... args) { std::ALGORITHM_NAME(std::forward<decltype(args)>(args)...); }

int main() {
    not_iterator iter;
    auto binary_predicate = [](int, int) { return true; };
    auto unary_predicate = [](int) { return true; };

    // Testing <algorithm>
    test_algorithm<algorithm_id::ANY_OF>(ALGORITHM_WRAPPER(any_of), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::ALL_OF>(ALGORITHM_WRAPPER(all_of), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::NONE_OF>(ALGORITHM_WRAPPER(none_of), iter, iter, binary_predicate);

    {
        auto function = [](int) {};
        test_algorithm<algorithm_id::FOR_EACH>(ALGORITHM_WRAPPER(for_each), iter, iter, function);
        test_algorithm<algorithm_id::FOR_EACH_N>(ALGORITHM_WRAPPER(for_each_n), iter, 5, function);
    }

    test_algorithm<algorithm_id::FIND_IF>(ALGORITHM_WRAPPER(find_if), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::FIND_IF_NOT>(ALGORITHM_WRAPPER(find_if_not), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::FIND>(ALGORITHM_WRAPPER(find), iter, iter, 1);
    test_algorithm<algorithm_id::FIND_END_PREDICATE>(ALGORITHM_WRAPPER(find_end), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::FIND_END>(ALGORITHM_WRAPPER(find_end), iter, iter, iter, iter);
    test_algorithm<algorithm_id::FIND_FIRST_OF_PREDICATE>(ALGORITHM_WRAPPER(find_first_of), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::FIND_FIRST_OF>(ALGORITHM_WRAPPER(find_first_of), iter, iter, iter, iter);
    test_algorithm<algorithm_id::ADJACENT_FIND_PREDICATE>(ALGORITHM_WRAPPER(adjacent_find), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::ADJACENT_FIND>(ALGORITHM_WRAPPER(adjacent_find), iter, iter);

    test_algorithm<algorithm_id::COUNT>(ALGORITHM_WRAPPER(count), iter, iter, 0);
    test_algorithm<algorithm_id::COUNT_IF>(ALGORITHM_WRAPPER(count_if), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::SEARCH_PREDICATE>(ALGORITHM_WRAPPER(search), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SEARCH>(ALGORITHM_WRAPPER(search), iter, iter, iter, iter);
    test_algorithm<algorithm_id::SEARCH_N_PREDICATE>(ALGORITHM_WRAPPER(search_n), iter, iter, 5, 0, binary_predicate);
    test_algorithm<algorithm_id::SEARCH_N>(ALGORITHM_WRAPPER(search_n), iter, iter, 5, 0);

    test_algorithm<algorithm_id::COPY>(ALGORITHM_WRAPPER(copy), iter, iter, iter);
    test_algorithm<algorithm_id::COPY_N>(ALGORITHM_WRAPPER(copy_n), iter, 5, iter);
    test_algorithm<algorithm_id::COPY_IF>(ALGORITHM_WRAPPER(copy_if), iter, iter, iter, unary_predicate);
    test_algorithm<algorithm_id::SWAP_RANGES>(ALGORITHM_WRAPPER(swap_ranges), iter, iter, iter);

    test_algorithm<algorithm_id::TRANSFORM_BINARYOP>(ALGORITHM_WRAPPER(transform), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::TRANSFORM_UNARYOP>(ALGORITHM_WRAPPER(transform), iter, iter, iter, unary_predicate);

    test_algorithm<algorithm_id::REPLACE_IF>(ALGORITHM_WRAPPER(replace_if), iter, iter, unary_predicate, 5);
    test_algorithm<algorithm_id::REPLACE>(ALGORITHM_WRAPPER(replace), iter, iter, 5, 0);
    test_algorithm<algorithm_id::REPLACE_COPY_IF>(ALGORITHM_WRAPPER(replace_copy_if), iter, iter, iter, unary_predicate, 5);
    test_algorithm<algorithm_id::REPLACE_COPY>(ALGORITHM_WRAPPER(replace_copy), iter, iter, iter, 5, 0);

    test_algorithm<algorithm_id::FILL>(ALGORITHM_WRAPPER(fill), iter, iter, 0);
    test_algorithm<algorithm_id::FILL_N>(ALGORITHM_WRAPPER(fill_n), iter, 5, 0);

    {
        auto gen = []() { return 1; };
        test_algorithm<algorithm_id::GENERATE>(ALGORITHM_WRAPPER(generate), iter, iter, gen);
        test_algorithm<algorithm_id::GENERATE_N>(ALGORITHM_WRAPPER(generate_n), iter, 5, gen);
    }

    test_algorithm<algorithm_id::REMOVE_COPY_IF>(ALGORITHM_WRAPPER(remove_copy_if), iter, iter, iter, unary_predicate);
    test_algorithm<algorithm_id::REMOVE_COPY>(ALGORITHM_WRAPPER(remove_copy), iter, iter, iter, 0);
    test_algorithm<algorithm_id::REMOVE_IF>(ALGORITHM_WRAPPER(remove_if), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::REMOVE>(ALGORITHM_WRAPPER(remove), iter, iter, 5);

    test_algorithm<algorithm_id::UNIQUE_PREDICATE>(ALGORITHM_WRAPPER(unique), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::UNIQUE>(ALGORITHM_WRAPPER(unique), iter, iter);
    test_algorithm<algorithm_id::UNIQUE_COPY_PREDICATE>(ALGORITHM_WRAPPER(unique_copy), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::UNIQUE_COPY>(ALGORITHM_WRAPPER(unique_copy), iter, iter, iter);

    test_algorithm<algorithm_id::REVERSE>(ALGORITHM_WRAPPER(reverse), iter, iter);
    test_algorithm<algorithm_id::REVERSE_COPY>(ALGORITHM_WRAPPER(reverse_copy), iter, iter, iter);
    test_algorithm<algorithm_id::ROTATE>(ALGORITHM_WRAPPER(rotate), iter, iter, iter);
    test_algorithm<algorithm_id::ROTATE_COPY>(ALGORITHM_WRAPPER(rotate_copy), iter, iter, iter, iter);

    test_algorithm<algorithm_id::IS_PARTITIONED>(ALGORITHM_WRAPPER(is_partitioned), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::PARTITION>(ALGORITHM_WRAPPER(partition), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::STABLE_PARTITION>(ALGORITHM_WRAPPER(stable_partition), iter, iter, unary_predicate);
    test_algorithm<algorithm_id::PARTITION_COPY>(ALGORITHM_WRAPPER(partition_copy), iter, iter, iter, iter, unary_predicate);

    test_algorithm<algorithm_id::SORT_COMPARE>(ALGORITHM_WRAPPER(sort), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SORT>(ALGORITHM_WRAPPER(sort), iter, iter);
    test_algorithm<algorithm_id::STABLE_SORT_COMPARE>(ALGORITHM_WRAPPER(stable_sort), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::STABLE_SORT>(ALGORITHM_WRAPPER(stable_sort), iter, iter);

    test_algorithm<algorithm_id::MISMATCH_4ITERS_PREDICATE>(ALGORITHM_WRAPPER(mismatch), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MISMATCH_3ITERS_PREDICATE>(ALGORITHM_WRAPPER(mismatch), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MISMATCH_4ITERS>(ALGORITHM_WRAPPER(mismatch), iter, iter, iter, iter);
    test_algorithm<algorithm_id::MISMATCH_3ITERS>(ALGORITHM_WRAPPER(mismatch), iter, iter, iter);

    test_algorithm<algorithm_id::EQUAL_4ITERS_PREDICATE>(ALGORITHM_WRAPPER(equal), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::EQUAL_3ITERS_PREDICATE>(ALGORITHM_WRAPPER(equal), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::EQUAL_4ITERS>(ALGORITHM_WRAPPER(equal), iter, iter, iter, iter);
    test_algorithm<algorithm_id::EQUAL_3ITERS>(ALGORITHM_WRAPPER(equal), iter, iter, iter);

    test_algorithm<algorithm_id::MOVE>(ALGORITHM_WRAPPER(move), iter, iter, iter);

    test_algorithm<algorithm_id::PARTIAL_SORT_COMPARE>(ALGORITHM_WRAPPER(partial_sort), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::PARTIAL_SORT>(ALGORITHM_WRAPPER(partial_sort), iter, iter, iter);
    test_algorithm<algorithm_id::PARTIAL_SORT_COPY_COMPARE>(ALGORITHM_WRAPPER(partial_sort_copy), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::PARTIAL_SORT_COPY>(ALGORITHM_WRAPPER(partial_sort_copy), iter, iter, iter, iter);

    test_algorithm<algorithm_id::IS_SORTED_UNTIL_COMPARE>(ALGORITHM_WRAPPER(is_sorted_until), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::IS_SORTED_UNTIL>(ALGORITHM_WRAPPER(is_sorted_until), iter, iter);
    test_algorithm<algorithm_id::IS_SORTED_COMPARE>(ALGORITHM_WRAPPER(is_sorted), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::IS_SORTED>(ALGORITHM_WRAPPER(is_sorted), iter, iter);

    test_algorithm<algorithm_id::MERGE_COMPARE>(ALGORITHM_WRAPPER(merge), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MERGE>(ALGORITHM_WRAPPER(merge), iter, iter, iter, iter, iter);
    test_algorithm<algorithm_id::INPLACE_MERGE_COMPARE>(ALGORITHM_WRAPPER(inplace_merge), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::INPLACE_MERGE>(ALGORITHM_WRAPPER(inplace_merge), iter, iter, iter);

    test_algorithm<algorithm_id::INCLUDES_COMPARE>(ALGORITHM_WRAPPER(includes), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::INCLUDES>(ALGORITHM_WRAPPER(includes), iter, iter, iter, iter);

    test_algorithm<algorithm_id::SET_UNION_COMPARE>(ALGORITHM_WRAPPER(set_union), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SET_UNION>(ALGORITHM_WRAPPER(set_union), iter, iter, iter, iter, iter);
    test_algorithm<algorithm_id::SET_INTERSECTION_COMPARE>(ALGORITHM_WRAPPER(set_intersection), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SET_INTERSECTION>(ALGORITHM_WRAPPER(set_intersection), iter, iter, iter, iter, iter);
    test_algorithm<algorithm_id::SET_DIFFERENCE_COMPARE>(ALGORITHM_WRAPPER(set_difference), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SET_DIFFERENCE>(ALGORITHM_WRAPPER(set_difference), iter, iter, iter, iter, iter);
    test_algorithm<algorithm_id::SET_SYMMETRIC_DIFFERENCE_COMPARE>(ALGORITHM_WRAPPER(set_symmetric_difference), iter, iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::SET_SYMMETRIC_DIFFERENCE>(ALGORITHM_WRAPPER(set_symmetric_difference), iter, iter, iter, iter, iter);

    test_algorithm<algorithm_id::IS_HEAP_UNTIL_COMPARE>(ALGORITHM_WRAPPER(is_heap_until), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::IS_HEAP_UNTIL>(ALGORITHM_WRAPPER(is_heap_until), iter, iter);
    test_algorithm<algorithm_id::IS_HEAP_COMPARE>(ALGORITHM_WRAPPER(is_heap), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::IS_HEAP>(ALGORITHM_WRAPPER(is_heap), iter, iter);

    test_algorithm<algorithm_id::MIN_ELEMENT_COMPARE>(ALGORITHM_WRAPPER(min_element), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MIN_ELEMENT>(ALGORITHM_WRAPPER(min_element), iter, iter);
    test_algorithm<algorithm_id::MAX_ELEMENT_COMPARE>(ALGORITHM_WRAPPER(max_element), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MAX_ELEMENT>(ALGORITHM_WRAPPER(max_element), iter, iter);
    test_algorithm<algorithm_id::MINMAX_ELEMENT_COMPARE>(ALGORITHM_WRAPPER(minmax_element), iter, iter, binary_predicate);
    test_algorithm<algorithm_id::MINMAX_ELEMENT>(ALGORITHM_WRAPPER(minmax_element), iter, iter);
    test_algorithm<algorithm_id::NTH_ELEMENT_COMPARE>(ALGORITHM_WRAPPER(nth_element), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::NTH_ELEMENT>(ALGORITHM_WRAPPER(nth_element), iter, iter, iter);

    test_algorithm<algorithm_id::LEXICOGRAPHICAL_COMPARE_COMPARE>(ALGORITHM_WRAPPER(lexicographical_compare), iter, iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::LEXICOGRAPHICAL_COMPARE>(ALGORITHM_WRAPPER(lexicographical_compare), iter, iter, iter, iter);

    test_algorithm<algorithm_id::SHIFT_LEFT>(ALGORITHM_WRAPPER(shift_left), iter, iter, 0);
    test_algorithm<algorithm_id::SHIFT_RIGHT>(ALGORITHM_WRAPPER(shift_right), iter, iter, 0);

    // // Testing <memory>
    test_algorithm<algorithm_id::UNINITIALIZED_COPY>(ALGORITHM_WRAPPER(uninitialized_copy), iter, iter, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_COPY_N>(ALGORITHM_WRAPPER(uninitialized_copy_n), iter, 0, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_MOVE>(ALGORITHM_WRAPPER(uninitialized_move), iter, iter, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_MOVE_N>(ALGORITHM_WRAPPER(uninitialized_move_n), iter, 0, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_FILL>(ALGORITHM_WRAPPER(uninitialized_fill), iter, iter, 0);
    test_algorithm<algorithm_id::UNINITIALIZED_FILL_N>(ALGORITHM_WRAPPER(uninitialized_fill_n), iter, 0, 0);

    test_algorithm<algorithm_id::DESTROY>(ALGORITHM_WRAPPER(destroy), iter, iter);
    test_algorithm<algorithm_id::DESTROY_N>(ALGORITHM_WRAPPER(destroy_n), iter, 0);

    test_algorithm<algorithm_id::UNINITIALIZED_DEFAULT_CONSTRUCT>(ALGORITHM_WRAPPER(uninitialized_default_construct), iter, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_DEFAULT_CONSTRUCT_N>(ALGORITHM_WRAPPER(uninitialized_default_construct_n), iter, 0);
    test_algorithm<algorithm_id::UNINITIALIZED_VALUE_CONSTRUCT>(ALGORITHM_WRAPPER(uninitialized_value_construct), iter, iter);
    test_algorithm<algorithm_id::UNINITIALIZED_VALUE_CONSTRUCT_N>(ALGORITHM_WRAPPER(uninitialized_value_construct_n), iter, 0);

    // // Testing <numeric>
    test_algorithm<algorithm_id::REDUCE_INIT_BINARYOP>(ALGORITHM_WRAPPER(reduce), iter, iter, 0, binary_predicate);
    test_algorithm<algorithm_id::REDUCE_INIT>(ALGORITHM_WRAPPER(reduce), iter, iter, 0);
    test_algorithm<algorithm_id::REDUCE>(ALGORITHM_WRAPPER(reduce), iter, iter);

    test_algorithm<algorithm_id::TRANSFORM_REDUCE>(ALGORITHM_WRAPPER(transform_reduce), iter, iter, iter, 0);
    test_algorithm<algorithm_id::TRANSFORM_REDUCE_BINARYOP_BINARYOP>(ALGORITHM_WRAPPER(transform_reduce), iter, iter, iter, 0, binary_predicate, binary_predicate);
    test_algorithm<algorithm_id::TRANSFORM_REDUCE_BINARYOP_UNARYOP>(ALGORITHM_WRAPPER(transform_reduce), iter, iter, 0, binary_predicate, unary_predicate);

    test_algorithm<algorithm_id::EXCLUSIVE_SCAN_INIT>(ALGORITHM_WRAPPER(exclusive_scan), iter, iter, iter, 0);
    test_algorithm<algorithm_id::EXCLUSIVE_SCAN_INIT_BINARYOP>(ALGORITHM_WRAPPER(exclusive_scan), iter, iter, iter, 0, binary_predicate);

    test_algorithm<algorithm_id::INCLUSIVE_SCAN>(ALGORITHM_WRAPPER(inclusive_scan), iter, iter, iter);
    test_algorithm<algorithm_id::INCLUSIVE_SCAN_BINARYOP>(ALGORITHM_WRAPPER(inclusive_scan), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::INCLUSIVE_SCAN_BINARYOP_INIT>(ALGORITHM_WRAPPER(inclusive_scan), iter, iter, iter, binary_predicate, 0);

    test_algorithm<algorithm_id::TRANSFORM_EXCLUSIVE_SCAN>(ALGORITHM_WRAPPER(transform_exclusive_scan), iter, iter, iter, 0, binary_predicate, unary_predicate);
    test_algorithm<algorithm_id::TRANSFORM_INCLUSIVE_SCAN>(ALGORITHM_WRAPPER(transform_inclusive_scan), iter, iter, iter, binary_predicate, unary_predicate);
    test_algorithm<algorithm_id::TRANSFORM_INCLUSIVE_SCAN_INIT>(ALGORITHM_WRAPPER(transform_inclusive_scan), iter, iter, iter, binary_predicate, unary_predicate, 0);

    test_algorithm<algorithm_id::ADJACENT_DIFFERENCE_BINARYOP>(ALGORITHM_WRAPPER(adjacent_difference), iter, iter, iter, binary_predicate);
    test_algorithm<algorithm_id::ADJACENT_DIFFERENCE>(ALGORITHM_WRAPPER(adjacent_difference), iter, iter, iter);

    return TestUtils::done();
}
