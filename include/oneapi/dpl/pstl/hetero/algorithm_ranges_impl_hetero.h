// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_ALGORITHM_RANGES_IMPL_HETERO_H
#define _ONEDPL_ALGORITHM_RANGES_IMPL_HETERO_H

#include "../algorithm_fwd.h"
#include "../parallel_backend.h"
#include "utils_hetero.h"

#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/utils_ranges_sycl.h"
#    include "dpcpp/unseq_backend_sycl.h"
#    include "dpcpp/parallel_backend_sycl_utils.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//------------------------------------------------------------------------
// walk_n
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Function, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk_n(_ExecutionPolicy&& __exec, _Function __f, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    if (__n > 0)
    {
        oneapi::dpl::__par_backend_hetero::__parallel_for(::std::forward<_ExecutionPolicy>(__exec),
                                                          unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
                                                          ::std::forward<_Ranges>(__rngs)...)
            .wait();
    }
}

//------------------------------------------------------------------------
// swap
//------------------------------------------------------------------------

template <typename _Name>
class __swap1_wrapper
{
};

template <typename _Name>
class __swap2_wrapper
{
};

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_swap(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Function __f)
{
    if (__rng1.size() <= __rng2.size())
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__swap1_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __f, __rng1, __rng2);
        return __rng1.size();
    }

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__swap2_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        __f, __rng2, __rng1);
    return __rng2.size();
}

//------------------------------------------------------------------------
// equal
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_equal(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Pred __pred)
{
    if (__rng1.empty() || __rng2.empty() || __rng1.size() != __rng2.size())
        return false;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, equal_predicate<_Pred>>;

    // TODO: in case of confilicting names
    // __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__or_policy_wrapper>()
    return !oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        ::std::forward<_ExecutionPolicy>(__exec), _Predicate{equal_predicate<_Pred>{__pred}},
        oneapi::dpl::__par_backend_hetero::__parallel_or_tag{},
        oneapi::dpl::__ranges::zip_view(::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2)));
}

//------------------------------------------------------------------------
// find_if
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_find_if(_ExecutionPolicy&& __exec, _Range&& __rng, _Pred __pred)
{
    //trivial pre-checks
    if (__rng.empty())
        return __rng.size();

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range>;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__find_policy_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, _TagType{}, ::std::forward<_Range>(__rng));
}

//------------------------------------------------------------------------
// find_end
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range1>>
__pattern_find_end(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Pred __pred)
{
    //trivial pre-checks
    if (__rng1.empty() || __rng2.empty() || __rng1.size() < __rng2.size())
        return __rng1.size();

    if (__rng1.size() == __rng2.size())
    {
        const bool __res = __pattern_equal(::std::forward<_ExecutionPolicy>(__exec), __rng1,
                                           ::std::forward<_Range2>(__rng2), __pred);
        return __res ? 0 : __rng1.size();
    }

    using _Predicate = unseq_backend::multiple_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = __par_backend_hetero::__parallel_find_backward_tag<_Range1>;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__find_policy_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, _TagType{}, ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2));
}

//------------------------------------------------------------------------
// find_first_of
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range1>>
__pattern_find_first_of(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Pred __pred)
{
    //trivial pre-checks
    if (__rng1.empty() || __rng2.empty())
        return __rng1.size();

    using _Predicate = unseq_backend::first_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range1>;

    //TODO: To check whether it makes sense to iterate over the second sequence in case of __rng1.size() < __rng2.size()
    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__find_policy_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, _TagType{}, ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2));
}

//------------------------------------------------------------------------
// any_of
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_any_of(_ExecutionPolicy&& __exec, _Range&& __rng, _Pred __pred)
{
    if (__rng.empty())
        return false;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, _Pred>;
    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __par_backend_hetero::make_wrapped_policy<oneapi::dpl::__par_backend_hetero::__or_policy_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, oneapi::dpl::__par_backend_hetero::__parallel_or_tag{}, ::std::forward<_Range>(__rng));
}

//------------------------------------------------------------------------
// search
//------------------------------------------------------------------------

template <typename Name>
class equal_wrapper
{
};

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range1>>
__pattern_search(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Pred __pred)
{
    //trivial pre-checks
    if (__rng2.empty())
        return 0;
    if (__rng1.size() < __rng2.size())
        return __rng1.size();

    if (__rng1.size() == __rng2.size())
    {
        const bool __res = __pattern_equal(
            __par_backend_hetero::make_wrapped_policy<equal_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
            __rng1, ::std::forward<_Range2>(__rng2), __pred);
        return __res ? 0 : __rng1.size();
    }

    using _Predicate = unseq_backend::multiple_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range1>;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<oneapi::dpl::__par_backend_hetero::__find_policy_wrapper>
            (::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, _TagType{}, ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2));
}

//------------------------------------------------------------------------
// search_n
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Size, typename _Tp, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_search_n(_ExecutionPolicy&& __exec, _Range&& __rng, _Size __count, const _Tp& __value,
                   _BinaryPredicate __pred)
{
    //TODO: To consider definition a kind of special factory "multiple_view" (addition to standard "single_view").
    //The factory "multiple_view" would generate a range of N identical values.
    auto __s_rng = oneapi::dpl::experimental::ranges::views::iota(0, __count) |
                   oneapi::dpl::experimental::ranges::views::transform([__value](auto) { return __value; });

    return __pattern_search(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __s_rng, __pred);
}

template <typename _Size>
_Size
return_value(bool __res, _Size __size, ::std::true_type)
{
    return __res ? 0 : __size;
}

template <typename _Size>
_Size
return_value(_Size __res, _Size __size, ::std::false_type)
{
    return __res == __size - 1 ? __size : __res;
}

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate, typename _OrFirstTag>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_adjacent_find(_ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __predicate,
                        _OrFirstTag __is__or_semantic)
{
    if (__rng.size() < 2)
        return __rng.size();

    using _Predicate =
        oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, adjacent_find_fn<_BinaryPredicate>>;
    using _TagType =
        typename ::std::conditional<__is__or_semantic(), oneapi::dpl::__par_backend_hetero::__parallel_or_tag,
                                    oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range>>::type;

    auto __rng1 = __rng | oneapi::dpl::experimental::ranges::views::take(__rng.size() - 1);
    auto __rng2 = __rng | oneapi::dpl::experimental::ranges::views::drop(1);

    // TODO: in case of confilicting names
    // __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__or_policy_wrapper>()
    auto result = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        ::std::forward<_ExecutionPolicy>(__exec), _Predicate{adjacent_find_fn<_BinaryPredicate>{__predicate}},
        _TagType{}, oneapi::dpl::__ranges::zip_view(__rng1, __rng2));

    // inverted conditional because of
    // reorder_predicate in glue_algorithm_impl.h
    return return_value(result, __rng.size(), __is__or_semantic);
}

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_count(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __predicate)
{
    if (__rng.size() == 0)
        return 0;

    using _ReduceValueType = oneapi::dpl::__internal::__difference_t<_Range>;

    auto __identity_init_fn = acc_handler_count<_Predicate>{__predicate};
    auto __identity_reduce_fn = ::std::plus<_ReduceValueType>{};

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<
               _ReduceValueType, decltype(__identity_reduce_fn), decltype(__identity_init_fn)>(
               ::std::forward<_ExecutionPolicy>(__exec), __identity_reduce_fn, __identity_init_fn,
               unseq_backend::__no_init_value{}, // no initial value
               ::std::forward<_Range>(__rng))
        .get();
}

//------------------------------------------------------------------------
// copy_if
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _CreateMaskOp, typename _CopyByMaskOp>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range1>>
__pattern_scan_copy(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _CreateMaskOp __create_mask_op,
                    _CopyByMaskOp __copy_by_mask_op)
{
    if (__rng1.size() == 0)
        return __rng1.size();

    using _SizeType = decltype(__rng1.size());
    using _ReduceOp = ::std::plus<_SizeType>;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _MaskAssigner = unseq_backend::__mask_assigner<1>;
    using _InitType = unseq_backend::__no_init_value<_SizeType>;
    using _DataAcc = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _Assigner __assign_op;
    _ReduceOp __reduce_op;
    _DataAcc __get_data_op;
    _MaskAssigner __add_mask_op;

    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, int32_t> __mask_buf(__exec,
                                                                                                  __rng1.size());

    auto __res =
        __par_backend_hetero::__parallel_transform_scan_base(
            ::std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__ranges::zip_view(
                __rng1, oneapi::dpl::__ranges::all_view<int32_t, __par_backend_hetero::access_mode::read_write>(
                            __mask_buf.get_buffer())),
            __rng2, __reduce_op, _InitType{},
            // local scan
            unseq_backend::__scan</*inclusive*/ ::std::true_type, _ExecutionPolicy, _ReduceOp, _DataAcc, _Assigner,
                                  _MaskAssigner, _CreateMaskOp, _InitType>{__reduce_op, __get_data_op, __assign_op,
                                                                           __add_mask_op, __create_mask_op},
            // scan between groups
            unseq_backend::__scan</*inclusive*/ ::std::true_type, _ExecutionPolicy, _ReduceOp, _DataAcc, _NoAssign,
                                  _Assigner, _DataAcc, _InitType>{__reduce_op, __get_data_op, _NoAssign{}, __assign_op,
                                                                  __get_data_op},
            // global scan
            __copy_by_mask_op)
            .get();

    return __res;
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Predicate,
          typename _Assign = oneapi::dpl::__internal::__pstl_assign>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range2>>
__pattern_copy_if(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Predicate __pred, _Assign)
{
    using _SizeType = decltype(__rng1.size());
    using _ReduceOp = ::std::plus<_SizeType>;

    unseq_backend::__create_mask<_Predicate, _SizeType> __create_mask_op{__pred};
    unseq_backend::__copy_by_mask<_ReduceOp, _Assign, /*inclusive*/ ::std::true_type, 1> __copy_by_mask_op;

    return __pattern_scan_copy(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                               ::std::forward<_Range2>(__rng2), __create_mask_op, __copy_by_mask_op);
}

//------------------------------------------------------------------------
// remove_if
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_remove_if(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    if (__rng.size() == 0)
        return __rng.size();

    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;

    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __buf(__exec, __rng.size());
    auto __copy_rng = oneapi::dpl::__ranges::views::all(__buf.get_buffer());

    auto __copy_last_id = __pattern_copy_if(__exec, __rng, __copy_rng, __not_pred<_Predicate>{__pred},
                                            oneapi::dpl::__internal::__pstl_assign());
    auto __copy_rng_truncated = __copy_rng | oneapi::dpl::experimental::ranges::views::take(__copy_last_id);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(::std::forward<_ExecutionPolicy>(__exec),
                                                        oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{},
                                                        __copy_rng_truncated, ::std::forward<_Range>(__rng));

    return __copy_last_id;
}

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate,
          typename _Assign = oneapi::dpl::__internal::__pstl_assign>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range2>>
__pattern_unique_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _BinaryPredicate __pred, _Assign)
{
    using _It1DifferenceType = oneapi::dpl::__internal::__difference_t<_Range1>;
    unseq_backend::__copy_by_mask<::std::plus<_It1DifferenceType>, _Assign, /*inclusive*/ ::std::true_type, 1>
        __copy_by_mask_op;
    __create_mask_unique_copy<__not_pred<_BinaryPredicate>, _It1DifferenceType> __create_mask_op{
        __not_pred<_BinaryPredicate>{__pred}};

    return __pattern_scan_copy(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng),
                               ::std::forward<_Range2>(__result), __create_mask_op, __copy_by_mask_op);
}

//------------------------------------------------------------------------
// unique
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_unique(_ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __pred)
{
    if (__rng.size() == 0)
        return __rng.size();

    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;

    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __buf(__exec, __rng.size());
    auto res_rng = oneapi::dpl::__ranges::views::all(__buf.get_buffer());
    auto res = __pattern_unique_copy(__exec, __rng, res_rng, __pred, oneapi::dpl::__internal::__pstl_assign());

    __pattern_walk_n(::std::forward<_ExecutionPolicy>(__exec), __brick_copy<_ExecutionPolicy>{}, res_rng,
                     ::std::forward<_Range>(__rng));
    return res;
}

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------

template <typename _Name>
class __copy1_wrapper
{
};

template <typename _Name>
class __copy2_wrapper
{
};

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range3>>
__pattern_merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    auto __n1 = __rng1.size();
    auto __n2 = __rng2.size();
    auto __n = __n1 + __n2;
    if (__n == 0)
        return 0;

    //To consider the direct copying pattern call in case just one of sequences is empty.
    if (__n1 == 0)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy1_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::forward<_Range2>(__rng2),
            ::std::forward<_Range3>(__rng3));
    }
    else if (__n2 == 0)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy2_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::forward<_Range1>(__rng1),
            ::std::forward<_Range3>(__rng3));
    }
    else
    {
        __par_backend_hetero::__parallel_merge(::std::forward<_ExecutionPolicy>(__exec),
                                               ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2),
                                               ::std::forward<_Range3>(__rng3), __comp)
            .wait();
    }

    return __n;
}

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp, _Proj __proj)
{
    if (__rng.size() >= 2)
        __par_backend_hetero::__parallel_stable_sort(::std::forward<_ExecutionPolicy>(__exec),
                                                     ::std::forward<_Range>(__rng), __comp, __proj)
            .wait();
}

//------------------------------------------------------------------------
// min_element
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_min_element(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    //If size == 1, result is the zero-indexed element. If size == 0, result is 0.
    if (__rng.size() < 2)
        return 0;

    using _IteratorValueType = typename ::std::iterator_traits<decltype(__rng.begin())>::value_type;
    using _IndexValueType = oneapi::dpl::__internal::__difference_t<_Range>;
    using _ReduceValueType = oneapi::dpl::__internal::tuple<_IndexValueType, _IteratorValueType>;

    auto __identity_init_fn = __acc_handler_minelement<_ReduceValueType>{};
    auto __identity_reduce_fn = [__comp](_ReduceValueType __a, _ReduceValueType __b) {
        using ::std::get;
        return __comp(get<1>(__b), get<1>(__a)) ? __b : __a;
    };

    auto __ret_idx =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType, decltype(__identity_reduce_fn),
                                                                       decltype(__identity_init_fn)>(
            ::std::forward<_ExecutionPolicy>(__exec), __identity_reduce_fn, __identity_init_fn,
            unseq_backend::__no_init_value{}, // no initial value
            ::std::forward<_Range>(__rng))
            .get();

    using ::std::get;
    return get<0>(__ret_idx);
}

//------------------------------------------------------------------------
// minmax_element
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<
    _ExecutionPolicy,
    ::std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__difference_t<_Range>>>
__pattern_minmax_element(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    //If size == 1, result is the zero-indexed element. If size == 0, result is 0.
    if (__rng.size() < 2)
        return ::std::make_pair(0, 0);

    using _IteratorValueType = typename ::std::iterator_traits<decltype(__rng.begin())>::value_type;
    using _IndexValueType = oneapi::dpl::__internal::__difference_t<_Range>;
    using _ReduceValueType =
        oneapi::dpl::__internal::tuple<_IndexValueType, _IndexValueType, _IteratorValueType, _IteratorValueType>;

    auto __identity_init_fn = __acc_handler_minmaxelement<_ReduceValueType>{};

    _ReduceValueType __ret =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType, __identity_reduce_fn<_Compare>,
                                                                       decltype(__identity_init_fn)>(
            ::std::forward<_ExecutionPolicy>(__exec), __identity_reduce_fn<_Compare>{__comp}, __identity_init_fn,
            unseq_backend::__no_init_value{}, // no initial value
            ::std::forward<_Range>(__rng))
            .get();

    using ::std::get;
    return ::std::make_pair(get<0>(__ret), get<1>(__ret));
}

//------------------------------------------------------------------------
// reduce_by_segment
//------------------------------------------------------------------------

template <typename _Name>
class __copy_keys_wrapper
{
};

template <typename _Name>
class __copy_values_wrapper
{
};

template <typename _Name>
class __reduce1_wrapper
{
};

template <typename _Name>
class __reduce2_wrapper
{
};

template <typename _Name>
class __assign_key1_wrapper
{
};

template <typename _Name>
class __assign_key2_wrapper
{
};

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range3>>
__pattern_reduce_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                            _Range4&& __out_values, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
{
    // The algorithm reduces values in __values where the
    // associated keys for the values are equal to the adjacent key.
    //
    // Example: __keys       = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 }
    //          __values     = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 }
    //
    //          __out_keys   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 }
    //          __out_values = { 1, 2, 3, 4, 2, 6, 2, 6, 0 }

    const auto __n = __keys.size();

    if (__n == 0)
        return 0;

    if (__n == 1)
    {
        __brick_copy<_ExecutionPolicy> __copy_range{};

        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_keys_wrapper>(__exec),
            __copy_range, ::std::forward<_Range1>(__keys), ::std::forward<_Range3>(__out_keys));

        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_values_wrapper>
                (::std::forward<_ExecutionPolicy>(__exec)),
            __copy_range, ::std::forward<_Range2>(__values), ::std::forward<_Range4>(__out_values));

        return 1;
    }

    using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;
    using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
    using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;

    // Round 1: reduce with extra indices added to avoid long segments
    // TODO: At threshold points check if the key is equal to the key at the previous threshold point, indicating a long sequence.
    // Skip a round of copy_if and reduces if there are none.
    auto __idx = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __diff_type>(__exec, __n)
                     .get_buffer();
    auto __tmp_out_keys =
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __key_type>(__exec, __n).get_buffer();
    auto __tmp_out_values =
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, __n).get_buffer();

    // Replicating first element of keys view to be able to compare (i-1)-th and (i)-th key with aligned sequences,
    //  dropping the last key for the i-1 sequence.
    auto __k1 =
        oneapi::dpl::__ranges::take_view_simple(oneapi::dpl::__ranges::replicate_start_view_simple(__keys, 1), __n);

    // view1 elements are a tuple of the element index and pairs of adjacent keys
    // view2 elements are a tuple of the elements where key-index pairs will be written by copy_if
    auto __view1 = experimental::ranges::zip_view(experimental::ranges::views::iota(0, __n), __k1, __keys);
    auto __view2 = experimental::ranges::zip_view(experimental::ranges::views::all_write(__tmp_out_keys),
                                                  experimental::ranges::views::all_write(__idx));

    // use work group size adjusted to shared local memory as the maximum segment size.
    ::std::size_t __wgroup_size =
        oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, sizeof(__key_type) + sizeof(__val_type));

    // element is copied if it is the 0th element (marks beginning of first segment), is in an index
    // evenly divisible by wg size (ensures segments are not long), or has a key not equal to the
    // adjacent element (marks end of real segments)
    // TODO: replace wgroup size with segment size based on platform specifics.
    auto __intermediate_result_end = __pattern_copy_if(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__assign_key1_wrapper>(__exec), __view1, __view2,
        [__n, __binary_pred, __wgroup_size](const auto& __a) {
            // The size of key range for the (i-1) view is one less, so for the 0th index we do not check the keys
            // for (i-1), but we still need to get its key value as it is the start of a segment
            const auto index = ::std::get<0>(__a);
            if (index == 0)
                return true;
            return index % __wgroup_size == 0                                 // segment size
                   || !__binary_pred(::std::get<1>(__a), ::std::get<2>(__a)); // key comparison
        },
        unseq_backend::__brick_assign_key_position{});

    //reduce by segment
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce1_wrapper>(__exec),
        unseq_backend::__brick_reduce_idx<_BinaryOperator, decltype(__n)>(__binary_op, __n), __intermediate_result_end,
        oneapi::dpl::__ranges::take_view_simple(experimental::ranges::views::all_read(__idx),
                                                __intermediate_result_end),
        ::std::forward<_Range2>(__values), experimental::ranges::views::all_write(__tmp_out_values))
        .wait();

    // Round 2: final reduction to get result for each segment of equal adjacent keys
    // create views over adjacent keys
    oneapi::dpl::__ranges::all_view<__key_type, __par_backend_hetero::access_mode::read_write> __new_keys(
        __tmp_out_keys);

    // Replicating first element of key views to be able to compare (i-1)-th and (i)-th key,
    //  dropping the last key for the i-1 sequence.  Only taking the appropriate number of keys to start with here.
    auto __clipped_new_keys = oneapi::dpl::__ranges::take_view_simple(__new_keys, __intermediate_result_end);

    auto __k3 = oneapi::dpl::__ranges::take_view_simple(
        oneapi::dpl::__ranges::replicate_start_view_simple(__clipped_new_keys, 1), __intermediate_result_end);

    // view3 elements are a tuple of the element index and pairs of adjacent keys
    // view4 elements are a tuple of the elements where key-index pairs will be written by copy_if
    auto __view3 = experimental::ranges::zip_view(experimental::ranges::views::iota(0, __intermediate_result_end), __k3,
                                                  __clipped_new_keys);
    auto __view4 = experimental::ranges::zip_view(experimental::ranges::views::all_write(__out_keys),
                                                  experimental::ranges::views::all_write(__idx));

    // element is copied if it is the 0th element (marks beginning of first segment), or has a key not equal to
    // the adjacent element (end of a segment). Artificial segments based on wg size are not created.
    auto __result_end = __pattern_copy_if(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__assign_key2_wrapper>(__exec), __view3, __view4,
        [__binary_pred](const auto& __a) {
            // The size of key range for the (i-1) view is one less, so for the 0th index we do not check the keys
            // for (i-1), but we still need to get its key value as it is the start of a segment
            if (::std::get<0>(__a) == 0)
                return true;
            return !__binary_pred(::std::get<1>(__a), ::std::get<2>(__a)); // keys comparison
        },
        unseq_backend::__brick_assign_key_position{});

    //reduce by segment
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce2_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        unseq_backend::__brick_reduce_idx<_BinaryOperator, decltype(__intermediate_result_end)>(
            __binary_op, __intermediate_result_end),
        __result_end,
        oneapi::dpl::__ranges::take_view_simple(experimental::ranges::views::all_read(__idx), __result_end),
        experimental::ranges::views::all_read(__tmp_out_values), ::std::forward<_Range4>(__out_values))
        .wait();

    return __result_end;
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_HETERO_H
