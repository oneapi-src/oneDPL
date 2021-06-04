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

#ifndef _ONEDPL_algorithm_ranges_impl_hetero_H
#define _ONEDPL_algorithm_ranges_impl_hetero_H

#include "../algorithm_fwd.h"
#include "../parallel_backend.h"

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

template <int _N = 0, typename _ExecutionPolicy, typename _Function, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk_n(_ExecutionPolicy&& __exec, _Function __f, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    if (__n > 0)
    {
        using __new_name = oneapi::dpl::__par_backend_hetero::__new_kernel_name<_ExecutionPolicy, _N>;
        auto __new_exec =
            oneapi::dpl::execution::make_hetero_policy<__new_name>(::std::forward<_ExecutionPolicy>(__exec));
        oneapi::dpl::__par_backend_hetero::__parallel_for(__new_exec,
                                                          unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
                                                          ::std::forward<_Ranges>(__rngs)...)
            .wait();
    }
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
        const bool __res = __pattern_equal(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
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
            ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2), __pred);
        return __res ? 0 : __rng1.size();
    }

    using _Predicate = unseq_backend::multiple_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range1>;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<
            oneapi::dpl::__par_backend_hetero::__find_policy_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
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

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
               ::std::forward<_ExecutionPolicy>(__exec),
               unseq_backend::transform_init<_ExecutionPolicy, decltype(__identity_reduce_fn),
                                             decltype(__identity_init_fn)>{__identity_reduce_fn, __identity_init_fn},
               __identity_reduce_fn,
               unseq_backend::reduce<_ExecutionPolicy, decltype(__identity_reduce_fn), _ReduceValueType>{
                   __identity_reduce_fn},
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
    using _InitType = unseq_backend::__scan_no_init<_SizeType>;
    using _DataAcc = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _Assigner __assign_op;
    _ReduceOp __reduce_op;
    _DataAcc __get_data_op;
    _MaskAssigner __add_mask_op;

    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, int32_t> __mask_buf(__exec,
                                                                                                  __rng1.size());

    auto __res =
        __par_backend_hetero::__parallel_transform_scan(
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

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Predicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range2>>
__pattern_copy_if(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Predicate __pred)
{
    using _SizeType = decltype(__rng1.size());
    using _ReduceOp = ::std::plus<_SizeType>;

    unseq_backend::__create_mask<_Predicate, _SizeType> __create_mask_op{__pred};
    unseq_backend::__copy_by_mask<_ReduceOp, /*inclusive*/ ::std::true_type, 1> __copy_by_mask_op;

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

    auto __copy_last_id = __pattern_copy_if(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                                            __copy_rng, __not_pred<_Predicate>{__pred});
    auto __copy_rng_truncated = __copy_rng | oneapi::dpl::experimental::ranges::views::take(__copy_last_id);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(::std::forward<_ExecutionPolicy>(__exec),
                                                        oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{},
                                                        __copy_rng_truncated, ::std::forward<_Range>(__rng));

    return __copy_last_id;
}

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range2>>
__pattern_unique_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _BinaryPredicate __pred)
{
    using _It1DifferenceType = oneapi::dpl::__internal::__difference_t<_Range1>;
    unseq_backend::__copy_by_mask<::std::plus<_It1DifferenceType>, /*inclusive*/ ::std::true_type, 1> __copy_by_mask_op;
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
    auto res =
        __pattern_unique_copy(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), res_rng, __pred);

    __pattern_walk_n(::std::forward<_ExecutionPolicy>(__exec), __brick_copy<_ExecutionPolicy>{}, res_rng,
                     ::std::forward<_Range>(__rng));
    return res;
}

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------

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
            ::std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{},
            ::std::forward<_Range2>(__rng2), ::std::forward<_Range3>(__rng3));
    }
    else if (__n2 == 0)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            ::std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{},
            ::std::forward<_Range1>(__rng1), ::std::forward<_Range3>(__rng3));
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

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    if (__rng.size() >= 2)
        __par_backend_hetero::__parallel_stable_sort(::std::forward<_ExecutionPolicy>(__exec),
                                                     ::std::forward<_Range>(__rng), __comp)
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
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
            ::std::forward<_ExecutionPolicy>(__exec),
            unseq_backend::transform_init<_ExecutionPolicy, decltype(__identity_reduce_fn),
                                          decltype(__identity_init_fn)>{__identity_reduce_fn, __identity_init_fn},
            __identity_reduce_fn,
            unseq_backend::reduce<_ExecutionPolicy, decltype(__identity_reduce_fn), _ReduceValueType>{
                __identity_reduce_fn},
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
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
            ::std::forward<_ExecutionPolicy>(__exec),
            unseq_backend::transform_init<_ExecutionPolicy, __identity_reduce_fn<_Compare>,
                                          decltype(__identity_init_fn)>{__identity_reduce_fn<_Compare>{__comp},
                                                                        __identity_init_fn},
            __identity_reduce_fn<_Compare>{__comp},
            unseq_backend::reduce<_ExecutionPolicy, __identity_reduce_fn<_Compare>, _ReduceValueType>{
                __identity_reduce_fn<_Compare>{__comp}},
            ::std::forward<_Range>(__rng))
            .get();

    using ::std::get;
    return ::std::make_pair(get<0>(__ret), get<1>(__ret));
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_algorithm_ranges_impl_hetero_H */
