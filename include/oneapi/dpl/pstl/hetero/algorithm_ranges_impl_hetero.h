// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#if _PSTL_BACKEND_SYCL
#    include "dpcpp/utils_ranges_sycl.h"
#    include "dpcpp/unseq_backend_sycl.h"
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
// walk1
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk1(_ExecutionPolicy&& __exec, _Range&& __rng, _Function __f)
{
    if (!__rng.empty())
        oneapi::dpl::__par_backend_hetero::__ranges::__parallel_for(
            ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f},
            __rng.size(), ::std::forward<_Range>(__rng));
}

//------------------------------------------------------------------------
// walk2
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk2(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Function __f)
{
    if (!__rng1.empty() && !__rng2.empty())
        oneapi::dpl::__par_backend_hetero::__ranges::__parallel_for(
            ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f},
            __rng1.size(), ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2));
}

//------------------------------------------------------------------------
// walk3
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk3(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Function __f)
{
    if (!__rng1.empty() && !__rng2.empty() && !__rng3.empty())
        oneapi::dpl::__par_backend_hetero::__ranges::__parallel_for(
            ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f},
            __rng1.size(), ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2),
            ::std::forward<_Range3>(__rng3));
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

template <typename _ExecutionPolicy, typename _Range, typename _Size, typename _Tp, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_search_n(_ExecutionPolicy&& __exec, _Range&& __rng, _Size __count, const _Tp& __value,
                   _BinaryPredicate __pred)
{
    using namespace oneapi::dpl::experimental::ranges;
    //TODO: To consider definition a kind of special factory "multiple_view" (addition to standard "single_view").
    //The factory "multiple_view" would generate a range of N identical values.
    auto __s_rng = views::iota(0, __count) | views::transform([__value](auto) { return __value; });

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

    auto __ret_idx = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_ExecutionPolicy, decltype(__identity_reduce_fn), decltype(__identity_init_fn)>{
            __identity_reduce_fn, __identity_init_fn},
        __identity_reduce_fn,
        unseq_backend::reduce<_ExecutionPolicy, decltype(__identity_reduce_fn), _ReduceValueType>{__identity_reduce_fn},
        ::std::forward<_Range>(__rng));

    using ::std::get;
    return get<0>(__ret_idx);
}

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

    _ReduceValueType __ret = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_ExecutionPolicy, __identity_reduce_fn<_Compare>, decltype(__identity_init_fn)>{
            __identity_reduce_fn<_Compare>{__comp}, __identity_init_fn},
        __identity_reduce_fn<_Compare>{__comp},
        unseq_backend::reduce<_ExecutionPolicy, __identity_reduce_fn<_Compare>, _ReduceValueType>{
            __identity_reduce_fn<_Compare>{__comp}},
        ::std::forward<_Range>(__rng));

    using ::std::get;
    return ::std::make_pair(get<0>(__ret), get<1>(__ret));
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_algorithm_ranges_impl_hetero_H */
