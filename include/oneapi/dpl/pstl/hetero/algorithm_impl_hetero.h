// -*- C++ -*-
//===-- algorithm_impl_hetero.h -------------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#ifndef _ONEDPL_algorithm_impl_hetero_H
#define _ONEDPL_algorithm_impl_hetero_H

#include "../algorithm_fwd.h"

#include "../parallel_backend.h"
#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/execution_sycl_defs.h"
#    include "dpcpp/parallel_backend_sycl_utils.h"
#    include "dpcpp/unseq_backend_sycl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// walk1
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk1(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f,
                /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return;

    auto __keep =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    oneapi::dpl::__par_backend_hetero::__parallel_for(__exec, unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f},
                                                      __n, __buf.all_view())
        .wait();
}

//------------------------------------------------------------------------
// walk1_n
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Size, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator>
__pattern_walk1_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, _Function __f,
                  /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    __pattern_walk1(::std::forward<_ExecutionPolicy>(__exec), __first, __first + __n, __f,
                    /*vector=*/::std::true_type(), /*parallel=*/::std::true_type());
    return __first + __n;
}

//------------------------------------------------------------------------
// walk2
//------------------------------------------------------------------------

// TODO: A tag _IsSync is used for provide a patterns call pipeline, where the last one should be synchronous
// Probably it should be re-designed by a pipeline approach, when a patern returns some sync obejects
// and ones are combined into a "pipeline" (probably like Range pipeline)
template <typename _IsSync = ::std::true_type,
          __par_backend_hetero::access_mode __acc_mode1 = __par_backend_hetero::access_mode::read,
          __par_backend_hetero::access_mode __acc_mode2 = __par_backend_hetero::access_mode::write,
          typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator2>
__pattern_walk2(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                _ForwardIterator2 __first2, _Function __f, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return __first2;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode1, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);

    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode2, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf1.all_view(), __buf2.all_view());
    oneapi::dpl::__internal::__invoke_if(_IsSync(), [&__future_obj]() { __future_obj.wait(); });

    return __first2 + __n;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _Size, typename _ForwardIterator2,
          typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator2>
__pattern_walk2_n(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2,
                  _Function __f, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    return __pattern_walk2(::std::forward<_ExecutionPolicy>(__exec), __first1, __first1 + __n, __first2, __f,
                           ::std::true_type(), ::std::true_type());
}

//------------------------------------------------------------------------
// swap
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator2>
__pattern_swap(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
               _ForwardIterator2 __first2, _Function __f, /*is_vector=*/::std::true_type,
               /*is_parallel=*/::std::true_type)
{
    return __pattern_walk2</*_IsSync=*/::std::true_type, __par_backend_hetero::access_mode::read_write,
                           __par_backend_hetero::access_mode::read_write>(::std::forward<_ExecutionPolicy>(__exec),
                                                                          __first1, __last1, __first2, __f,
                                                                          ::std::true_type(), ::std::true_type());
}

//------------------------------------------------------------------------
// walk3
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _ForwardIterator3,
          typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator3>
__pattern_walk3(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f, /*vector=*/::std::true_type,
                /*parallel=*/::std::true_type)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return __first3;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);
    auto __keep3 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator3>();
    auto __buf3 = __keep3(__first3, __first3 + __n);

    oneapi::dpl::__par_backend_hetero::__parallel_for(::std::forward<_ExecutionPolicy>(__exec),
                                                      unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
                                                      __buf1.all_view(), __buf2.all_view(), __buf3.all_view())
        .wait();

    return __first3 + __n;
}

//------------------------------------------------------------------------
// walk_brick, walk_brick_n
//------------------------------------------------------------------------

template <typename _Name>
struct __walk_brick_wrapper
{
};

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk_brick(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f,
                     /*parallel=*/::std::true_type)
{
    if (__last - __first <= 0)
        return;

    __pattern_walk1(
        __par_backend_hetero::make_wrapped_policy<__walk_brick_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first, __last, __f,
        /*vector=*/::std::true_type{}, /*parallel=*/::std::true_type{});
}

template <typename _Name>
struct __walk_brick_n_wrapper
{
};

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Size, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator>
__pattern_walk_brick_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, _Function __f,
                       /*parallel=*/::std::true_type)
{
    __pattern_walk1(
        __par_backend_hetero::make_wrapped_policy<__walk_brick_n_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first, __first + __n, __f,
        /*vector=*/::std::true_type{}, /*parallel=*/::std::true_type{});
    return __first + __n;
}

//------------------------------------------------------------------------
// walk2_brick, walk2_brick_n
//------------------------------------------------------------------------

template <typename _Name>
struct __walk2_brick_wrapper
{
};

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Brick>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator2>
__pattern_walk2_brick(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                      _ForwardIterator2 __first2, _Brick __brick, /*parallel*/ ::std::true_type)
{
    return __pattern_walk2(
        __par_backend_hetero::make_wrapped_policy<__walk2_brick_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __last1, __first2, __brick,
        /*vector=*/::std::true_type{}, /*parallel*/ ::std::true_type{});
}

template <typename _Name>
struct __walk2_brick_n_wrapper
{
};

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _Size, typename _ForwardIterator2,
          typename _Brick>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator2>
__pattern_walk2_brick_n(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2,
                        _Brick __brick, /*parallel*/ ::std::true_type)
{

    return __pattern_walk2(
        __par_backend_hetero::make_wrapped_policy<__walk2_brick_n_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __first1 + __n, __first2, __brick,
        /*vector=*/::std::true_type{}, /*parallel*/ ::std::true_type{});
}

//------------------------------------------------------------------------
// fill
//------------------------------------------------------------------------

template <typename _SourceT>
struct fill_functor
{
    _SourceT __value;
    template <typename _TargetT>
    void
    operator()(_TargetT& __target) const
    {
        __target = __value;
    }
};

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _T>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator>
__pattern_fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _T& __value,
               /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    __pattern_walk1(::std::forward<_ExecutionPolicy>(__exec),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__first),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__last),
                    fill_functor<_T>{__value}, ::std::true_type{}, ::std::true_type{});
    return __last;
}

//------------------------------------------------------------------------
// generate
//------------------------------------------------------------------------

template <typename _Generator>
struct generate_functor
{
    _Generator __g;

    template <typename _TargetT>
    void
    operator()(_TargetT& value) const
    {
        value = __g();
    }
};

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Generator>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator>
__pattern_generate(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Generator __g,
                   /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    __pattern_walk1(::std::forward<_ExecutionPolicy>(__exec),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__first),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__last),
                    generate_functor<_Generator>{__g}, ::std::true_type{}, ::std::true_type{});
    return __last;
}

//------------------------------------------------------------------------
// brick_copy, brick_move
//------------------------------------------------------------------------

template <typename _ExecutionPolicy>
struct __brick_copy_n<_ExecutionPolicy,
                      oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>>
{
    template <typename _SourceT, typename _TargetT>
    void
    operator()(_SourceT&& __source, _TargetT&& __target) const
    {
        __target = ::std::forward<_SourceT>(__source);
    }
};

template <typename _ExecutionPolicy>
struct __brick_copy<_ExecutionPolicy,
                    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>>
{
    template <typename _SourceT, typename _TargetT>
    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
    operator()(_SourceT&& __source, _TargetT&& __target) const
    {
        __target = ::std::forward<_SourceT>(__source);
    }
};

template <typename _ExecutionPolicy>
struct __brick_move<_ExecutionPolicy,
                    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>>
{
    template <typename _SourceT, typename _TargetT>
    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
    operator()(_SourceT&& __source, _TargetT&& __target) const
    {
        __target = ::std::move(__source);
    }
};

template <typename _SourceT, typename _ExecutionPolicy>
struct __brick_fill<_SourceT, _ExecutionPolicy,
                    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>>
{
    _SourceT __value;
    template <typename _TargetT>
    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
    operator()(_TargetT& __target) const
    {
        __target = __value;
    }
};

template <typename _SourceT, typename _ExecutionPolicy>
struct __brick_fill_n<_SourceT, _ExecutionPolicy,
                      oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>>
{
    _SourceT __value;
    template <typename _TargetT>
    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
    operator()(_TargetT& __target) const
    {
        __target = __value;
    }
};

//------------------------------------------------------------------------
// min_element, max_element
//------------------------------------------------------------------------

template <typename _ReduceValueType>
struct __acc_handler_minelement
{

    template <typename _GlobalIdx, typename _Acc>
    _ReduceValueType
    operator()(_GlobalIdx __gidx, _Acc __acc) const
    {
        // TODO: __acc is tuple<accessors...> when zip_iterators are passed
        return _ReduceValueType{__gidx, __acc[__gidx]};
    }
};

template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_min_element(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp,
                      /*vector*/ ::std::true_type, /*parallel*/ ::std::true_type)
{
    if (__first == __last)
        return __last;

    using _IteratorValueType = typename ::std::iterator_traits<_Iterator>::value_type;
    using _IndexValueType =
        typename ::std::make_unsigned<typename ::std::iterator_traits<_Iterator>::difference_type>::type;
    using _ReduceValueType = tuple<_IndexValueType, _IteratorValueType>;

    auto __identity_init_fn = __acc_handler_minelement<_ReduceValueType>{};
    auto __identity_reduce_fn = [__comp](_ReduceValueType __a, _ReduceValueType __b) {
        using ::std::get;
        if (__comp(get<1>(__b), get<1>(__a)))
        {
            return __b;
        }
        return __a;
    };

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    auto __ret_idx = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_ExecutionPolicy, decltype(__identity_reduce_fn), decltype(__identity_init_fn)>{
            __identity_reduce_fn, __identity_init_fn},
        __identity_reduce_fn,
        unseq_backend::reduce<_ExecutionPolicy, decltype(__identity_reduce_fn), _ReduceValueType>{__identity_reduce_fn},
        __buf.all_view());

    return __first + ::std::get<0>(__ret_idx);
}

// TODO:
//   The following minmax_element implementation
//   has at worst 2N applications of the predicate
//   whereas the standard says about (3/2)N applications.
//
//   The issue is in the first reduce iteration which make N comparison instead of possible N/2.
//   It takes place due to the way we initialize buffer in transform stage:
//      each ReduceValueType consists of {min_element_index, max_element_index, min_element_value, max_element_value}
//      and in the initial stage `__identity_init_fn` we take the same buffer element as the min element and max element
//      Thus, in the first iteration we have N element buffer to make N comparisons (min and max for each two ReduceValueType's)
//
//   One of possible solution for it is to make initial reduce of each two elements
//   to get N/2 element buffer with ReduceValueType's
//   resulting in N/2 comparisons in the first iteration (one comparison with stride=2 for N)
//   Thus, there will be sum( N/2 + N/2 + N/4 + N/8 + ... ) or (N/2 + N) comparisons
//   However the solution requires use of custom pattern or substantial redesign of existing parallel_transform_reduce.
//

template <typename _ReduceValueType>
struct __acc_handler_minmaxelement
{

    template <typename _GlobalIdx, typename _Acc>
    _ReduceValueType
    operator()(_GlobalIdx __gidx, _Acc __acc) const
    {
        // TODO: Doesn't work with `zip_iterator`.
        //       In that case the first and the second arguments of `_ReduceValueType` will be
        //       a `tuple` of `difference_type`, not the `difference_type` itself.
        return _ReduceValueType{__gidx, __gidx, __acc[__gidx], __acc[__gidx]};
    }
};

template <typename _Compare>
struct __identity_reduce_fn
{
    _Compare __comp;
    template <typename _ReduceValueType>
    _ReduceValueType
    operator()(_ReduceValueType __a, _ReduceValueType __b) const
    {
        using ::std::get;
        auto __chosen_for_min = __a;
        auto __chosen_for_max = __b;
        // if b "<" a or if b "==" a and b_index < a_index
        if (__comp(get<2>(__b), get<2>(__a)) || (!__comp(get<2>(__a), get<2>(__b)) && get<0>(__b) < get<0>(__a)))
            __chosen_for_min = __b;
        // if a ">" b or if a "==" b and a_index > b_index
        if (__comp(get<3>(__b), get<3>(__a)) || (!__comp(get<3>(__a), get<3>(__b)) && get<1>(__b) < get<1>(__a)))
            __chosen_for_max = __a;
        auto __result = _ReduceValueType{get<0>(__chosen_for_min), get<1>(__chosen_for_max), get<2>(__chosen_for_min),
                                         get<3>(__chosen_for_max)};

        return __result;
    }
};

template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, ::std::pair<_Iterator, _Iterator>>
__pattern_minmax_element(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp,
                         /*vector*/ ::std::true_type, /*parallel*/ ::std::true_type)
{
    if (__first == __last)
        return ::std::make_pair(__first, __first);

    using _IteratorValueType = typename ::std::iterator_traits<_Iterator>::value_type;
    using _IndexValueType =
        typename ::std::make_unsigned<typename ::std::iterator_traits<_Iterator>::difference_type>::type;
    using _ReduceValueType = ::std::tuple<_IndexValueType, _IndexValueType, _IteratorValueType, _IteratorValueType>;

    auto __identity_init_fn = __acc_handler_minmaxelement<_ReduceValueType>{};

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    _ReduceValueType __ret = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType, /*__grainsize=*/ 8>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_ExecutionPolicy, __identity_reduce_fn<_Compare>, decltype(__identity_init_fn)>{
            __identity_reduce_fn<_Compare>{__comp}, __identity_init_fn},
        __identity_reduce_fn<_Compare>{__comp},
        unseq_backend::reduce<_ExecutionPolicy, __identity_reduce_fn<_Compare>, _ReduceValueType>{
            __identity_reduce_fn<_Compare>{__comp}},
        __buf.all_view());
    return ::std::make_pair<_Iterator, _Iterator>(__first + ::std::get<0>(__ret), __first + ::std::get<1>(__ret));
}

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------

template <typename _Predicate>
struct adjacent_find_fn
{
    _Predicate __predicate;

    // the functor is being used instead of a lambda because
    // at this level we don't know what type we get during zip_iterator unpack
    // whereas lambdas with auto in arg parameters are supported since C++14
    template <typename _Pack>
    bool
    operator()(const _Pack& __packed_neighbor_values) const
    {
        using ::std::get;
        return __predicate(get<0>(__packed_neighbor_values), get<1>(__packed_neighbor_values));
    }
};

template <typename _ExecutionPolicy, typename _Iterator, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_adjacent_find(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _BinaryPredicate __predicate,
                        /*parallel*/ ::std::true_type, /*vector*/ ::std::true_type,
                        oneapi::dpl::__internal::__or_semantic)
{
    if (__last - __first < 2)
        return __last;

    using _Predicate =
        oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, adjacent_find_fn<_BinaryPredicate>>;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf1 = __keep1(__first, __last - 1);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf2 = __keep2(__first + 1, __last);

    // TODO: in case of confilicting names
    // __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__or_policy_wrapper>()
    bool result = __par_backend_hetero::__parallel_find_or(
        ::std::forward<_ExecutionPolicy>(__exec), _Predicate{adjacent_find_fn<_BinaryPredicate>{__predicate}},
        __par_backend_hetero::__parallel_or_tag{},
        oneapi::dpl::__ranges::make_zip_view(__buf1.all_view(), __buf2.all_view()));

    // inverted conditional because of
    // reorder_predicate in glue_algorithm_impl.h
    return result ? __first : __last;
}

template <typename _ExecutionPolicy, typename _Iterator, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_adjacent_find(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _BinaryPredicate __predicate,
                        /*parallel*/ ::std::true_type, /*vector*/ ::std::true_type,
                        oneapi::dpl::__internal::__first_semantic)
{
    if (__last - __first < 2)
        return __last;

    using _Predicate =
        oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, adjacent_find_fn<_BinaryPredicate>>;

    auto __result = __par_backend_hetero::__parallel_find(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::zip(
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first + 1)),
        __par_backend_hetero::zip(
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last - 1),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last)),
        _Predicate{adjacent_find_fn<_BinaryPredicate>{__predicate}}, ::std::true_type{});

    auto __zip_at_first = __par_backend_hetero::zip(
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first + 1));
    _Iterator __result_iterator = __first + (__result - __zip_at_first);
    return (__result_iterator == __last - 1) ? __last : __result_iterator;
}

//------------------------------------------------------------------------
// count, count_if
//------------------------------------------------------------------------

template <typename _Predicate>
struct acc_handler_count
{
    _Predicate __predicate;

    // int is being implicitly casted to difference_type
    // otherwise we can only pass the difference_type as a functor template parameter
    template <typename _Acc, typename _GlobalIdx>
    int
    operator()(_GlobalIdx gidx, _Acc acc) const
    {
        return (__predicate(acc[gidx]) ? 1 : 0);
    }
};

template <typename _ExecutionPolicy, typename _Iterator, typename _Predicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<
    _ExecutionPolicy, typename ::std::iterator_traits<_Iterator>::difference_type>
__pattern_count(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Predicate __predicate,
                /*parallel*/ ::std::true_type, /*vector*/ ::std::true_type)
{
    if (__first == __last)
        return 0;

    using _ReduceValueType = typename ::std::iterator_traits<_Iterator>::difference_type;

    auto __identity_init_fn = acc_handler_count<_Predicate>{__predicate};
    auto __identity_reduce_fn = ::std::plus<_ReduceValueType>{};

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_ExecutionPolicy, decltype(__identity_reduce_fn), decltype(__identity_init_fn)>{
            __identity_reduce_fn, __identity_init_fn},
        __identity_reduce_fn,
        unseq_backend::reduce<_ExecutionPolicy, decltype(__identity_reduce_fn), _ReduceValueType>{__identity_reduce_fn},
        __buf.all_view());
}

//------------------------------------------------------------------------
// any_of
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_any_of(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Pred __pred,
                 /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__first == __last)
        return false;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, _Pred>;

    return __par_backend_hetero::__parallel_or(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last), _Predicate{__pred});
}

//------------------------------------------------------------------------
// equal
//------------------------------------------------------------------------

template <typename _Pred>
struct equal_predicate
{
    _Pred __pred;

    template <typename _Value>
    bool
    operator()(const _Value& __val) const
    {
        using ::std::get;
        return !__pred(get<0>(__val), get<1>(__val));
    }
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_equal(_ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2,
                _Iterator2 __last2, _Pred __pred,
                /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__last1 == __first1 || __last2 == __first2 || __last1 - __first1 != __last2 - __first2)
        return false;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, equal_predicate<_Pred>>;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __buf2 = __keep2(__first2, __last2);

    // TODO: in case of confilicting names
    // __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__or_policy_wrapper>()
    return !__par_backend_hetero::__parallel_find_or(
        ::std::forward<_ExecutionPolicy>(__exec), _Predicate{equal_predicate<_Pred>{__pred}},
        __par_backend_hetero::__parallel_or_tag{},
        oneapi::dpl::__ranges::make_zip_view(__buf1.all_view(), __buf2.all_view()));
}

//------------------------------------------------------------------------
// equal version for sequences with equal length
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_equal(_ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2, _Pred __pred,
                /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    return oneapi::dpl::__internal::__pattern_equal(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
                                                    __first2, __first2 + (__last1 - __first1), __pred,
                                                    /*vector=*/::std::true_type{}, /*parallel=*/::std::true_type{});
}

//------------------------------------------------------------------------
// find_if
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_find_if(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Pred __pred,
                  /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__first == __last)
        return __last;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, _Pred>;

    return __par_backend_hetero::__parallel_find(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last), _Predicate{__pred},
        ::std::true_type{});
}

//------------------------------------------------------------------------
// find_end
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator1>
__pattern_find_end(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
                   _Iterator2 __s_last, _Pred __pred, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__first == __last || __s_last == __s_first || __last - __first < __s_last - __s_first)
        return __last;

    if (__last - __first == __s_last - __s_first)
    {
        const bool __res = __pattern_equal(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __s_first, __pred,
                                           ::std::true_type(), ::std::true_type());
        return __res ? __first : __last;
    }
    else
    {
        using _Predicate = unseq_backend::multiple_match_pred<_ExecutionPolicy, _Pred>;

        return __par_backend_hetero::__parallel_find(
            ::std::forward<_ExecutionPolicy>(__exec),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__s_first),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__s_last), _Predicate{__pred},
            ::std::false_type{});
    }
}

//------------------------------------------------------------------------
// find_first_of
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator1>
__pattern_find_first_of(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
                        _Iterator2 __s_last, _Pred __pred, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__first == __last || __s_last == __s_first)
        return __last;

    using _Predicate = unseq_backend::first_match_pred<_ExecutionPolicy, _Pred>;

    // TODO: To check whether it makes sense to iterate over the second sequence in case of
    // distance(__first, __last) < distance(__s_first, __s_last).
    return __par_backend_hetero::__parallel_find(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__s_first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__s_last), _Predicate{__pred},
        ::std::true_type{});
}

//------------------------------------------------------------------------
// search
//------------------------------------------------------------------------

template <typename Name>
class equal_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator1>
__pattern_search(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
                 _Iterator2 __s_last, _Pred __pred, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__s_last == __s_first)
        return __first;

    if (__last - __first < __s_last - __s_first)
        return __last;

    if (__last - __first == __s_last - __s_first)
    {
        const bool __res = __pattern_equal(
            __par_backend_hetero::make_wrapped_policy<equal_wrapper>(::std::forward<_ExecutionPolicy>(__exec)), __first,
            __last, __s_first, __pred, ::std::true_type(), ::std::true_type());
        return __res ? __first : __last;
    }

    using _Predicate = unseq_backend::multiple_match_pred<_ExecutionPolicy, _Pred>;
    return __par_backend_hetero::__parallel_find(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__s_first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__s_last), _Predicate{__pred},
        ::std::true_type{});
}

//------------------------------------------------------------------------
// search_n
//------------------------------------------------------------------------

template <typename _Tp, typename _Pred>
struct __search_n_unary_predicate
{
    _Tp __value_;
    _Pred __pred_;

    template <typename _Value>
    bool
    operator()(const _Value& __val) const
    {
        return !__pred_(__val, __value_);
    }
};

template <typename _ExecutionPolicy, typename _Iterator, typename _Size, typename _Tp, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_search_n(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Size __count, const _Tp& __value,
                   _BinaryPredicate __pred, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__count <= 0)
        return __first;

    if (__last - __first < __count)
        return __last;

    if (__last - __first == __count)
    {
        return (!__internal::__pattern_any_of(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                              __search_n_unary_predicate<_Tp, _BinaryPredicate>{__value, __pred},
                                              ::std::true_type{}, ::std::true_type{}))
                   ? __first
                   : __last;
    }

    using _Predicate = unseq_backend::n_elem_match_pred<_ExecutionPolicy, _BinaryPredicate, _Tp, _Size>;
    return __par_backend_hetero::__parallel_find(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last),
        _Predicate{__pred, __value, __count}, ::std::true_type{});
}

//------------------------------------------------------------------------
// mismatch
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, ::std::pair<_Iterator1, _Iterator2>>
__pattern_mismatch(_ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2,
                   _Iterator2 __last2, _Pred __pred, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    auto __n = ::std::min(__last1 - __first1, __last2 - __first2);
    if (__n <= 0)
        return ::std::make_pair(__first1, __first2);

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, equal_predicate<_Pred>>;

    auto __first_zip = __par_backend_hetero::zip(
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first1),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first2));
    auto __result =
        __par_backend_hetero::__parallel_find(::std::forward<_ExecutionPolicy>(__exec), __first_zip, __first_zip + __n,
                                              _Predicate{equal_predicate<_Pred>{__pred}}, ::std::true_type{});
    __n = __result - __first_zip;
    return ::std::make_pair(__first1 + __n, __first2 + __n);
}

//------------------------------------------------------------------------
// copy_if
//------------------------------------------------------------------------

// create mask
template <typename _Pred, typename _Tp>
struct __create_mask
{
    _Pred __pred;

    template <typename _Idx, typename _Input>
    _Tp
    operator()(const _Idx __idx, const _Input& __input) const
    {
        using ::std::get;
        // 1. apply __pred
        auto __temp = __pred(get<0>(__input[__idx]));
        // 2. initialize mask
        get<1>(__input[__idx]) = __temp;
        return _Tp(__temp);
    }
};

// get mask without predicate application
template <typename _Tp, const int N>
struct __get_mask
{
    template <typename _Idx, typename _Input>
    _Tp
    operator()(const _Idx __idx, const _Input& __input) const
    {
        using ::std::get;
        return _Tp(get<N>(__input[__idx]));
    }
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _IteratorOrTuple, typename _CreateMaskOp,
          typename _CopyByMaskOp>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<
    _ExecutionPolicy, ::std::pair<_IteratorOrTuple, typename ::std::iterator_traits<_Iterator1>::difference_type>>
__pattern_scan_copy(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _IteratorOrTuple __output_first,
                    _CreateMaskOp __create_mask_op, _CopyByMaskOp __copy_by_mask_op)
{
    using _It1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;
    using _ReduceOp = ::std::plus<_It1DifferenceType>;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _MaskAssigner = unseq_backend::__mask_assigner<1>;
    using _InitType = unseq_backend::__scan_no_init<_It1DifferenceType>;
    using _DataAcc = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    if (__first == __last)
        return ::std::make_pair(__output_first, _It1DifferenceType{0});

    _Assigner __assign_op;
    _ReduceOp __reduce_op;
    _DataAcc __get_data_op;
    _MaskAssigner __add_mask_op;

    // temporary buffer to store boolean mask
    auto __n = __last - __first;
    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, int32_t> __mask_buf(__exec, __n);

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _IteratorOrTuple>();
    auto __buf2 = __keep2(__output_first, __output_first + __n);

    auto __res = __par_backend_hetero::__parallel_transform_scan(
        ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::make_zip_view(
            __buf1.all_view(), oneapi::dpl::__ranges::all_view<int32_t, __par_backend_hetero::access_mode::read_write>(
                                   __mask_buf.get_buffer())),
        __buf2.all_view(), __reduce_op, _InitType{},
        // local scan
        unseq_backend::__scan</*inclusive*/ ::std::true_type, _ExecutionPolicy, _ReduceOp, _DataAcc, _Assigner,
                              _MaskAssigner, _CreateMaskOp, _InitType>{__reduce_op, __get_data_op, __assign_op,
                                                                       __add_mask_op, __create_mask_op},
        // scan between groups
        unseq_backend::__scan</*inclusive*/ ::std::true_type, _ExecutionPolicy, _ReduceOp, _DataAcc, _NoAssign,
                              _Assigner, _DataAcc, _InitType>{__reduce_op, __get_data_op, _NoAssign{}, __assign_op,
                                                              __get_data_op},
        // global scan
        __copy_by_mask_op);

    return ::std::make_pair(__output_first + __res.first, __res.second);
}

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Predicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator2>
__pattern_copy_if(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result_first,
                  _Predicate __pred, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    using _It1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;
    using _ReduceOp = ::std::plus<_It1DifferenceType>;

    __create_mask<_Predicate, _It1DifferenceType> __create_mask_op{__pred};
    unseq_backend::__copy_by_mask<_ReduceOp, /*inclusive*/ ::std::true_type, 1> __copy_by_mask_op;

    auto __result = __pattern_scan_copy(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result_first,
                                        __create_mask_op, __copy_by_mask_op);
    return __result_first + __result.second;
}

//------------------------------------------------------------------------
// partition_copy
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Iterator3,
          typename _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, ::std::pair<_Iterator2, _Iterator3>>
__pattern_partition_copy(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result1,
                         _Iterator3 __result2, _UnaryPredicate __pred, /*vector*/ ::std::true_type,
                         /*parallel*/ ::std::true_type)
{
    if (__first == __last)
        return ::std::make_pair(__result1, __result2);

    using _It1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;
    using _ReduceOp = ::std::plus<_It1DifferenceType>;

    __create_mask<_UnaryPredicate, _It1DifferenceType> __create_mask_op{__pred};
    unseq_backend::__partition_by_mask<_ReduceOp, /*inclusive*/ ::std::true_type> __copy_by_mask_op{_ReduceOp{}};

    auto __result = __pattern_scan_copy(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        __par_backend_hetero::zip(
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__result1),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__result2)),
        __create_mask_op, __copy_by_mask_op);

    return ::std::make_pair(__result1 + __result.second, __result2 + (__last - __first - __result.second));
}

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template <typename _Predicate, typename _ValueType>
struct __create_mask_unique_copy
{
    _Predicate __predicate;

    template <typename _Idx, typename _Acc>
    _ValueType
    operator()(_Idx __idx, _Acc& __acc) const
    {
        using ::std::get;

        auto __predicate_result = 1;
        if (__idx != 0)
            __predicate_result = __predicate(get<0>(__acc[__idx]), get<0>(__acc[__idx + (-1)]));

        get<1>(__acc[__idx]) = __predicate_result;
        return _ValueType{__predicate_result};
    }
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator2>
__pattern_unique_copy(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result_first,
                      _BinaryPredicate __pred, /*vector*/ ::std::true_type, /*parallel*/ ::std::true_type)
{
    using _It1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;
    unseq_backend::__copy_by_mask<::std::plus<_It1DifferenceType>, /*inclusive*/ ::std::true_type, 1> __copy_by_mask_op;
    __create_mask_unique_copy<__not_pred<_BinaryPredicate>, _It1DifferenceType> __create_mask_op{
        __not_pred<_BinaryPredicate>{__pred}};

    auto __result = __pattern_scan_copy(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result_first,
                                        __create_mask_op, __copy_by_mask_op);

    return __result_first + __result.second;
}

template <typename Name>
class copy_back_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator, typename _Predicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_remove_if(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Predicate __pred,
                    /*vector*/ ::std::true_type, /*parallel*/ ::std::true_type)
{
    if (__last == __first)
        return __last;

    using _ValueType = typename ::std::iterator_traits<_Iterator>::value_type;

    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __buf(__exec,
                                                                                                __last - __first);
    auto __copy_first = __buf.get();
    auto __copy_last = __pattern_copy_if(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __copy_first,
                                         __not_pred<_Predicate>{__pred},
                                         /*vector=*/::std::true_type{}, /*parallel*/ ::std::true_type{});

    //TODO: optimize copy back depending on Iterator, i.e. set_final_data for host iterator/pointer
    return __pattern_walk2(
        __par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __copy_first, __copy_last, __first, __brick_copy<_ExecutionPolicy>{}, ::std::true_type{}, ::std::true_type{});
}

template <typename _ExecutionPolicy, typename _Iterator, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_unique(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _BinaryPredicate __pred,
                 /*vector*/ ::std::true_type, /*parallel*/ ::std::true_type)
{
    if (__last - __first < 2)
        return __last;

    using _ValueType = typename ::std::iterator_traits<_Iterator>::value_type;

    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __buf(__exec,
                                                                                                __last - __first);
    auto __copy_first = __buf.get();
    auto __copy_last =
        __pattern_unique_copy(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __copy_first, __pred,
                              /*vector=*/::std::true_type{}, /*parallel*/ ::std::true_type{});

    //TODO: optimize copy back depending on Iterator, i.e. set_final_data for host iterator/pointer
    return __pattern_walk2</*_IsSync=*/::std::true_type, __par_backend_hetero::access_mode::read_write,
                           __par_backend_hetero::access_mode::read_write>(
        __par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __copy_first, __copy_last, __first, __brick_copy<_ExecutionPolicy>{}, ::std::true_type{}, ::std::true_type{});
}

//------------------------------------------------------------------------
// is_partitioned
//------------------------------------------------------------------------

enum _IsPartitionedReduceType : signed char
{
    __broken,
    __all_true,
    __all_false,
    __true_false
};

template <typename _Predicate>
struct acc_handler_is_partitioned
{
    _Predicate __predicate;

    // int is being implicitly casted to difference_type
    // otherwise we can only pass the difference_type as a functor template parameter
    template <typename _Acc, typename _GlobalIdx>
    _IsPartitionedReduceType
    operator()(_GlobalIdx gidx, _Acc acc) const
    {
        return (__predicate(acc[gidx]) ? __all_true : __all_false);
    }
};

template <typename _ExecutionPolicy, typename _Iterator, typename _Predicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_is_partitioned(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Predicate __predicate,
                         /*parallel*/ ::std::true_type, /*vector*/ ::std::true_type)
{
    if (__last - __first < 2)
        return true;

    using _ReduceValueType = _IsPartitionedReduceType;

    auto __identity_init_fn = acc_handler_is_partitioned<_Predicate>{__predicate};
    auto __identity_reduce_fn = [](_ReduceValueType __val1, _ReduceValueType __val2) -> _ReduceValueType {
        _ReduceValueType __table[] = {__broken,     __broken,     __broken,     __broken, __broken,    __all_true,
                                      __true_false, __true_false, __broken,     __broken, __all_false, __broken,
                                      __broken,     __broken,     __true_false, __broken};
        return __table[__val1 * 4 + __val2];
    };

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    _ReduceValueType __res = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_ExecutionPolicy, decltype(__identity_reduce_fn), decltype(__identity_init_fn)>{
            __identity_reduce_fn, __identity_init_fn},
        __identity_reduce_fn,
        unseq_backend::reduce<_ExecutionPolicy, decltype(__identity_reduce_fn), _ReduceValueType>{__identity_reduce_fn},
        __buf.all_view());

    return __broken != __identity_reduce_fn(_ReduceValueType{__all_true}, __res);
}

//------------------------------------------------------------------------
// is_heap / is_heap_until
//------------------------------------------------------------------------

template <class _Comp>
struct __is_heap_check
{
    mutable _Comp __comp_;

    template <class _Idx, class _Accessor>
    bool
    operator()(const _Idx __idx, const _Accessor& __acc) const
    {
        // Make sure that we have a signed integer here to avoid getting negative value when __idx == 0
        using _SignedIdx = typename ::std::make_signed<_Idx>::type;
        return __comp_(__acc[(static_cast<_SignedIdx>(__idx) - 1) / 2], __acc[__idx]);
    }
};

template <typename _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
__pattern_is_heap_until(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last,
                        _Compare __comp, /* vector */ ::std::true_type, /* parallel = */ ::std::true_type)
{
    if (__last - __first < 2)
        return __last;

    using _Predicate =
        oneapi::dpl::unseq_backend::single_match_pred_by_idx<_ExecutionPolicy, __is_heap_check<_Compare>>;

    return __par_backend_hetero::__parallel_find(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last), _Predicate{__comp},
        ::std::true_type{});
}

template <typename _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_is_heap(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last,
                  _Compare __comp, /* vector */ ::std::true_type, /* parallel = */ ::std::true_type)
{
    if (__last - __first < 2)
        return true;

    using _Predicate =
        oneapi::dpl::unseq_backend::single_match_pred_by_idx<_ExecutionPolicy, __is_heap_check<_Compare>>;

    return !__par_backend_hetero::__parallel_or(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last), _Predicate{__comp});
}

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Iterator3, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator3>
__pattern_merge(_ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2,
                _Iterator2 __last2, _Iterator3 __d_first, _Compare __comp, /*vector=*/::std::true_type,
                /*parallel=*/::std::true_type)
{
    auto __n1 = __last1 - __first1;
    auto __n2 = __last2 - __first2;
    auto __n = __n1 + __n2;
    if (__n == 0)
        return __d_first;

    //To consider the direct copying pattern call in case just one of sequences is empty.
    if (__n1 == 0)
        oneapi::dpl::__internal::__pattern_walk2_brick(
            ::std::forward<_ExecutionPolicy>(__exec), __first2, __last2, __d_first,
            oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::true_type());
    else if (__n2 == 0)
        oneapi::dpl::__internal::__pattern_walk2_brick(
            ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __d_first,
            oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::true_type());
    else
    {
        auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
        auto __buf1 = __keep1(__first1, __last1);
        auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
        auto __buf2 = __keep2(__first2, __last2);

        auto __keep3 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator3>();
        auto __buf3 = __keep3(__d_first, __d_first + __n);

        __par_backend_hetero::__parallel_merge(::std::forward<_ExecutionPolicy>(__exec), __buf1.all_view(),
                                               __buf2.all_view(), __buf3.all_view(), __comp)
            .wait();
    }
    return __d_first + __n;
}
//------------------------------------------------------------------------
// inplace_merge
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_inplace_merge(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __middle, _Iterator __last,
                        _Compare __comp, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    using _ValueType = typename ::std::iterator_traits<_Iterator>::value_type;

    if (__first == __middle || __middle == __last || __first == __last)
        return;

    assert(__first < __middle && __middle < __last);

    auto __n = __last - __first;
    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __buf(__exec, __n);
    auto __copy_first = __buf.get();
    auto __copy_last = __copy_first + __n;

    __pattern_merge(::std::forward<_ExecutionPolicy>(__exec),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__middle),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__middle),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__copy_first),
                    __comp, ::std::true_type{}, ::std::true_type{});

    //TODO: optimize copy back depending on Iterator, i.e. set_final_data for host iterator/pointer
    __pattern_walk2(
        __par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __copy_first, __copy_last, __first, __brick_move<_ExecutionPolicy>{}, ::std::true_type{}, ::std::true_type{});
}

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp,
               /*vector=*/::std::true_type, /*parallel=*/::std::true_type, /*is_move_constructible=*/::std::true_type)
{
    if (__last - __first < 2)
        return;

    __par_backend_hetero::__parallel_stable_sort(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__last), __comp)
        .wait();
}

//------------------------------------------------------------------------
// stable_sort
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_stable_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp,
                      /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__last - __first < 2)
        return;

    __par_backend_hetero::__parallel_stable_sort(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__last), __comp)
        .wait();
}

template <typename Name>
class copy_back_wrapper2
{
};

template <typename _ExecutionPolicy, typename _Iterator, typename _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_stable_partition(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _UnaryPredicate __pred,
                           /*vector*/ ::std::true_type, /*parallel*/ ::std::true_type)
{
    if (__last == __first)
        return __last;
    else if (__last - __first < 2)
        return __pattern_any_of(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __pred, ::std::true_type(),
                                ::std::true_type())
                   ? __last
                   : __first;

    using _ValueType = typename ::std::iterator_traits<_Iterator>::value_type;

    auto __n = __last - __first;

    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __true_buf(__exec, __n);
    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __false_buf(__exec, __n);
    auto __true_result = __true_buf.get();
    auto __false_result = __false_buf.get();

    auto copy_result = __pattern_partition_copy(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                                __true_result, __false_result, __pred,
                                                /*vector=*/::std::true_type{}, /*parallel*/ ::std::true_type{});
    auto true_count = copy_result.first - __true_result;

    //TODO: optimize copy back if possible (inplace, decrease number of submits)
    __pattern_walk2</*_IsSync=*/::std::false_type>(
        __par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __true_result, copy_result.first, __first, __brick_move<_ExecutionPolicy>{}, ::std::true_type{},
        ::std::true_type{});
    __pattern_walk2(
        __par_backend_hetero::make_wrapped_policy<copy_back_wrapper2>(::std::forward<_ExecutionPolicy>(__exec)),
        __false_result, copy_result.second, __first + true_count, __brick_move<_ExecutionPolicy>{}, ::std::true_type{},
        ::std::true_type{});

    return __first + true_count;
}

template <typename _ExecutionPolicy, typename _Iterator, typename _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_partition(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _UnaryPredicate __pred,
                    /*vector*/ ::std::true_type, /*parallel*/ ::std::true_type)
{
    //TODO: consider nonstable aproaches
    return __pattern_stable_partition(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __pred,
                                      ::std::true_type(), ::std::true_type());
}

//------------------------------------------------------------------------
// lexicographical_compare
//------------------------------------------------------------------------

template <typename _Predicate, typename _ReduceValueType>
struct acc_handler_lexicographical_compare
{
    _Predicate __predicate;

    template <typename _GlobalIdx, typename _Acc1, typename _Acc2>
    _ReduceValueType
    operator()(_GlobalIdx __gidx, _Acc1 __acc1, _Acc2 __acc2) const
    {
        auto __s1_val = __acc1[__gidx];
        auto __s2_val = __acc2[__gidx];

        int32_t __is_s1_val_less = __predicate(__s1_val, __s2_val);
        int32_t __is_s1_val_greater = __predicate(__s2_val, __s1_val);

        // 1 if __s1_val <  __s2_val, -1 if __s1_val <  __s2_val, 0 if __s1_val == __s2_val
        return _ReduceValueType{1 * __is_s1_val_less - 1 * __is_s1_val_greater};
    }
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_lexicographical_compare(_ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1,
                                  _Iterator2 __first2, _Iterator2 __last2, _Compare __comp, /*vector*/ ::std::true_type,
                                  /*parallel*/ ::std::true_type)
{
    //trivial pre-checks
    if (__first2 == __last2)
        return false;
    if (__first1 == __last1)
        return true;

    using _Iterator1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;
    using _ReduceValueType = int32_t;

    auto __identity_init_fn = acc_handler_lexicographical_compare<_Compare, _ReduceValueType>{__comp};
    auto __identity_reduce_fn = [](_ReduceValueType __a, _ReduceValueType __b) -> _ReduceValueType {
        bool __is_mismatched = __a != 0;
        return __a * __is_mismatched + __b * !__is_mismatched;
    };

    auto __shared_size = ::std::min(__last1 - __first1, (_Iterator1DifferenceType)(__last2 - __first2));

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first1, __first1 + __shared_size);

    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __shared_size);

    auto __ret_idx = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_ExecutionPolicy, decltype(__identity_reduce_fn), decltype(__identity_init_fn)>{
            __identity_reduce_fn, __identity_init_fn},
        __identity_reduce_fn,
        unseq_backend::reduce<_ExecutionPolicy, decltype(__identity_reduce_fn), _ReduceValueType>{__identity_reduce_fn},
        __buf1.all_view(), __buf2.all_view());

    return __ret_idx ? __ret_idx == 1 : (__last1 - __first1) < (__last2 - __first2);
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, bool>
__pattern_includes(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                   _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp, /*vector=*/::std::true_type,
                   /*parallel=*/::std::true_type)
{
    //according to the spec
    if (__first2 == __last2)
        return true;

    //optimization; {1} - the first sequence, {2} - the second sequence
    //{1} is empty or size_of{2} > size_of{1}
    if (__first1 == __last1 || __last2 - __first2 > __last1 - __first1)
        return false;

    typedef typename ::std::iterator_traits<_ForwardIterator1>::difference_type _Size1;
    typedef typename ::std::iterator_traits<_ForwardIterator2>::difference_type _Size2;

    using __brick_include_type = unseq_backend::__brick_includes<_ExecutionPolicy, _Compare, _Size1, _Size2>;
    return !__par_backend_hetero::__parallel_or(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first2),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last2),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first1),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last1),
        __brick_include_type(__comp, __last1 - __first1, __last2 - __first2));
}

//------------------------------------------------------------------------
// partial_sort
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_partial_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last, _Compare __comp,
                       /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__last - __first < 2)
        return;

    __par_backend_hetero::__parallel_partial_sort(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__mid),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__last), __comp)
        .wait();
}

//------------------------------------------------------------------------
// partial_sort_copy
//------------------------------------------------------------------------

template <typename _KernelName>
struct __initial_copy_1
{
};

template <typename _KernelName>
struct __initial_copy_2
{
};

template <typename _KernelName>
struct __copy_back
{
};

template <typename _KernelName>
struct __partial_sort_1
{
};

template <typename _KernelName>
struct __partial_sort_2
{
};

template <typename _ExecutionPolicy, typename _InIterator, typename _OutIterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _OutIterator>
__pattern_partial_sort_copy(_ExecutionPolicy&& __exec, _InIterator __first, _InIterator __last,
                            _OutIterator __out_first, _OutIterator __out_last, _Compare __comp,
                            /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    using _ValueType = typename ::std::iterator_traits<_InIterator>::value_type;

    auto __in_size = __last - __first;
    auto __out_size = __out_last - __out_first;

    if (__in_size == 0 || __out_size == 0)
        return __out_first;

    // TODO: we can avoid a separate __pattern_walk2 for initial copy: it can be done during sort itself
    // like it's done for CPU version, but it's better to be done together with merge cutoff implmenetation
    // as it uses a similar mechanism.
    if (__in_size <= __out_size)
    {
        // If our output buffer is larger than the input buffer, simply copy elements to the output and use
        // full sort on them.
        auto __out_end = __pattern_walk2</*_IsSync=*/::std::false_type>(
            __par_backend_hetero::make_wrapped_policy<__initial_copy_1>(::std::forward<_ExecutionPolicy>(__exec)),
            __first, __last, __out_first, __brick_copy<_ExecutionPolicy>{}, ::std::true_type{}, ::std::true_type{});

        // Use reqular sort as partial_sort isn't required to be stable
        __pattern_sort(
            __par_backend_hetero::make_wrapped_policy<__partial_sort_1>(::std::forward<_ExecutionPolicy>(__exec)),
            __out_first, __out_end, __comp, ::std::true_type{}, ::std::true_type{}, ::std::true_type{});

        return __out_end;
    }
    else
    {
        // If our input buffer is smaller than the input bufer do the following:
        // - create a temporary buffer and copy all the elements from the input buffer there
        // - run partial sort on the temporary buffer
        // - copy k elements from the temporary buffer to the output buffer.
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __buf(__exec, __in_size);

        auto __buf_first = __buf.get();
        auto __buf_last = __pattern_walk2</*_IsSync=*/::std::false_type>(
            __par_backend_hetero::make_wrapped_policy<__initial_copy_2>(::std::forward<_ExecutionPolicy>(__exec)),
            __first, __last, __buf_first, __brick_copy<_ExecutionPolicy>{}, ::std::true_type{}, ::std::true_type{});

        auto __buf_mid = __buf_first + __out_size;

        __par_backend_hetero::__parallel_partial_sort(
            __par_backend_hetero::make_wrapped_policy<__partial_sort_2>(::std::forward<_ExecutionPolicy>(__exec)),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__buf_first),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__buf_mid),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__buf_last), __comp);

        return __pattern_walk2(
            __par_backend_hetero::make_wrapped_policy<__copy_back>(::std::forward<_ExecutionPolicy>(__exec)),
            __buf_first, __buf_mid, __out_first, __brick_copy<_ExecutionPolicy>{}, ::std::true_type{},
            ::std::true_type{});
    }
}

//------------------------------------------------------------------------
// nth_element
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_nth_element(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __nth, _Iterator __last, _Compare __comp,
                      /*vector*/ ::std::true_type, /*parallel*/ ::std::true_type) noexcept
{
    if (__first == __last || __nth == __last)
        return;

    // TODO: check partition-based implementation
    // - try to avoid host dereference issue
    // - measure performance of the issue-free implementation
    __pattern_partial_sort(::std::forward<_ExecutionPolicy>(__exec), __first, __nth + 1, __last, __comp,
                           /*vector*/ ::std::true_type{}, /*parallel*/ ::std::true_type{});
}

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_reverse(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, /*vector=*/::std::true_type,
                  /*parallel=*/::std::true_type)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::__reverse_functor<typename ::std::iterator_traits<_Iterator>::difference_type>{__n}, __n / 2,
        __buf.all_view())
        .wait();
}

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _BidirectionalIterator, typename _ForwardIterator>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator>
__pattern_reverse_copy(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __last,
                       _ForwardIterator __result, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return __result;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _BidirectionalIterator>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator>();
    auto __buf2 = __keep2(__result, __result + __n);
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::__reverse_copy<typename ::std::iterator_traits<_BidirectionalIterator>::difference_type>{__n},
        __n, __buf1.all_view(), __buf2.all_view())
        .wait();

    return __result + __n;
}

//------------------------------------------------------------------------
// rotate
//------------------------------------------------------------------------
//Advantages over "3x reverse" version of algorithm:
//1:Not sensitive to size of shift
//  (With 3x reverse was large variance)
//2:The average time is better until ~10e8 elements
//Wrapper needed to avoid kernel problems
template <typename Name>
class __rotate_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator>
__pattern_rotate(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __new_first, _Iterator __last,
                 /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return __first;

    using _Tp = typename ::std::iterator_traits<_Iterator>::value_type;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);
    auto __temp_buf = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _Tp>(__exec, __n);

    auto __temp_rng =
        oneapi::dpl::__ranges::all_view<_Tp, __par_backend_hetero::access_mode::write>(__temp_buf.get_buffer());

    const auto __shift = __new_first - __first;
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__rotate_wrapper>(__exec),
        unseq_backend::__rotate_copy<typename ::std::iterator_traits<_Iterator>::difference_type>{__n, __shift}, __n,
        __buf.all_view(), __temp_rng);

    using _Function = __brick_move<_ExecutionPolicy>;
    auto __brick = unseq_backend::walk_n<_ExecutionPolicy, _Function>{_Function{}};

    oneapi::dpl::__par_backend_hetero::__parallel_for(::std::forward<_ExecutionPolicy>(__exec), __brick, __n,
                                                      __temp_rng, __buf.all_view())
        .wait();

    return __first + (__last - __new_first);
}

//------------------------------------------------------------------------
// rotate_copy
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _BidirectionalIterator, typename _ForwardIterator>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator>
__pattern_rotate_copy(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __new_first,
                      _BidirectionalIterator __last, _ForwardIterator __result, /*vector=*/::std::true_type,
                      /*parallel=*/::std::true_type)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return __result;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _BidirectionalIterator>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator>();
    auto __buf2 = __keep2(__result, __result + __n);

    const auto __shift = __new_first - __first;

    oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::__rotate_copy<typename ::std::iterator_traits<_BidirectionalIterator>::difference_type>{__n,
                                                                                                               __shift},
        __n, __buf1.all_view(), __buf2.all_view())
        .wait();

    return __result + __n;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator,
          typename _Compare, typename _IsOpDifference>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_hetero_set_op(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                        _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                        _Compare __comp, _IsOpDifference)
{
    typedef typename ::std::iterator_traits<_ForwardIterator1>::difference_type _Size1;
    typedef typename ::std::iterator_traits<_ForwardIterator2>::difference_type _Size2;

    const _Size1 __n1 = __last1 - __first1;
    const _Size2 __n2 = __last2 - __first2;

    //Algo is based on the recommended approach of set_intersection algo for GPU: binary search + scan (copying by mask).
    using _ReduceOp = ::std::plus<_Size1>;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _MaskAssigner = unseq_backend::__mask_assigner<2>;
    using _InitType = unseq_backend::__scan_no_init<_Size1>;
    using _DataAcc = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _ReduceOp __reduce_op;
    _Assigner __assign_op;
    _DataAcc __get_data_op;
    unseq_backend::__copy_by_mask<_ReduceOp, /*inclusive*/ ::std::true_type, 2> __copy_by_mask_op;
    unseq_backend::__brick_set_op<_ExecutionPolicy, _Compare, _Size1, _Size2, _IsOpDifference> __create_mask_op{
        __comp, __n1, __n2};

    // temporary buffer to store boolean mask
    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, int32_t> __mask_buf(__exec, __n1);

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __last2);

    auto __keep3 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutputIterator>();
    auto __buf3 = __keep3(__result, __result + __n1 + __n2);

    auto __result_size =
        __par_backend_hetero::__parallel_transform_scan(
            ::std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__ranges::make_zip_view(
                __buf1.all_view(), __buf2.all_view(),
                oneapi::dpl::__ranges::all_view<int32_t, __par_backend_hetero::access_mode::read_write>(
                    __mask_buf.get_buffer())),
            __buf3.all_view(), __reduce_op, _InitType{},
            // local scan
            unseq_backend::__scan</*inclusive*/ ::std::true_type, _ExecutionPolicy, _ReduceOp, _DataAcc, _Assigner,
                                  _MaskAssigner, decltype(__create_mask_op), _InitType>{
                __reduce_op, __get_data_op, __assign_op, _MaskAssigner{}, __create_mask_op},
            // scan between groups
            unseq_backend::__scan</*inclusive=*/::std::true_type, _ExecutionPolicy, _ReduceOp, _DataAcc, _NoAssign,
                                  _Assigner, _DataAcc, _InitType>{__reduce_op, __get_data_op, _NoAssign{}, __assign_op,
                                                                  __get_data_op},
            // global scan
            __copy_by_mask_op)
            .second;

    return __result + __result_size;
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator,
          typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_set_intersection(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                           _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                           _Compare __comp, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    // intersection is empty
    if (__first1 == __last1 || __first2 == __last2)
        return __result;

    return __pattern_hetero_set_op(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2,
                                   __result, __comp, unseq_backend::_IntersectionTag());
}

//Dummy names to avoid kernel problems
template <typename Name>
class __set_difference_copy_case_1
{
};

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator,
          typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_set_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                         _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                         _Compare __comp, /*vector=*/::std::true_type,
                         /*parallel=*/::std::true_type)
{
    // {} \ {2}: the difference is empty
    if (__first1 == __last1)
        return __result;

    // {1} \ {}: the difference is {1}
    if (__first2 == __last2)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_difference_copy_case_1>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __result, oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::true_type());
    }

    return __pattern_hetero_set_op(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2,
                                   __result, __comp, unseq_backend::_DifferenceTag());
}

//Dummy names to avoid kernel problems
template <typename Name>
class __set_union_copy_case_1
{
};

template <typename Name>
class __set_union_copy_case_2
{
};

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator,
          typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_set_union(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                    _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                    /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__first1 == __last1 && __first2 == __last2)
        return __result;

    //{1} is empty
    if (__first1 == __last1)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_union_copy_case_1>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first2, __last2, __result, oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::true_type());
    }

    //{2} is empty
    if (__first2 == __last2)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_union_copy_case_2>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __result, oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::true_type());
    }

    typedef typename ::std::iterator_traits<_OutputIterator>::value_type _ValueType;

    // temporary buffer to store intermediate result
    const auto __n2 = __last2 - __first2;
    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __diff(__exec, __n2);
    auto __buf = __diff.get();

    //1. Calc difference {2} \ {1}
    const auto __n_diff = oneapi::dpl::__internal::__pattern_hetero_set_op(::std::forward<_ExecutionPolicy>(__exec),
                                                                           __first2, __last2, __first1, __last1, __buf,
                                                                           __comp, unseq_backend::_DifferenceTag()) -
                          __buf;
    //2. Merge {1} and the difference
    return oneapi::dpl::__internal::__pattern_merge(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __buf,
                                                    __buf + __n_diff, __result, __comp,
                                                    /*vector=*/::std::true_type(), /*parallel=*/::std::true_type());
}

//Dummy names to avoid kernel problems
template <typename Name>
class __set_symmetric_difference_copy_case_1
{
};

template <typename Name>
class __set_symmetric_difference_copy_case_2
{
};

template <typename Name>
class __set_symmetric_difference_phase_1
{
};

template <typename Name>
class __set_symmetric_difference_phase_2
{
};

//------------------------------------------------------------------------
// set_symmetric_difference
//------------------------------------------------------------------------
// At the moment the algo imlementation based on 3 phases:
// 1. Calc difference {1} \ {2}
// 2. Calc difference {2} \ {1}
// 3. Merge the differences
template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator,
          typename _Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_set_symmetric_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                   _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                                   _Compare __comp, /*vector=*/::std::true_type,
                                   /*parallel=*/::std::true_type)
{
    if (__first1 == __last1 && __first2 == __last2)
        return __result;

    //{1} is empty
    if (__first1 == __last1)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_copy_case_1>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first2, __last2, __result, oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::true_type());
    }

    //{2} is empty
    if (__first2 == __last2)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_copy_case_2>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __result, oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{}, ::std::true_type());
    }

    typedef typename ::std::iterator_traits<_OutputIterator>::value_type _ValueType;

    // temporary buffers to store intermediate result
    const auto __n1 = __last1 - __first1;
    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __diff_1(__exec, __n1);
    auto __buf_1 = __diff_1.get();
    const auto __n2 = __last2 - __first2;
    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _ValueType> __diff_2(__exec, __n2);
    auto __buf_2 = __diff_2.get();

    //1. Calc difference {1} \ {2}
    const auto __n_diff_1 =
        oneapi::dpl::__internal::__pattern_hetero_set_op(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_phase_1>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __first2, __last2, __buf_1, __comp, unseq_backend::_DifferenceTag()) -
        __buf_1;

    //2. Calc difference {2} \ {1}
    const auto __n_diff_2 =
        oneapi::dpl::__internal::__pattern_hetero_set_op(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_phase_2>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first2, __last2, __first1, __last1, __buf_2, __comp, unseq_backend::_DifferenceTag()) -
        __buf_2;

    //3. Merge the differences
    return oneapi::dpl::__internal::__pattern_merge(::std::forward<_ExecutionPolicy>(__exec), __buf_1,
                                                    __buf_1 + __n_diff_1, __buf_2, __buf_2 + __n_diff_2, __result,
                                                    __comp, ::std::true_type(), ::std::true_type());
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_algorithm_impl_hetero_H */
