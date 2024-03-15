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

#ifndef _ONEDPL_ALGORITHM_RANGES_IMPL_H
#define _ONEDPL_ALGORITHM_RANGES_IMPL_H

#if _ONEDPL___cplusplus >= 202002L

#include <ranges>
#include "algorithm_fwd.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//---------------------------------------------------------------------------------------------------------------------
// pattern_for_each
//---------------------------------------------------------------------------------------------------------------------

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Fun>
decltype(auto)
__pattern_for_each(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    using _It = std::ranges::iterator_t<_R>;
    auto __view = std::views::common(::std::forward<_R>(__r));
    
    auto __f_1 = [__f, __proj](auto&& __val) { __f(__proj(__val));};
    
    oneapi::dpl::__internal::__pattern_walk1(__tag, ::std::forward<_ExecutionPolicy>(__exec), __view.begin(),
        __view.end(), __f_1);

    using __return_t = std::ranges::for_each_result<std::ranges::borrowed_iterator_t<_R>, _Fun>;
    return __return_t{__r.begin() + __r.size(), std::move(__f)};
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Fun>
decltype(auto)
__pattern_for_each(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    return std::ranges::for_each(std::forward<_R>(__r), __f, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_transform
//---------------------------------------------------------------------------------------------------------------------

template<typename _IsVector, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _F, typename _Proj>
decltype(auto)
__pattern_transform(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r,
                    _F __op, _Proj __proj)
{
    assert(__in_r.size() == __out_r.size());

    auto __unary_op = [=](auto&& __val) -> decltype(auto) { return __op(__proj(__val));};
    
    using _ItIn = std::ranges::iterator_t<_InRange>;
    using _ItOut = std::ranges::iterator_t<_OutRange>;

    auto __view_in = std::views::common(::std::forward<_InRange>(__in_r));
    auto __view_out = std::views::common(::std::forward<_OutRange>(__out_r));

    oneapi::dpl::__internal::__pattern_walk2(__tag, ::std::forward<_ExecutionPolicy>(__exec), __view_in.begin(),
        __view_in.end(), __view_out.begin(),
        oneapi::dpl::__internal::__transform_functor<decltype(__unary_op)>{::std::move(__unary_op)});

    using __return_t = std::ranges::unary_transform_result<std::ranges::borrowed_iterator_t<_InRange>,
        std::ranges::borrowed_iterator_t<_OutRange>>;

    return __return_t{__in_r.begin() + __in_r.size(), __out_r.begin() + __out_r.size()};
}

template<typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _F, typename _Proj>
decltype(auto)
__pattern_transform(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r,
                    _F __op, _Proj __proj)
{
    return std::ranges::transform(std::forward<_InRange>(__in_r), __out_r.begin(), __op, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_find_if
//---------------------------------------------------------------------------------------------------------------------

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
decltype(auto)
__pattern_find_if(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    auto __pred_1 = [__pred, __proj](auto&& __val) { return __pred(__proj(__val));};

    using _It = std::ranges::iterator_t<_R>;
    auto __view = std::views::common(::std::forward<_R>(__r));

    return std::ranges::borrowed_iterator_t<_R>(oneapi::dpl::__internal::__pattern_find_if(__tag,
        ::std::forward<_ExecutionPolicy>(__exec), __view.begin(), __view.end(), __pred_1));
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
decltype(auto)
__pattern_find_if(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::find_if(std::forward<_R>(__r), __pred, __proj);
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
bool
__pattern_any_of(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    auto __pred_1 = [__pred, __proj](auto&& __val) { return __pred(__proj(__val));};

    using _It = std::ranges::iterator_t<_R>;
    auto __view = std::views::common(::std::forward<_R>(__r));

    return oneapi::dpl::__internal::__pattern_any_of(__tag, ::std::forward<_ExecutionPolicy>(__exec), __view.begin(),
        __view.end(), __pred_1);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
bool
__pattern_any_of(_Tag __tag, _ExecutionPolicy&&, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::any_of(std::forward<_R>(__r), __pred, __proj);
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL___cplusplus >= 202002L

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_H
