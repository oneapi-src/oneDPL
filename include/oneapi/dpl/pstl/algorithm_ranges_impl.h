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

//------------------------------------------------------------------------
// walk_n
//------------------------------------------------------------------------

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
    return __return_t{__internal::__get_result(__r), std::move(__f)};
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Fun>
decltype(auto)
__pattern_for_each(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    return std::ranges::for_each(std::forward<_R>(__r), __f, __proj);
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL___cplusplus >= 202002L

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_H
