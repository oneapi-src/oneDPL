// -*- C++ -*-
//===-- induction.h -------------------------------------------------------===//
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

#ifndef _ONEDPL_EXPERIMENTAL_INDUCTION_H
#define _ONEDPL_EXPERIMENTAL_INDUCTION_H

#include <type_traits>

#include "induction_impl.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{
inline namespace parallelism_v2
{

template <typename _Tp>
oneapi::dpl::__internal::__induction_object<_Tp, void>
induction(_Tp&& __var)
{
    return {::std::forward<_Tp>(__var)};
}

template <typename _Tp, typename _Sp>
oneapi::dpl::__internal::__induction_object<_Tp, _Sp>
induction(_Tp&& __var, _Sp __stride)
{
    return {::std::forward<_Tp>(__var), __stride};
}

} // namespace parallelism_v2
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXPERIMENTAL_INDUCTION_H
