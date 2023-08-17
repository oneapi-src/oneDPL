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

#ifndef _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H
#define _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H

#include "../pstl/onedpl_config.h"

namespace oneapi
{
namespace dpl
{

template <typename _ExecutionPolicy, typename _InputIterator, typename _Size, typename _T, typename _OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _OutputIterator>
histogram(_ExecutionPolicy&& policy, _InputIterator __first, _InputIterator __last, const _Size& num_bins,
          const _T& __first_bin_min_val, const _T& __last_bin_max_val, _OutputIterator __histogram_first);

template <typename _ExecutionPolicy, typename _InputIterator1, typename _InputIterator2, typename _OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _OutputIterator>
histogram(_ExecutionPolicy&& policy, _InputIterator1 __first, _InputIterator1 __last, _InputIterator2 __boundary_first,
          _InputIterator2 __boundary_last, _OutputIterator __histogram_first);

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H
