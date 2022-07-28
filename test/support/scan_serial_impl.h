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

#ifndef _SCAN_SERIAL_IMPL_H
#define _SCAN_SERIAL_IMPL_H

#include <iterator>

// We provide the no execution policy versions of the exclusive_scan and inclusive_scan due checking correctness result of the versions with execution policies.
//TODO: to add a macro for availability of ver implementations
template <class InputIterator, class OutputIterator, class T>
OutputIterator
exclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result, T init)
{
    for (; first != last; ++first, ++result)
    {
        auto res = init;
        init = init + *first;
        *result = res;
    }
    return result;
}

template <class InputIterator, class OutputIterator, class T, class BinaryOperation>
OutputIterator
exclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result, T init, BinaryOperation binary_op)
{
    for (; first != last; ++first, ++result)
    {
	auto res = init;
        init = binary_op(init, *first);
        *result = res;
    }
    return result;
}

// Note: N4582 is missing the ", class T".  Issue was reported 2016-Apr-11 to cxxeditor@gmail.com
template <class InputIterator, class OutputIterator, class BinaryOperation, class T>
OutputIterator
inclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result, BinaryOperation binary_op, T init)
{
    for (; first != last; ++first, ++result)
    {
        init = binary_op(init, *first);
        *result = init;
    }
    return result;
}

template <class InputIterator, class OutputIterator, class BinaryOperation>
OutputIterator
inclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result, BinaryOperation binary_op)
{
    if (first != last)
    {
        auto tmp = *first;
        *result = tmp;
        return inclusive_scan_serial(++first, last, ++result, binary_op, tmp);
    }
    else
    {
        return result;
    }
}

template <class InputIterator, class OutputIterator>
OutputIterator
inclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result)
{
    typedef typename ::std::iterator_traits<InputIterator>::value_type input_type;
    return inclusive_scan_serial(first, last, result, ::std::plus<input_type>());
}

template<typename ViewKeys, typename ViewVals, typename Res, typename Size, typename BinaryOperation>
void inclusive_scan_by_segment_serial(ViewKeys keys, ViewVals vals, Res& res, Size n, BinaryOperation binary_op)
{
    for (Size i = 0; i < n; ++i)
        if (i == 0 || keys[i] != keys[i - 1])
            res[i] = vals[i];
        else
            res[i] = binary_op(res[i - 1], vals[i]);
}

#endif //  _SCAN_SERIAL_IMPL_H
