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

#ifndef _REDUCE_SERIAL_IMPL_H
#define _REDUCE_SERIAL_IMPL_H

#include <iterator>

template <typename ViewKeys, typename ViewVals, typename ResKeys, typename ResVals, typename Size, typename T,
          typename BinaryPred, typename BinaryOperation>
::std::size_t
reduce_by_segment_serial(ViewKeys keys, ViewVals vals, ResKeys& res_keys, ResVals& res_vals, Size n, T init,
                         BinaryPred binary_pred, BinaryOperation binary_op)
{
    if (n < 1)
        return 0;

    size_t segment_count = 0;
    res_vals[segment_count] = init;
    res_keys[segment_count] = keys[0];
    for (Size i = 0; i < n; ++i)
    {
        res_vals[segment_count] = binary_op(res_vals[segment_count], vals[i]);
        if (i < n - 1 && !binary_pred(keys[i], keys[i + 1]))
        {
            ++segment_count;
            res_keys[segment_count] = keys[i + 1];
            res_vals[segment_count] = init;
        }
    }
    return segment_count + 1;
}

#endif //_REDUCE_SERIAL_IMPL_H