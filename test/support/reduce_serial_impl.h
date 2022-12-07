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

template <typename ViewKeys, typename ViewVals, typename ResKeys, typename ResVals, typename Size,
          typename BinaryPredicate, typename BinaryOperation>
::std::size_t
reduce_by_segment_serial(ViewKeys keys, ViewVals vals, ResKeys& res_keys, ResVals& res_vals, Size n,
                         BinaryPredicate binary_pred, BinaryOperation binary_op)
{
    if (n < 1)
        return 0;
    
    using ValT = typename ::std::decay<decltype(vals[0])>::type;
    using KeyT = typename ::std::decay<decltype(keys[0])>::type;
    KeyT tmp_key = keys[0];
    ValT tmp_val = vals[0];
    std::size_t segment_count = 0;
    for (Size i = 1; i < n; ++i)
    {
        if (binary_pred(keys[i - 1], keys[i]))
        {
            tmp_val = binary_op(tmp_val, vals[i]);
        }
        else
        {
            res_keys[segment_count] = tmp_key;
            res_vals[segment_count] = tmp_val;
            segment_count++;
            tmp_key = keys[i];
            tmp_val = vals[i];
        }
    }
    res_keys[segment_count] = tmp_key;
    res_vals[segment_count] = tmp_val;

    return segment_count + 1;
}

#endif //_REDUCE_SERIAL_IMPL_H
