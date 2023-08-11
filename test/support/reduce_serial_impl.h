// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _REDUCE_SERIAL_IMPL_H
#define _REDUCE_SERIAL_IMPL_H

#include <iterator>

template <typename RandAccessItKeysIn, typename RandAccessItValsIn, typename RandAccessItKeysOut,
          typename RandAccessItValsOut, typename Size, typename BinaryPredicate, typename BinaryOperation>
Size
reduce_by_segment_serial(RandAccessItKeysIn keys, RandAccessItValsIn vals,
                         RandAccessItKeysOut res_keys, RandAccessItValsOut res_vals, Size n,
                         BinaryPredicate binary_pred, BinaryOperation binary_op)
{
    if (n < 1)
        return 0;
    
    using ValT = typename ::std::decay<decltype(vals[0])>::type;
    using KeyT = typename ::std::decay<decltype(keys[0])>::type;
    KeyT tmp_key = keys[0];
    ValT tmp_val = vals[0];
    Size segment_count = 0;
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
