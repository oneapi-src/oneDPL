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
