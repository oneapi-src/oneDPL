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

//This file checks data transfer between host and device

#ifndef CHECK_DATA_H
#define CHECK_DATA_H

template <typename T1, typename T2>
bool
check_data(const T1* device_iter, const T2* host_iter, int N)
{
    for (int i = 0; i < N; ++i)
    {
        if (*(host_iter + i) != *(device_iter + i))
            return false;
    }
    return true;
}

#endif
