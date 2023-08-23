// -*- C++ -*-
//===-- oneapi_tbb_parallel_sort_h.pass.cpp ----------------------------------===//
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

#include <oneapi/tbb/parallel_sort.h>
#include "support/utils.h"

int main() {
    int array[3];
    oneapi::tbb::parallel_sort(array, array + 3);
    return TestUtils::done();
}
