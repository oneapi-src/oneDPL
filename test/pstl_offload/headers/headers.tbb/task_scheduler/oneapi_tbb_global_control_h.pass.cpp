// -*- C++ -*-
//===-- oneapi_tbb_global_control_h.pass.cpp ----------------------------------===//
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

#include <oneapi/tbb/global_control.h>
#include "support/utils.h"

int main() {
    oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, 1);
    return TestUtils::done();
}
