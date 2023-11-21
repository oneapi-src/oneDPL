// -*- C++ -*-
//===-- transform_reduce.pass.cpp -----------------------------------------===//
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

#define _ONEDPL_DETECT_SPIRV_COMPILATION 1
#include "transform_reduce.pass.h"

int
main()
{
    return run_tests();
}
