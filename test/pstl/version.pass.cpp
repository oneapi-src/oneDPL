// -*- C++ -*-
//===-- version.pass.cpp --------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

#include <pstl/internal/pstl_config.h>
#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"


static_assert(_PSTL_VERSION == 10000);
static_assert(_PSTL_VERSION_MAJOR == 10);
static_assert(_PSTL_VERSION_MINOR == 00);
static_assert(_PSTL_VERSION_PATCH == 0);

static_assert(__INTEL_PSTL_VERSION == 211);
static_assert(__INTEL_PSTL_VERSION_MAJOR == 2);
static_assert(__INTEL_PSTL_VERSION_MINOR == 11);

int main() {
    std::cout << TestUtils::done() << std::endl;
    return 0;
}
