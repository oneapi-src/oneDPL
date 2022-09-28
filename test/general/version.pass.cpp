// -*- C++ -*-
//===-- version.pass.cpp --------------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)

#include <oneapi/dpl/pstl/onedpl_config.h>

#include "support/utils.h"

#define HAS_EXECUTION_HEADER __has_include(<execution>)

#if !HAS_EXECUTION_HEADER
static_assert(_PSTL_VERSION == 11000, "");
static_assert(_PSTL_VERSION_MAJOR == 11, "");
static_assert(_PSTL_VERSION_MINOR == 00, "");
static_assert(_PSTL_VERSION_PATCH == 0, "");
#endif

static_assert(ONEDPL_VERSION_MAJOR == 2021, "");
static_assert(ONEDPL_VERSION_MINOR == 7, "");
static_assert(ONEDPL_VERSION_PATCH == 1, "");

int main() {

    return TestUtils::done();
}
