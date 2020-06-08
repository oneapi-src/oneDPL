// -*- C++ -*-
//===-- type_requirements.pass.cpp ---------------------------------------===//
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

#include "support/pstl_test_config.h"
#include "support/utils.h"

#if _PSTL_BACKEND_SYCL

#include <dpstd/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>

template <typename... Args>
void CheckTuple() {
    static_assert(std::is_trivially_copyable<dpstd::__par_backend_hetero::__internal::tuple<Args...>>::value, "");
    static_assert(std::is_standard_layout<dpstd::__par_backend_hetero::__internal::tuple<Args...>>::value, "");
};

#endif


int main() {
#if _PSTL_BACKEND_SYCL
    CheckTuple<int>();
    CheckTuple<int, long, float>();
#endif
    std::cout << TestUtils::done() << std::endl;
    return 0;
}
