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

#include "reduce_by_segment.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    // test with flag pred
    test_flag_pred<sycl::usm::alloc::device, class KernelName1, std::uint64_t>();
    test_flag_pred<sycl::usm::alloc::device, class KernelName2, dpl::complex<float>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
