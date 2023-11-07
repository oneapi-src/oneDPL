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

#include "support/test_config.h"

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using dpl::optional;

template <class KernelTest, class Opt, class T>
void
test()
{
    sycl::queue q;
    {

        q.submit([&](sycl::handler& cgh) {
            cgh.single_task<KernelTest>([=]() { static_assert(dpl::is_same<typename Opt::value_type, T>::value); });
        });
    }
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test<KernelTest1, optional<int>, int>();
    test<KernelTest2, optional<const int>, const int>();
    test<KernelTest3, optional<double>, double>();
    test<KernelTest4, optional<const double>, const double>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
