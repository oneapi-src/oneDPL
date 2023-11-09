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

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/utils.h"

// #include "testsuite_struct.h" KSATODO required to remove?
// Looks like this code exist at https://github.com/search?q=NoexceptMoveAssignClass&type=code
// For example: https://github.com/bfg-repo-cleaner-demos/gcc-original/blob/86eac679c8bd26de9c5a1d2d8c20adfe59b59924/libstdc%2B%2B-v3/testsuite/util/testsuite_tr1.h#L222
// Repository: https://github.com/bfg-repo-cleaner-demos/gcc-original/tree/86eac679c8bd26de9c5a1d2d8c20adfe59b59924

#if TEST_DPCPP_BACKEND_PRESENT
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef dpl::pair<int, int> tt1;
                typedef dpl::pair<int, float> tt2;
                typedef dpl::pair<NoexceptMoveAssignClass, NoexceptMoveAssignClass> tt3;

                static_assert(std::is_nothrow_move_assignable<tt1>::value);
                static_assert(std::is_nothrow_move_assignable<tt2>::value);
                static_assert(std::is_nothrow_move_assignable<tt3>::value);
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
