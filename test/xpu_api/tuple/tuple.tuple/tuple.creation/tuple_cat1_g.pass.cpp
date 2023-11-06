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

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/utility>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/array>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
void
kernel_test1(sycl::queue& deviceQueue)
{
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest1>([=]() {
                static_assert(dpl::is_same<decltype(dpl::tuple_cat()), dpl::tuple<>>::value);
                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::tuple<>>())), dpl::tuple<>>::value);
                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::tuple<>&>())), dpl::tuple<>>::value);
                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<const dpl::tuple<>>())), dpl::tuple<>>::value);
                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<const dpl::tuple<>&>())), dpl::tuple<>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::pair<int, bool>>())),
                                           dpl::tuple<int, bool>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::pair<int, bool>&>())),
                                           dpl::tuple<int, bool>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<const dpl::pair<int, bool>>())),
                                           dpl::tuple<int, bool>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<const dpl::pair<int, bool>&>())),
                                           dpl::tuple<int, bool>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::array<int, 3>>())),
                                           dpl::tuple<int, int, int>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::array<int, 3>&>())),
                                           dpl::tuple<int, int, int>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<const dpl::array<int, 3>>())),
                                           dpl::tuple<int, int, int>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<const dpl::array<int, 3>&>())),
                                           dpl::tuple<int, int, int>>::value);
                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::tuple<>>(), dpl::declval<dpl::tuple<>>())),
                                 dpl::tuple<>>::value);
                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::tuple<>>(), dpl::declval<dpl::tuple<>>(),
                                                         dpl::declval<dpl::tuple<>>())),
                                 dpl::tuple<>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(
                                               dpl::declval<dpl::tuple<>>(), dpl::declval<dpl::array<char, 0>>(),
                                               dpl::declval<dpl::array<int, 0>>(), dpl::declval<dpl::tuple<>>())),
                                           dpl::tuple<>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::tuple<int>>(),
                                                                   dpl::declval<dpl::tuple<float>>())),
                                           dpl::tuple<int, float>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::tuple<int>>(),
                                                                   dpl::declval<dpl::tuple<float>>(),
                                                                   dpl::declval<dpl::tuple<const long&>>())),
                                           dpl::tuple<int, float, const long&>>::value);
                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(
                                     dpl::declval<dpl::array<wchar_t, 3>&>(), dpl::declval<dpl::tuple<float>>(),
                                     dpl::declval<dpl::tuple<>>(), dpl::declval<dpl::tuple<unsigned&>>(),
                                     dpl::declval<dpl::pair<bool, std::nullptr_t>>())),
                                 dpl::tuple<wchar_t, wchar_t, wchar_t, float, unsigned&, bool, std::nullptr_t>>::value);

                dpl::array<int, 3> a3;
                dpl::pair<float, bool> pdb;
                dpl::tuple<unsigned, float, std::nullptr_t, void*> t;
                int i{};
                float d{};
                int* pi{};
                dpl::tuple<int&, float&, int*&> to{i, d, pi};

                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(a3, pdb, t, a3, pdb, t)),
                                 dpl::tuple<int, int, int, float, bool, unsigned, float, std::nullptr_t, void*, int,
                                            int, int, float, bool, unsigned, float, std::nullptr_t, void*>>::value);

                dpl::tuple_cat(dpl::tuple<int, char, void*>{}, to, a3, dpl::tuple<>{},
                               dpl::pair<float, std::nullptr_t>{}, pdb, to);
            });
        });
    }
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest2>([=]() {
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::tuple<int>>(),
                                                                   dpl::declval<dpl::tuple<double>>())),
                                           dpl::tuple<int, double>>::value);
                static_assert(dpl::is_same<decltype(dpl::tuple_cat(dpl::declval<dpl::tuple<int>>(),
                                                                   dpl::declval<dpl::tuple<double>>(),
                                                                   dpl::declval<dpl::tuple<const long&>>())),
                                           dpl::tuple<int, double, const long&>>::value);
                static_assert(dpl::is_same<
                              decltype(dpl::tuple_cat(dpl::declval<dpl::array<wchar_t, 3>&>(),
                                                      dpl::declval<dpl::tuple<double>>(), dpl::declval<dpl::tuple<>>(),
                                                      dpl::declval<dpl::tuple<unsigned&>>(),
                                                      dpl::declval<dpl::pair<bool, std::nullptr_t>>())),
                              dpl::tuple<wchar_t, wchar_t, wchar_t, double, unsigned&, bool, std::nullptr_t>>::value);

                dpl::array<int, 3> a3;
                dpl::pair<double, bool> pdb;
                dpl::tuple<unsigned, float, std::nullptr_t, void*> t;
                static_assert(
                    dpl::is_same<decltype(dpl::tuple_cat(a3, pdb, t, a3, pdb, t)),
                                 dpl::tuple<int, int, int, double, bool, unsigned, float, std::nullptr_t, void*, int,
                                            int, int, double, bool, unsigned, float, std::nullptr_t, void*>>::value);
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test2(deviceQueue);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
