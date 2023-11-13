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
#include <oneapi/dpl/complex>

#include "support/utils.h"
#include "support/MoveOnly.h"

#if TEST_DPCPP_BACKEND_PRESENT
struct CopyOnly
{
    CopyOnly() = default;
    CopyOnly(CopyOnly const&) noexcept = default;
    CopyOnly&
    operator=(CopyOnly const&)
    {
        return *this;
    }
};

struct NoexceptMoveOnly
{
    NoexceptMoveOnly() = default;
    NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept = default;
    NoexceptMoveOnly&
    operator=(NoexceptMoveOnly&&) noexcept
    {
        return *this;
    }
};

struct NotMoveConstructible
{
    NotMoveConstructible&
    operator=(NotMoveConstructible&&)
    {
        return *this;
    }

    NotMoveConstructible(NotMoveConstructible&&) = delete;
};

struct NotMoveAssignable
{
    NotMoveAssignable(NotMoveAssignable&&) = delete;

    NotMoveAssignable&
    operator=(NotMoveAssignable&&) = delete;
};

template <class Tp>
auto
can_swap_test(int) -> decltype(dpl::swap(dpl::declval<Tp>(), dpl::declval<Tp>()));

template <class Tp>
auto
can_swap_test(...) -> dpl::false_type;

template <class Tp>
constexpr bool
can_swap()
{
    return dpl::is_same_v<decltype(can_swap_test<Tp>(0)), void>;
}

class KernelSwapTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::range<1> numOfItems_acc{2};
    int acc[2] = {1, 2};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        sycl::buffer<int, 1> acc_buffer(acc, numOfItems_acc);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto acc_dev = acc_buffer.get_access<sycl::access::mode::read_write>(cgh);
            cgh.single_task<class KernelSwapTest>([=]() {
                {
                    int i = 1;
                    int j = 2;
                    dpl::swap(i, j);
                    ret_access[0] = (i == 2);
                    ret_access[0] &= (j == 1);
                }

                {
                    int a = 1;
                    int b = 2;
                    int* i = &a;
                    int* j = &b;
                    dpl::swap(i, j);
                    ret_access[0] &= (*i == 2);
                    ret_access[0] &= (*j == 1);
                }

                {
                    std::complex<float> c1(1.5f, 2.5f);
                    std::complex<float> c2(1.f, 5.5f);
                    ret_access[0] &= (c1.real() == 1.5f && c1.imag() == 2.5f);
                    ret_access[0] &= (c2.real() == 1.f && c2.imag() == 5.5f);
                    dpl::swap(c1, c2);
                    ret_access[0] &= (c2.real() == 1.5 && c2.imag() == 2.5);
                    ret_access[0] &= (c1.real() == 1 && c1.imag() == 5.5);
                }

                {
                    // test that the swap
                    static_assert(can_swap<CopyOnly&>());
                    static_assert(can_swap<MoveOnly&>());
                    static_assert(can_swap<NoexceptMoveOnly&>());

                    static_assert(!can_swap<NotMoveConstructible&>());
                    static_assert(!can_swap<NotMoveAssignable&>());

                    CopyOnly c;
                    MoveOnly m;
                    NoexceptMoveOnly nm;
                    static_assert(!noexcept(dpl::swap(c, c)));
                    static_assert(!noexcept(dpl::swap(m, m)));
                    static_assert(noexcept(dpl::swap(nm, nm)));
                }

                {
                    ret_access[0] &= (acc_dev[0] == 1);
                    ret_access[0] &= (acc_dev[1] == 2);
                    dpl::swap(acc_dev[0], acc_dev[1]);
                }
            });
        });
    }

    EXPECT_TRUE(ret && acc[0] == 2 && acc[1] == 1, "Wrong result of dpl::swap check");
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
