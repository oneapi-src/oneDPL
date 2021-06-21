//===-- xpu_for_each_n.pass.cpp -------------------------------------------===//
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

#include <oneapi/dpl/algorithm>

#include "support/test_iterators.h"
#include "support/utils.h"

#include <cassert>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

struct for_each_test
{
    for_each_test(int c) : count(c) {}
    int count;
    void
    operator()(int& i)
    {
        ++i;
        ++count;
    }
};

#if _ONEDPL_CPP17_FUNCTIONALITY_PRESENT
using oneapi::dpl::for_each_n;

void
kernel_test(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                typedef input_iterator<int*> Iter;
                int ia[] = {0, 1, 2, 3, 4, 5};
                const unsigned s = sizeof(ia) / sizeof(ia[0]);
                {
                    {
                        auto f = for_each_test(0);
                        Iter it = for_each_n(Iter(ia), 0, std::ref(f));
                        ret_acc[0] &= (it == Iter(ia));
                        ret_acc[0] &= (f.count == 0);
                    }

                    {
                        auto f = for_each_test(0);
                        Iter it = for_each_n(Iter(ia), s, std::ref(f));

                        ret_acc[0] &= (it == Iter(ia + s));
                        ret_acc[0] &= (f.count == s);
                        for (unsigned i = 0; i < s; ++i)
                            ret_acc[0] &= (ia[i] == static_cast<int>(i + 1));
                    }

                    {
                        auto f = for_each_test(0);
                        Iter it = for_each_n(Iter(ia), 1, std::ref(f));

                        ret_acc[0] &= (it == Iter(ia + 1));
                        ret_acc[0] &= (f.count == 1);
                        for (unsigned i = 0; i < 1; ++i)
                            ret_acc[0] &= (ia[i] == static_cast<int>(i + 2));
                    }
                }
            });
        });
    }
    assert(ret);
}
#endif

int
main()
{
    sycl::queue deviceQueue;
#if _ONEDPL_CPP17_FUNCTIONALITY_PRESENT
    kernel_test(deviceQueue);
#else
    std::cout << TestUtils::done(0) << ::std::endl;
#endif
    return 0;
}
