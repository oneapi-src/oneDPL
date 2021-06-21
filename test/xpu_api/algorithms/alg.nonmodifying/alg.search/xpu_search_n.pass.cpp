//===-- xpu_search_n.pass.cpp ---------------------------------------------===//
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

#include <cassert>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using oneapi::dpl::search_n;

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<Iter>([=]() {
                int ia[] = {0, 1, 2, 3, 4, 5};
                const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 0, 0) == Iter(ia));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 1, 0) == Iter(ia + 0));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 2, 0) == Iter(ia + sa));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), sa, 0) == Iter(ia + sa));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 0, 3) == Iter(ia));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 1, 3) == Iter(ia + 3));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 2, 3) == Iter(ia + sa));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), sa, 3) == Iter(ia + sa));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 0, 5) == Iter(ia));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 1, 5) == Iter(ia + 5));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), 2, 5) == Iter(ia + sa));
                ret_acc[0] &= (search_n(Iter(ia), Iter(ia + sa), sa, 5) == Iter(ia + sa));

                int ib[] = {0, 0, 1, 1, 2, 2};
                const unsigned sb = sizeof(ib) / sizeof(ib[0]);
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 0, 0) == Iter(ib));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 1, 0) == Iter(ib + 0));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 2, 0) == Iter(ib + 0));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 3, 0) == Iter(ib + sb));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), sb, 0) == Iter(ib + sb));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 0, 1) == Iter(ib));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 1, 1) == Iter(ib + 2));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 2, 1) == Iter(ib + 2));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 3, 1) == Iter(ib + sb));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), sb, 1) == Iter(ib + sb));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 0, 2) == Iter(ib));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 1, 2) == Iter(ib + 4));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 2, 2) == Iter(ib + 4));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), 3, 2) == Iter(ib + sb));
                ret_acc[0] &= (search_n(Iter(ib), Iter(ib + sb), sb, 2) == Iter(ib + sb));

                int ic[] = {0, 0, 0};
                const unsigned sc = sizeof(ic) / sizeof(ic[0]);
                ret_acc[0] &= (search_n(Iter(ic), Iter(ic + sc), 0, 0) == Iter(ic));
                ret_acc[0] &= (search_n(Iter(ic), Iter(ic + sc), 1, 0) == Iter(ic));
                ret_acc[0] &= (search_n(Iter(ic), Iter(ic + sc), 2, 0) == Iter(ic));
                ret_acc[0] &= (search_n(Iter(ic), Iter(ic + sc), 3, 0) == Iter(ic));
                ret_acc[0] &= (search_n(Iter(ic), Iter(ic + sc), 4, 0) == Iter(ic + sc));
            });
        });
    }
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    test<forward_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>>(deviceQueue);
    return 0;
}
