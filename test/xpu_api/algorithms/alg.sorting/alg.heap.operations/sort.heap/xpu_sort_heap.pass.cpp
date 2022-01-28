//===-- xpu_sort_heap.pass.cpp --------------------------------------------===//
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

#include <CL/sycl.hpp>
#include <cassert>

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    using T = typename std::iterator_traits<Iter>::value_type;
    bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter>([=]() {
                T orig[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
                T work[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
                for (int n = 0; n < 15; ++n)
                {
                    dpl::make_heap(work, work + n);
                    dpl::sort_heap(Iter(work), Iter(work + n));
                    ret_acc[0] &= dpl::is_sorted(work, work + n);
                    ret_acc[0] &= dpl::is_permutation(work, work + n, orig);
                    dpl::copy(orig, orig + n, work);
                }
            });
        });
    }
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);
    test<random_access_iterator<float*>>(deviceQueue);
    test<float*>(deviceQueue);
    if (deviceQueue.get_device().has(sycl::aspect::fp64))
    {
        test<random_access_iterator<double*>>(deviceQueue);
        test<double*>(deviceQueue);
    }
    return 0;
}
