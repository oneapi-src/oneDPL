//===-- xpu_find_end_pred.pass.cpp ----------------------------------------===//
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

using oneapi::dpl::find_end;

struct eq_struct
{
    template <class T>
    bool
    operator()(const T& x, const T& y)
    {
        return x == y;
    }
};

template <typename T1, typename T2>
class KernelName;

template <class Iter1, class Iter2>
void
test(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName<Iter1, Iter2>>([=]() {
                int ia[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0};
                const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                int b[] = {0};
                auto eq = eq_struct();
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia + sa), Iter2(b), Iter2(b + 1), eq) == Iter1(ia + sa - 1));
                int c[] = {0, 1};
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia + sa), Iter2(c), Iter2(c + 2), eq) == Iter1(ia + 18));
                int d[] = {0, 1, 2};
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia + sa), Iter2(d), Iter2(d + 3), eq) == Iter1(ia + 15));
                int e[] = {0, 1, 2, 3};
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia + sa), Iter2(e), Iter2(e + 4), eq) == Iter1(ia + 11));
                int f[] = {0, 1, 2, 3, 4};
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia + sa), Iter2(f), Iter2(f + 5), eq) == Iter1(ia + 6));
                int g[] = {0, 1, 2, 3, 4, 5};
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia + sa), Iter2(g), Iter2(g + 6), eq) == Iter1(ia));
                int h[] = {0, 1, 2, 3, 4, 5, 6};
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia + sa), Iter2(h), Iter2(h + 7), eq) == Iter1(ia + sa));
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia + sa), Iter2(b), Iter2(b), eq) == Iter1(ia + sa));
                ret_acc[0] &= (find_end(Iter1(ia), Iter1(ia), Iter2(b), Iter2(b + 1), eq) == Iter1(ia));
            });
        });
    }
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    test<forward_iterator<const int*>, forward_iterator<const int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<const int*>>(deviceQueue);
    return 0;
}
