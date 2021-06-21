//===-- xpu_search_pred.pass.cpp ------------------------------------------===//
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

struct eq_struct
{
    template <class T>
    bool
    operator()(const T& x, const T& y)
    {
        return x == y;
    }
};

using oneapi::dpl::search;

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
                int ia[] = {0, 1, 2, 3, 4, 5};
                const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                auto eq = eq_struct();
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia), eq) == Iter1(ia));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + 1), eq) == Iter1(ia));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 1), Iter2(ia + 2), eq) == Iter1(ia + 1));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 2), eq) == Iter1(ia));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 3), eq) == Iter1(ia + 2));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 3), eq) == Iter1(ia + 2));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia), Iter2(ia + 2), Iter2(ia + 3), eq) == Iter1(ia));
                ret_acc[0] &=
                    (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + sa - 1), Iter2(ia + sa), eq) == Iter1(ia + sa - 1));
                ret_acc[0] &=
                    (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + sa - 3), Iter2(ia + sa), eq) == Iter1(ia + sa - 3));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + sa), eq) == Iter1(ia));
                ret_acc[0] &=
                    (search(Iter1(ia), Iter1(ia + sa - 1), Iter2(ia), Iter2(ia + sa), eq) == Iter1(ia + sa - 1));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + 1), Iter2(ia), Iter2(ia + sa), eq) == Iter1(ia + 1));
                int ib[] = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
                const unsigned sb = sizeof(ib) / sizeof(ib[0]);
                int ic[] = {1};
                ret_acc[0] &= (search(Iter1(ib), Iter1(ib + sb), Iter2(ic), Iter2(ic + 1), eq) == Iter1(ib + 1));
                int id[] = {1, 2};
                ret_acc[0] &= (search(Iter1(ib), Iter1(ib + sb), Iter2(id), Iter2(id + 2), eq) == Iter1(ib + 1));
                int ie[] = {1, 2, 3};
                ret_acc[0] &= (search(Iter1(ib), Iter1(ib + sb), Iter2(ie), Iter2(ie + 3), eq) == Iter1(ib + 4));
                int ig[] = {1, 2, 3, 4};
                ret_acc[0] &= (search(Iter1(ib), Iter1(ib + sb), Iter2(ig), Iter2(ig + 4), eq) == Iter1(ib + 8));
                int ih[] = {0, 1, 1, 1, 1, 2, 3, 0, 1, 2, 3, 4};
                const unsigned sh = sizeof(ih) / sizeof(ih[0]);
                int ii[] = {1, 1, 2};
                ret_acc[0] &= (search(Iter1(ih), Iter1(ih + sh), Iter2(ii), Iter2(ii + 3), eq) == Iter1(ih + 3));
            });
        });
    }
    assert(ret);
}

int
main(int, char**)
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
