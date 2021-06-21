//===-- xpu_search.pass.cpp -----------------------------------------------===//
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

using oneapi::dpl::search;

namespace User
{
struct S
{
    S(int x) : x_(x) {}
    int x_;
};
bool
operator==(S lhs, S rhs)
{
    return lhs.x_ == rhs.x_;
}
template <class T, class U>
void
make_pair(T&&, U&&) = delete;
} // namespace User

template <typename T1, typename T2>
class KernelName;

template <typename T1>
class KernelName1;

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
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia)) == Iter1(ia));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + 1)) == Iter1(ia));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 1), Iter2(ia + 2)) == Iter1(ia + 1));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 2)) == Iter1(ia));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 3)) == Iter1(ia + 2));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + 2), Iter2(ia + 3)) == Iter1(ia + 2));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia), Iter2(ia + 2), Iter2(ia + 3)) == Iter1(ia));
                ret_acc[0] &=
                    (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + sa - 1), Iter2(ia + sa)) == Iter1(ia + sa - 1));
                ret_acc[0] &=
                    (search(Iter1(ia), Iter1(ia + sa), Iter2(ia + sa - 3), Iter2(ia + sa)) == Iter1(ia + sa - 3));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + sa)) == Iter1(ia));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + sa - 1), Iter2(ia), Iter2(ia + sa)) == Iter1(ia + sa - 1));
                ret_acc[0] &= (search(Iter1(ia), Iter1(ia + 1), Iter2(ia), Iter2(ia + sa)) == Iter1(ia + 1));
                int ib[] = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
                const unsigned sb = sizeof(ib) / sizeof(ib[0]);
                int ic[] = {1};
                ret_acc[0] &= (search(Iter1(ib), Iter1(ib + sb), Iter2(ic), Iter2(ic + 1)) == Iter1(ib + 1));
                int id[] = {1, 2};
                ret_acc[0] &= (search(Iter1(ib), Iter1(ib + sb), Iter2(id), Iter2(id + 2)) == Iter1(ib + 1));
                int ie[] = {1, 2, 3};
                ret_acc[0] &= (search(Iter1(ib), Iter1(ib + sb), Iter2(ie), Iter2(ie + 3)) == Iter1(ib + 4));
                int ig[] = {1, 2, 3, 4};
                ret_acc[0] &= (search(Iter1(ib), Iter1(ib + sb), Iter2(ig), Iter2(ig + 4)) == Iter1(ib + 8));
                int ih[] = {0, 1, 1, 1, 1, 2, 3, 0, 1, 2, 3, 4};
                const unsigned sh = sizeof(ih) / sizeof(ih[0]);
                int ii[] = {1, 1, 2};
                ret_acc[0] &= (search(Iter1(ih), Iter1(ih + sh), Iter2(ii), Iter2(ii + 3)) == Iter1(ih + 3));
                int ij[] = {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
                const unsigned sj = sizeof(ij) / sizeof(ij[0]);
                int ik[] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
                const unsigned sk = sizeof(ik) / sizeof(ik[0]);
                ret_acc[0] &= (search(Iter1(ij), Iter1(ij + sj), Iter2(ik), Iter2(ik + sk)) == Iter1(ij + 6));
            });
        });
    }
    assert(ret);
}

template <class Iter>
void
adl_test(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelName1<Iter>>([=]() {
                User::S ua[] = {1};
                ret_acc[0] &= (search(Iter(ua), Iter(ua), Iter(ua), Iter(ua)) == Iter(ua));
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

    adl_test<forward_iterator<User::S*>>(deviceQueue);
    adl_test<random_access_iterator<User::S*>>(deviceQueue);
    return 0;
}
