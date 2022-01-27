//===-- xpu_rotate.pass.cpp -----------------------------------------------===//
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

#include "support/math_utils.h"
#include "support/test_iterators.h"

#include <cassert>
#include <CL/sycl.hpp>

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    bool ret = true;
    using T = typename std::iterator_traits<Iter>::value_type;
    T ia[] = {0, 1, 2, 3, 4};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    sycl::range<1> item1{1};
    sycl::range<1> itemN{sa};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<T, 1> buffer2(ia, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.template get_access<sycl::access::mode::write>(cgh);
            auto acc_arr1 = buffer2.template get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter>([=]() {
                auto r = dpl::rotate(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + 1), Iter(&acc_arr1[0] + sa));
                ret_acc[0] &= (base(r) == &acc_arr1[0] + 4);
            });
        });
    }
    assert(ret);
    if (std::is_floating_point<T>::value)
    {
        assert(approx_equal_fp(ia[0], T(1)));
        assert(approx_equal_fp(ia[1], T(2)));
        assert(approx_equal_fp(ia[2], T(3)));
        assert(approx_equal_fp(ia[3], T(4)));
        assert(approx_equal_fp(ia[4], T(0)));
    }
    else
    {
        assert(ia[0] == 1);
        assert(ia[1] == 2);
        assert(ia[2] == 3);
        assert(ia[3] == 4);
        assert(ia[4] == 0);
    }
}

int
main()
{
    sycl::queue deviceQueue;
    test<forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);
    test<forward_iterator<float*>>(deviceQueue);
    test<bidirectional_iterator<float*>>(deviceQueue);
    test<random_access_iterator<float*>>(deviceQueue);
    test<float*>(deviceQueue);
    if (deviceQueue.get_device().has(sycl::aspect::fp64))
    {
        test<forward_iterator<double*>>(deviceQueue);
        test<bidirectional_iterator<double*>>(deviceQueue);
        test<random_access_iterator<double*>>(deviceQueue);
        test<double*>(deviceQueue);
    }
    return 0;
}
