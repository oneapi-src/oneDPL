//===-- xpu_rotate_copy.pass.cpp ------------------------------------------===//
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

template <class InIter, class OutIter>
class KernelTest;

template <class InIter, class OutIter>
void
test(sycl::queue& deviceQueue)
{
    bool ret = true;
    using T = typename std::iterator_traits<InIter>::value_type;
    T ia[] = {0, 1, 2, 3, 4};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    T ib[sa] = {0};
    sycl::range<1> item1{1};
    sycl::range<1> itemN{sa};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<T, 1> buffer2(ia, itemN);
        sycl::buffer<T, 1> buffer3(ib, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.template get_access<sycl::access::mode::write>(cgh);
            auto acc_arr1 = buffer2.template get_access<sycl::access::mode::write>(cgh);
            auto acc_arr2 = buffer3.template get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest<InIter, OutIter>>([=]() {
                OutIter r = dpl::rotate_copy(InIter(&acc_arr1[0]), InIter(&acc_arr1[0] + 2), InIter(&acc_arr1[0] + sa),
                                             OutIter(&acc_arr2[0]));
                ret_acc[0] = (base(r) == &acc_arr2[0] + sa);
            });
        });
    }
    assert(ret);
    if (std::is_floating_point<T>::value)
    {
        assert(approx_equal_fp(ib[0], T(2)));
        assert(approx_equal_fp(ib[1], T(3)));
        assert(approx_equal_fp(ib[2], T(4)));
        assert(approx_equal_fp(ib[3], T(0)));
        assert(approx_equal_fp(ib[4], T(1)));
    }
    else
    {
        assert(ib[0] == 2);
        assert(ib[1] == 3);
        assert(ib[2] == 4);
        assert(ib[3] == 0);
        assert(ib[4] == 1);
    }
}

int
main()
{
    sycl::queue deviceQueue;
    test<bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, int*>(deviceQueue);

    test<const int*, output_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<int*>>(deviceQueue);
    test<const int*, int*>(deviceQueue);

    test<bidirectional_iterator<const float*>, output_iterator<float*>>(deviceQueue);
    test<bidirectional_iterator<const float*>, forward_iterator<float*>>(deviceQueue);
    test<bidirectional_iterator<const float*>, bidirectional_iterator<float*>>(deviceQueue);
    test<bidirectional_iterator<const float*>, random_access_iterator<float*>>(deviceQueue);
    test<bidirectional_iterator<const float*>, float*>(deviceQueue);

    test<random_access_iterator<const float*>, output_iterator<float*>>(deviceQueue);
    test<random_access_iterator<const float*>, forward_iterator<float*>>(deviceQueue);
    test<random_access_iterator<const float*>, bidirectional_iterator<float*>>(deviceQueue);
    test<random_access_iterator<const float*>, random_access_iterator<float*>>(deviceQueue);
    test<random_access_iterator<const float*>, float*>(deviceQueue);

    test<const float*, output_iterator<float*>>(deviceQueue);
    test<const float*, forward_iterator<float*>>(deviceQueue);
    test<const float*, bidirectional_iterator<float*>>(deviceQueue);
    test<const float*, random_access_iterator<float*>>(deviceQueue);
    test<const float*, float*>(deviceQueue);

    test<bidirectional_iterator<const double*>, output_iterator<double*>>(deviceQueue);
    test<bidirectional_iterator<const double*>, forward_iterator<double*>>(deviceQueue);
    test<bidirectional_iterator<const double*>, bidirectional_iterator<double*>>(deviceQueue);
    test<bidirectional_iterator<const double*>, random_access_iterator<double*>>(deviceQueue);
    test<bidirectional_iterator<const double*>, double*>(deviceQueue);

    test<random_access_iterator<const double*>, output_iterator<double*>>(deviceQueue);
    test<random_access_iterator<const double*>, forward_iterator<double*>>(deviceQueue);
    test<random_access_iterator<const double*>, bidirectional_iterator<double*>>(deviceQueue);
    test<random_access_iterator<const double*>, random_access_iterator<double*>>(deviceQueue);
    test<random_access_iterator<const double*>, double*>(deviceQueue);

    test<const double*, output_iterator<double*>>(deviceQueue);
    test<const double*, forward_iterator<double*>>(deviceQueue);
    test<const double*, bidirectional_iterator<double*>>(deviceQueue);
    test<const double*, random_access_iterator<double*>>(deviceQueue);
    test<const double*, double*>(deviceQueue);

    return 0;
}
