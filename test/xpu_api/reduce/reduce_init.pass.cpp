// -*- C++ -*-
//===-- reduce_init.pass.cpp
//--------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#include <CL/sycl.hpp>
#include <iostream>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/type_traits>

#include "test_iterators.h"

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T> class KernelTest;

template <typename _T1, typename _T2> void ASSERT_EQUAL(_T1 &&X, _T2 &&Y) {
  if (X != Y)
    std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y
              << ")" << std::endl;
}

template <class Iter> void test() {
  cl::sycl::queue deviceQueue;
  int input[6] = {1, 2, 3, 4, 5, 6};
  int output[8] = {};
  cl::sycl::range<1> numOfItems1{6};
  cl::sycl::range<1> numOfItems2{8};

  {
    cl::sycl::buffer<int, 1> buffer1(input, numOfItems1);
    cl::sycl::buffer<int, 1> buffer2(output, numOfItems2);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto in = buffer1.get_access<sycl_read>(cgh);
      auto out = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<class KernelTest<Iter>>([=]() {
        out[0] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0]), 0);
        out[1] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0]), 1);
        out[2] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1), 0);
        out[3] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1), 2);
        out[4] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2), 0);
        out[5] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2), 3);
        out[6] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 0);
        out[7] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 4);
  });
    });
  }
  int ref[8] = {0, 1, 1, 3, 3, 6, 21, 25};
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQUAL(ref[i], output[i]);
  }
}
template <typename T, typename Init, class KernelTest> void test_return_type() {
  cl::sycl::queue deviceQueue;
  {
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.single_task<KernelTest>([=]() {
        T *p = nullptr;
        static_assert(oneapi::dpl::is_same_v<Init, decltype(oneapi::dpl::reduce(
                                                       p, p, Init{},
                                                       oneapi::dpl::plus<>()))>,
                      "");
      });
    });
  }
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;

int main() {
#if (_MSC_VER >= 1912 && _MSVC_LANG >= 201703L) ||                             \
    (_GLIBCXX_RELEASE >= 9 && __GLIBCXX__ >= 20190503 &&                       \
     __cplusplus >= 201703L)
  test_return_type<char, int, KernelTest1>();
  test_return_type<int, int, KernelTest2>();
  test_return_type<float, int, KernelTest3>();
  test_return_type<short, float, KernelTest4>();

  test<input_iterator<const int *>>();
  test<forward_iterator<const int *>>();
  test<bidirectional_iterator<const int *>>();
  test<random_access_iterator<const int *>>();
  test<const int *>();
  std::cout << "done" << std::endl;
#else
  std::cout << "skipping test" << std::endl;
#endif
  return 0;
}
