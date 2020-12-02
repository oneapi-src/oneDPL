// -*- C++ -*-
//===-- reduce.pass.cpp
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
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/type_traits>

#include "test_iterators.h"

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename _T1, typename _T2> void ASSERT_EQUAL(_T1 &&X, _T2 &&Y) {
  if (X != Y)
    std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y
              << ")" << std::endl;
}

template <class T> class KernelTest1;
template <class T> class KernelTest2;

template <class Iter> void test() {
  cl::sycl::queue deviceQueue;
  int input[6] = {1, 2, 3, 4, 5, 6};
  int output[4] = {};
  cl::sycl::range<1> numOfItems1{6};
  cl::sycl::range<1> numOfItems2{4};

  {
    cl::sycl::buffer<int, 1> buffer1(input, numOfItems1);
    cl::sycl::buffer<int, 1> buffer2(output, numOfItems2);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto in = buffer1.get_access<sycl_read>(cgh);
      auto out = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<class KernelTest1<Iter>>([=]() {
        out[0] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0]));
        out[1] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1));
        out[2] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2));
        out[3] = oneapi::dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6));
  });
    });
  }
  const int ref[4] = {0, 1, 3, 21};
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQUAL(ref[i], output[i]);
  }
}
template <typename T> void test_return_type() {
  cl::sycl::queue deviceQueue;
  {
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.single_task<class KernelTest2<T>>([=]() {

        T *p = nullptr;
        static_assert(
            oneapi::dpl::is_same_v<T, decltype(oneapi::dpl::reduce(p, p))>);
     });
    });
  }
}

int main() {
#if (_MSC_VER >= 1912 && _MSVC_LANG >= 201703L) ||                             \
    (_GLIBCXX_RELEASE >= 9 && __GLIBCXX__ >= 20190503 &&                       \
     __cplusplus >= 201703L)
  test_return_type<char>();
  test_return_type<int>();
  test_return_type<unsigned long>();
  test_return_type<float>();

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
