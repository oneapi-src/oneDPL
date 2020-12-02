// -*- C++ -*-
//===-- transform_reduce_iter_iter_iter_init.pass.cpp
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
#include <oneapi/dpl/iterator>
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

template <class Iter1, class Iter2, class KernelName> void test() {
  cl::sycl::queue deviceQueue;
  int input1[6] = {1, 2, 3, 4, 5, 6};
  unsigned int input2[6] = {2, 4, 6, 8, 10, 12};
  int output[8] = {};
  cl::sycl::range<1> numOfItems1{6};
  cl::sycl::range<1> numOfItems2{8};
  {
    cl::sycl::buffer<int, 1> buffer1(input1, numOfItems1);
    cl::sycl::buffer<unsigned int, 1> buffer2(input2, numOfItems1);
    cl::sycl::buffer<int, 1> buffer3(output, numOfItems2);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto in1 = buffer1.get_access<sycl_read>(cgh);
      auto in2 = buffer2.get_access<sycl_read>(cgh);
      auto out = buffer3.get_access<sycl_write>(cgh);
      cgh.single_task<KernelName>([=]() {
        out[0] = oneapi::dpl::transform_reduce(Iter1(&in1[0]), Iter1(&in1[0]),
                                               Iter2(&in2[0]), 0);
        out[1] = oneapi::dpl::transform_reduce(Iter2(&in2[0]), Iter2(&in2[0]),
                                               Iter1(&in1[0]), 1);
        out[2] = oneapi::dpl::transform_reduce(
            Iter1(&in1[0]), Iter1(&in1[0] + 1), Iter2(&in2[0]), 0);
        out[3] = oneapi::dpl::transform_reduce(
            Iter2(&in2[0]), Iter2(&in2[0] + 1), Iter1(&in1[0]), 2);
        out[4] = oneapi::dpl::transform_reduce(
            Iter1(&in1[0]), Iter1(&in1[0] + 2), Iter2(&in2[0]), 0);
        out[5] = oneapi::dpl::transform_reduce(
            Iter2(&in2[0]), Iter2(&in2[0] + 2), Iter1(&in1[0]), 3);
        out[6] = oneapi::dpl::transform_reduce(
            Iter1(&in1[0]), Iter1(&in1[0] + 6), Iter2(&in2[0]), 0);
        out[7] = oneapi::dpl::transform_reduce(
            Iter2(&in2[0]), Iter2(&in2[0] + 6), Iter1(&in1[0]), 4);
      });
    });
  }
  int ref[8] = {0, 1, 2, 4, 10, 13, 182, 186};
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQUAL(ref[i], output[i]);
  }
}

template <typename T, typename Init, class KernelName> void test_return_type() {
  cl::sycl::queue deviceQueue;
  {
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.single_task<KernelName>([=]() {
        T *p = nullptr;
        static_assert(
            oneapi::dpl::is_same_v<Init, decltype(oneapi::dpl::transform_reduce(
                                             p, p, p, Init{}))>);
      });
    });
  }
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;
class KernelTest11;
class KernelTest12;
class KernelTest13;
class KernelTest14;
class KernelTest15;
class KernelTest16;
class KernelTest17;
class KernelTest18;
class KernelTest19;
class KernelTest20;
class KernelTest21;
class KernelTest22;
class KernelTest23;
class KernelTest24;
class KernelTest25;

int main() {
#if (_MSC_VER >= 1912 && _MSVC_LANG >= 201703L) ||                             \
    (_GLIBCXX_RELEASE >= 9 && __GLIBCXX__ >= 20190503 &&                       \
     __cplusplus >= 201703L)
  test_return_type<char, int, KernelTest1>();
  test_return_type<int, int, KernelTest2>();
  test_return_type<int, unsigned long, KernelTest3>();
  test_return_type<float, int, KernelTest4>();
  test_return_type<short, float, KernelTest5>();

  //  All the iterator categories
  test<input_iterator<const int *>, input_iterator<const unsigned int *>,
       KernelTest6>();

  test<input_iterator<const int *>, forward_iterator<const unsigned int *>,
       KernelTest7>();
  test<input_iterator<const int *>,
       bidirectional_iterator<const unsigned int *>, KernelTest8>();
  test<input_iterator<const int *>,
       random_access_iterator<const unsigned int *>, KernelTest9>();

  test<forward_iterator<const int *>, input_iterator<const unsigned int *>,
       KernelTest10>();
  test<forward_iterator<const int *>, forward_iterator<const unsigned int *>,
       KernelTest11>();
  test<forward_iterator<const int *>,
       bidirectional_iterator<const unsigned int *>, KernelTest12>();
  test<forward_iterator<const int *>,
       random_access_iterator<const unsigned int *>, KernelTest13>();

  test<bidirectional_iterator<const int *>,
       input_iterator<const unsigned int *>, KernelTest14>();
  test<bidirectional_iterator<const int *>,
       forward_iterator<const unsigned int *>, KernelTest15>();
  test<bidirectional_iterator<const int *>,
       bidirectional_iterator<const unsigned int *>, KernelTest16>();
  test<bidirectional_iterator<const int *>,
       random_access_iterator<const unsigned int *>, KernelTest17>();

  test<random_access_iterator<const int *>,
       input_iterator<const unsigned int *>, KernelTest18>();
  test<random_access_iterator<const int *>,
       forward_iterator<const unsigned int *>, KernelTest19>();
  test<random_access_iterator<const int *>,
       bidirectional_iterator<const unsigned int *>, KernelTest20>();
  test<random_access_iterator<const int *>,
       random_access_iterator<const unsigned int *>, KernelTest21>();

  //  Just plain pointers (const vs. non-const, too)
  test<const int *, const unsigned int *, KernelTest22>();
  test<const int *, unsigned int *, KernelTest23>();
  test<int *, const unsigned int *, KernelTest24>();
  test<int *, unsigned int *, KernelTest25>();
  std::cout << "done" << std::endl;
#else
  std::cout << "skipping test" << std::endl;
#endif
  return 0;
}
