// -*- C++ -*-
//===-- inner_product.pass.cpp
//--------------------------------------------===//
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

#include <CL/sycl.hpp>
#include <iostream>
#include <oneapi/dpl/numeric>

#include "support/test_iterators.h"

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename _T1, typename _T2> void ASSERT_EQUAL(_T1 &&X, _T2 &&Y) {
  if (X != Y)
    std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y
              << ")" << std::endl;
}

template <class Iter1, class Iter2, class Test> void test() {
  cl::sycl::queue deviceQueue;
  int input1[6] = {1, 2, 3, 4, 5, 6};
  int input2[6] = {6, 5, 4, 3, 2, 1};
  int output[8] = {};
  cl::sycl::range<1> numOfItems1{6};
  cl::sycl::range<1> numOfItems2{8};

  {
    cl::sycl::buffer<int, 1> buffer1(input1, numOfItems1);
    cl::sycl::buffer<int, 1> buffer2(input2, numOfItems1);
    cl::sycl::buffer<int, 1> buffer3(output, numOfItems2);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto in1 = buffer1.get_access<sycl_read>(cgh);
      auto in2 = buffer2.get_access<sycl_read>(cgh);
      auto out = buffer3.get_access<sycl_write>(cgh);
      cgh.single_task<Test>([=]() {
        out[0] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0]),
                                            Iter2(&in2[0]), 0);
        out[1] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0]),
                                            Iter2(&in2[0]), 10);
        out[2] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 1),
                                            Iter2(&in2[0]), 0);
        out[3] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 1),
                                            Iter2(&in2[0]), 10);
        out[4] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 2),
                                            Iter2(&in2[0]), 0);
        out[5] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 2),
                                            Iter2(&in2[0]), 10);
        out[6] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 6),
                                            Iter2(&in2[0]), 0);
        out[7] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 6),
                                            Iter2(&in2[0]), 10);
      });
    });
  }
  const int ref[8] = {0, 10, 6, 16, 16, 26, 56, 66};
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQUAL(ref[i], output[i]);
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
  test<input_iterator<const int *>, input_iterator<const int *>, KernelTest1>();
  test<input_iterator<const int *>, forward_iterator<const int *>,
       KernelTest2>();
  test<input_iterator<const int *>, bidirectional_iterator<const int *>,
       KernelTest3>();
  test<input_iterator<const int *>, random_access_iterator<const int *>,
       KernelTest4>();
  test<input_iterator<const int *>, const int *, KernelTest5>();

  test<forward_iterator<const int *>, input_iterator<const int *>,
       KernelTest6>();
  test<forward_iterator<const int *>, forward_iterator<const int *>,
       KernelTest7>();
  test<forward_iterator<const int *>, bidirectional_iterator<const int *>,
       KernelTest8>();
  test<forward_iterator<const int *>, random_access_iterator<const int *>,
       KernelTest9>();
  test<forward_iterator<const int *>, const int *, KernelTest10>();

  test<bidirectional_iterator<const int *>, input_iterator<const int *>,
       KernelTest11>();
  test<bidirectional_iterator<const int *>, forward_iterator<const int *>,
       KernelTest12>();
  test<bidirectional_iterator<const int *>, bidirectional_iterator<const int *>,
       KernelTest13>();
  test<bidirectional_iterator<const int *>, random_access_iterator<const int *>,
       KernelTest14>();
  test<bidirectional_iterator<const int *>, const int *, KernelTest15>();

  test<random_access_iterator<const int *>, input_iterator<const int *>,
       KernelTest16>();
  test<random_access_iterator<const int *>, forward_iterator<const int *>,
       KernelTest17>();
  test<random_access_iterator<const int *>, bidirectional_iterator<const int *>,
       KernelTest18>();
  test<random_access_iterator<const int *>, random_access_iterator<const int *>,
       KernelTest19>();
  test<random_access_iterator<const int *>, const int *, KernelTest20>();

  test<const int *, input_iterator<const int *>, KernelTest21>();
  test<const int *, forward_iterator<const int *>, KernelTest22>();
  test<const int *, bidirectional_iterator<const int *>, KernelTest23>();
  test<const int *, random_access_iterator<const int *>, KernelTest24>();
  test<const int *, const int *, KernelTest25>();
  std::cout << "done" << std::endl;
  return 0;
}
