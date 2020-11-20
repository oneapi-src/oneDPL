// -*- C++ -*-
//===-- lower_bound_cpu.pass.cpp --------------------------------------------===//
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

#include <iostream>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

int main()
{
    int key[10] = {0, 2, 2, 2, 3, 3, 3, 3, 6, 6};
    int val[5] = {0, 2, 4, 7, 6};
    int res[5];
  
     // call algorithm
     oneapi::dpl::binary_search(oneapi::dpl::execution::par, std::begin(key), std::end(key), std::begin(val), std::end(val), std::begin(res), std::less<int>());

     //check data
     if((res[0] == true) && (res[1] == true) && (res[2] ==false) && (res[3] == false) && (res[4] == true))
         std::cout << "done" << std::endl;
     else
         std::cout << "Values do not match." << std::endl;

     return 0;
}
