// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(iterator)
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace std;
int main()
{
   const int num_elelemts = 10;
   vector<float> result(num_elelemts);
   vector<int> index(num_elelemts);
   vector<float> data(num_elelemts);

   for(int i =0; i< num_elelemts; i++){
      data[i] = i;
      index[i] = num_elelemts - 1 -i;
   }

   using namespace oneapi;
   auto permutation_first = oneapi::dpl::make_permutation_iterator(data.begin(), num_elelemts, index.begin());
   auto permutation_last = permutation_first + num_elelemts;

   //With policy
#if TEST_DPCPP_BACKEND_PRESENT
   std::copy(oneapi::dpl::execution::dpcpp_default, permutation_first, permutation_last, result.begin());
   cout<<"With policy:"<<endl;
   for(int i = 0; i < num_elelemts; i++) 
      cout << result[i] << " ";
   cout<<endl;
#endif //TEST_DPCPP_BACKEND_PRESENT
   
   //Without policy
   std::copy(permutation_first, permutation_last, result.begin());
   cout<<"Without policy:"<<endl;
   for(int i = 0; i < num_elelemts; i++) 
      cout << result[i] << " ";
   cout<<endl;
   
   return TestUtils::done();
}
