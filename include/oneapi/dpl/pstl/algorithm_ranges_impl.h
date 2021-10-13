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

#ifndef _ONEDPL_ALGORITHM_RANGES_IMPL_H
#define _ONEDPL_ALGORITHM_RANGES_IMPL_H

#include <iterator>
#include <type_traits>
#include <functional>
#include <algorithm>
#include <numeric>

#include "glue_algorithm_impl.h"

#include "algorithm_fwd.h"

#include "execution_impl.h"
#include "parallel_backend_utils.h"
#include "unseq_backend_simd.h"

#include "parallel_backend.h"
#include "parallel_impl.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

namespace __ranges
{

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4>
oneapi::dpl::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range3>>
__pattern_reduce_by_key(_ExecutionPolicy&& __exec, _Range1&& __keys_in, _Range2&& __vals_in, _Range3&& __keys_out, _Range4&& __vals_out)
{
    using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;

    auto __n = __keys_in.size();
    __par_backend::__buffer<_ExecutionPolicy, __diff_type> __buf(__n);
    auto __idx = __buf.get();

   using namespace oneapi::dpl::experimental::ranges;

    auto __iota = views::iota((__diff_type)0, __n + 1);
    //TODO: using iterator based API due to there is not  range-based API yet. 
    auto __res = std::copy_if(::std::forward<_ExecutionPolicy>(__exec), __iota.begin(), __iota.end(), __idx,
        [&__keys_in, __n](auto __i) { return __i == 0 || __i == __n || __keys_in[__i] != __keys_in[__i - 1];});

    auto __nres = __res - __idx;

//    for(int i = 0; i < nres-1; ++i)
  //  {
//        C[i] = A[Idx[i]];
  //      std::cout << C[i] << " ";
        //std::cout << Idx[i] << " ";
//    }
//    std::cout << std::endl;
   

    auto __v = views::iota(0, __nres-1);
    //TODO: using iterator based API due to there is not  range-based API yet. 
    ::std::for_each(::std::forward<_ExecutionPolicy>(__exec), __v.begin(), __v.begin() + __v.size(),
        [&__keys_in, &__vals_in, &__keys_out, &__vals_out, &__idx, __exec](auto i)
        {
            __keys_out[i] = __keys_in[__idx[i]];
            __vals_out[i] = reduce(::std::execution::seq, __vals_in.begin() + __idx[i], __vals_in.begin() + __idx[i+1]);
        }
    );

//    for(int i = 0; i < nres-1; ++i)
//    {
//        std::cout << D[i] << " ";
//    }
//    std::cout << std::endl;

     return __nres;
//thrust::pair<int*,int*> new_end;
//new_end = thrust::reduce_by_key(thrust::host, A, A + N, B, C, D);
// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
}

} //__ranges

} //__internal
} //dpl
} //oneapi

#endif /* _ONEDPL_ALGORITHM_RANGES_IMPL_H */
