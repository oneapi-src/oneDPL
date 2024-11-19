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

#ifndef _ONEDPL_INTERNAL_OMP_PARALLEL_HISTOGRAM_H
#define _ONEDPL_INTERNAL_OMP_PARALLEL_HISTOGRAM_H

#include "util.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <typename _RandomAccessIterator1, typename _Size, typename _RandomAccessIterator2, typename _Fp>
void
__histogram_body(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _Size __num_bins,
                 _RandomAccessIterator2 __histogram_first, _Fp __f)
{
    using _HistogramValueT = typename ::std::iterator_traits<_RandomAccessIterator2>::value_type;

    const std::size_t __num_threads = omp_get_num_threads();
    const std::size_t __size = __last - __first;

    // Initial partition of the iteration space into chunks. If the range is too small,
    // this will result in a nonsense policy, so we check on the size as well below.
    auto __policy1 = oneapi::dpl::__omp_backend::__chunk_partitioner(__first, __last);
    auto __policy2 = oneapi::dpl::__omp_backend::__chunk_partitioner(__histogram_first, __histogram_first + __num_bins);

    std::vector<std::vector<_HistogramValueT>> __local_histograms(__num_threads, std::vector<_HistogramValueT>(__num_bins, _HistogramValueT{0}));

    //TODO: use histogram output for zeroth thread?

    // main loop
    _PSTL_PRAGMA(omp taskloop shared(__local_histograms))
    for (std::size_t __chunk = 0; __chunk < __policy1.__n_chunks; ++__chunk)
    {
        oneapi::dpl::__omp_backend::__process_chunk(
            __policy1, __first, __chunk, [&](auto __chunk_first, auto __chunk_last) {
                auto __thread_num = omp_get_thread_num();
                __f(__chunk_first, __chunk_last, __local_histograms[__thread_num].begin());
            });
    }

    _PSTL_PRAGMA(omp taskloop shared(__local_histograms))
    for (std::size_t __chunk = 0; __chunk < __policy2.__n_chunks; ++__chunk)
    {
        oneapi::dpl::__omp_backend::__process_chunk(
            __policy2, __histogram_first, __chunk, [&](auto __chunk_first, auto __chunk_last) {
                for (auto __iter = __chunk_first; __iter != __chunk_last; ++__iter)
                {
                    *__iter = __local_histograms[0][__iter - __histogram_first];
                }
                for (auto __iter = __chunk_first; __iter != __chunk_last; ++__iter)
                {
                    for (std::size_t __i = 1; __i < __num_threads; ++__i)
                    {
                        *__iter += __local_histograms[__i][__iter - __histogram_first];
                    }
                }
            });
    }
}

template <class _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _RandomAccessIterator2,
          typename _Fp>
void
__parallel_histogram(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&&, _RandomAccessIterator1 __first,
                     _RandomAccessIterator1 __last, _Size __num_bins, _RandomAccessIterator2 __histogram_first, _Fp __f)
{
    if (omp_in_parallel())
    {
        // We don't create a nested parallel region in an existing parallel
        // region: just create tasks
        oneapi::dpl::__omp_backend::__histogram_body(__first, __last, __num_bins, __histogram_first, __f);
    }
    else
    {
        // Create a parallel region, and a single thread will create tasks
        // for the region.
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single nowait)
        {
            oneapi::dpl::__omp_backend::__histogram_body(__first, __last, __num_bins, __histogram_first, __f);
        }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_HISTOGRAM_H
