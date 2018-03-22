/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef __PSTL_parallel_impl_H
#define __PSTL_parallel_impl_H

#include <atomic>
// This header defines the minimum set of parallel routines required to support Parallel STL,
// implemented on top of Intel(R) Threading Building Blocks (Intel(R) TBB) library

namespace pstl {
namespace internal {

//------------------------------------------------------------------------
// parallel_find
//-----------------------------------------------------------------------
/** Return extremum value returned by brick f[i,j) for subranges [i,j) of [first,last)
Each f[i,j) must return a value in [i,j). */
template<class Index, class Brick, class Compare>
Index parallel_find(Index first, Index last, Brick f, Compare comp, bool b_first) {
    typedef typename std::iterator_traits<Index>::difference_type difference_type;
    const difference_type n = last - first;
    difference_type initial_dist = b_first ? n : -1;
    std::atomic<difference_type> extremum(initial_dist);
    // TODO: find out what is better here: parallel_for or parallel_reduce
    par_backend::parallel_for(first, last, [comp, f, first, &extremum](Index i, Index j) {
        // See "Reducing Contention Through Priority Updates", PPoPP '13, for discussion of
        // why using a shared variable scales fairly well in this situation.
        if (comp(i - first, extremum)) {
            Index res = f(i, j);
            // If not 'last' returned then we found what we want so put this to extremum
            if (res != j) {
                const difference_type k = res - first;
                for (difference_type old = extremum; comp(k, old); old = extremum) {
                    extremum.compare_exchange_weak(old, k);
                }
            }
        }
    });
    return extremum != initial_dist ? first + extremum : last;
}

//------------------------------------------------------------------------
// parallel_or
//------------------------------------------------------------------------
//! Return true if brick f[i,j) returns true for some subrange [i,j) of [first,last)
template<class Index, class Brick>
bool parallel_or(Index first, Index last, Brick f) {
    std::atomic<bool> found(false);
    par_backend::parallel_for(first, last, [f, &found](Index i, Index j) {
        if (!found.load(std::memory_order_relaxed) && f(i, j)) {
            found.store(true, std::memory_order_relaxed);
            par_backend::cancel_execution();
        }
    });
    return found;
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_parallel_impl_H */
