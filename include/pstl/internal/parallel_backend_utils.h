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

#ifndef __PSTL_parallel_backend_utils_H
#define __PSTL_parallel_backend_utils_H

#include <iterator>
#include <utility>
#include "utils.h"

namespace pstl {
namespace par_backend {

//! Destroy sequence [xs,xe)
struct serial_destroy {
    template<typename RandomAccessIterator>
    void operator()(RandomAccessIterator zs, RandomAccessIterator ze) {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        while (zs != ze) {
            --ze;
            (*ze).~T();
        }
    }
};

//! Merge sequences [xs,xe) and [ys,ye) to output sequence [zs,(xe-xs)+(ye-ys)), using std::move
template<class MoveValues, class MoveSequences>
struct serial_move_merge {
    const std::size_t my_nmerge;
    MoveValues move_values;
    MoveSequences move_sequences;

    explicit serial_move_merge(std::size_t nmerge, MoveValues move_values_, MoveSequences move_sequences_)
        : my_nmerge(nmerge), move_values(move_values_), move_sequences(move_sequences_) {}
    template<class RandomAccessIterator1, class RandomAccessIterator2, class RandomAccessIterator3, class Compare>
    void operator()(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 ys, RandomAccessIterator2 ye, RandomAccessIterator3 zs, Compare comp) {
        auto n = my_nmerge;
        assert(n > 0);
        if (xs != xe) {
            if (ys != ye) {
                for (;;) {
                    if (comp(*ys, *xs)) {
                        move_values(ys, zs);
                        ++zs, --n;
                        if (++ys == ye) {
                            break;
                        }
                        else if (n == 0) {
                            zs = move_sequences(ys, ye, zs);
                            break;
                        }
                        else {}
                    }
                    else {
                        move_values(xs, zs);
                        ++zs, --n;
                        if (++xs == xe) {
                            move_sequences(ys, ye, zs);
                            return;
                        }
                        else if (n == 0) {
                            zs = move_sequences(xs, xe, zs);
                            move_sequences(ys, ye, zs);
                            return;
                        }
                        else {}
                    }
                }
            }
            ys = xs;
            ye = xe;
        }
        move_sequences(ys, ye, zs);
    }
};

template<typename RandomAccessIterator1, typename RandomAccessIterator2>
void init_buf(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 zs, bool bMove) {
    const RandomAccessIterator2 ze = zs + (xe - xs);
    typedef typename std::iterator_traits<RandomAccessIterator2>::value_type T;
    if (bMove) {
        // Initialize the temporary buffer and move keys to it.
        for (; zs != ze; ++xs, ++zs)
            new(&*zs) T(std::move(*xs));
    }
    else {
        // Initialize the temporary buffer
        for (; zs != ze; ++zs)
            new(&*zs) T;
    }
}

template<typename Buf>
class stack {
    typedef typename std::iterator_traits<decltype(Buf(0).get())>::value_type T;
    typedef typename std::iterator_traits<T*>::difference_type difference_type;

    Buf my_buf;
    T* my_ptr;
    difference_type my_maxsize;

    stack(const stack&) = delete;
    void operator=(const stack&) = delete;
public:
    stack(difference_type max_size): my_buf(max_size), my_maxsize(max_size) { my_ptr = my_buf.get(); }
    ~stack() {
        assert(size() <= my_maxsize);
        while(!empty())
            pop();
    }

    const Buf& buffer() const { return my_buf; }
    size_t size() const {
        assert(my_ptr - my_buf.get() <= my_maxsize);
        assert(my_ptr - my_buf.get() >= 0);
        return my_ptr - my_buf.get();
    }
    bool empty() const { assert(my_ptr >= my_buf.get()); return my_ptr == my_buf.get();}
    void push(const T& v) {
        assert(size() < my_maxsize);
        new (my_ptr) T(v); ++my_ptr;
    }
    const T& top() const { return *(my_ptr-1); }
    void pop() {
        assert(my_ptr > my_buf.get());
        --my_ptr; (*my_ptr).~T();
    }
};

} // namespace par_backend
} // namespace pstl

#endif /* __PSTL_parallel_backend_utils_H */
