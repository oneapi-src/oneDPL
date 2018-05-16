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

namespace __pstl {
namespace par_backend {

//! Destroy sequence [xs,xe)
struct serial_destroy {
    template<typename _RandomAccessIterator>
    void operator()(_RandomAccessIterator __zs, _RandomAccessIterator __ze) {
        typedef typename std::iterator_traits<_RandomAccessIterator>::value_type _Tp;
        while (__zs != __ze) {
            --__ze;
            (*__ze).~_Tp();
        }
    }
};

//! Merge sequences [xs,xe) and [ys,ye) to output sequence [zs,(xe-xs)+(ye-ys)), using std::move
struct serial_move_merge {
    template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _OutputIterator, class _Compare>
    void operator()(_RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe, _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye,
                    _OutputIterator __zs, _Compare __comp) {
        if (__xs != __xe) {
            if (__ys != __ye) {
                for (;;)
                    if (__comp(*__ys, *__xs)) {
                        *__zs = std::move(*__ys);
                        ++__zs;
                        if (++__ys == __ye)
                            break;
                    }
                    else {
                        *__zs = std::move(*__xs);
                        ++__zs;
                        if (++__xs == __xe) {
                            std::move(__ys, __ye, __zs);
                            return;
                        }
                    }
            }
            __ys = __xs;
            __ye = __xe;
        }
        std::move(__ys, __ye, __zs);
    }
};

template<typename _RandomAccessIterator1, typename _OutputIterator>
void init_buf(_RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe, _OutputIterator __zs, bool __bMove) {
    const _OutputIterator __ze = __zs + (__xe - __xs);
    typedef typename std::iterator_traits<_OutputIterator>::value_type _Tp;
    if (__bMove) {
        // Initialize the temporary buffer and move keys to it.
        for (; __zs != __ze; ++__xs, ++__zs)
            new(&*__zs) _Tp(std::move(*__xs));
    }
    else {
        // Initialize the temporary buffer
        for (; __zs != __ze; ++__zs)
            new(&*__zs) _Tp;
    }
}

template<typename _Buf>
class stack {
    typedef typename std::iterator_traits<decltype(_Buf(0).get())>::value_type _value_type;
    typedef typename std::iterator_traits<_value_type*>::difference_type _difference_type;

    _Buf _M_buf;
    _value_type* _M_ptr;
    _difference_type _M_maxsize;

    stack(const stack&) = delete;
    void operator=(const stack&) = delete;
public:
    stack(_difference_type __max_size)
      : _M_buf(__max_size)
      , _M_maxsize(__max_size)
    { _M_ptr = _M_buf.get(); }

    ~stack() {
        assert(size() <= _M_maxsize);
        while(!empty())
            pop();
    }

    const _Buf& buffer() const { return _M_buf; }
    size_t size() const {
        assert(_M_ptr - _M_buf.get() <=_M_maxsize);
        assert(_M_ptr - _M_buf.get() >= 0);
        return _M_ptr - _M_buf.get();
    }
    bool empty() const { assert(_M_ptr >= _M_buf.get()); return _M_ptr == _M_buf.get();}
    void push(const _value_type& __v) {
        assert(size() < _M_maxsize);
        new (_M_ptr) _value_type(__v); ++_M_ptr;
    }
    const _value_type& top() const { return *(_M_ptr-1); }
    void pop() {
        assert(_M_ptr > _M_buf.get());
        --_M_ptr; (*_M_ptr).~_value_type();
    }
};

} // namespace par_backend
} // namespace __pstl

#endif /* __PSTL_parallel_backend_utils_H */
