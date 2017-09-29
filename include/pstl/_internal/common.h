/*
    Copyright (c) 2017 Intel Corporation

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

#ifndef __PSTL_common_H
#define __PSTL_common_H

// Header contains implementation of common utilities.

#if __PSTL_USE_TBB
#include <tbb/tbb_thread.h>
#endif

namespace __icp_algorithm {

static int __PSTL_get_workers_num() {
#if __PSTL_USE_TBB
    return tbb::tbb_thread::hardware_concurrency();
#else
    __PSTL_PRAGMA_MESSAGE("Backend was not specified");
    return 1;
#endif
}

// FIXME - make grain_size use compiler information, or make parallel_for/parallel_transform_reduce use introspection for
// better estimate.

//! Helper for parallel_for and parallel_reduce
template<typename DifferenceType>
DifferenceType __PSTL_grain_size( DifferenceType m ) {
    const size_t oversub = 8;
    int n = __PSTL_get_workers_num();
    m /= oversub*n;
    const int min_grain = 1;
    const int max_grain = 1<<16;
    if( m<min_grain )
        m = min_grain;
    else if( m>max_grain )
        m = max_grain;
    return m;
}

//! Raw memory buffer with automatic freeing and no exceptions.
/** Some of our algorithms need to start with raw memory buffer,
not an initialize array, because initialization/destruction
would make the span be at least O(N). */
class raw_buffer {
    void* ptr;
    raw_buffer(const raw_buffer&) = delete;
    void operator=(const raw_buffer&) = delete;
public:
    //! Try to obtain buffer of given size.
    raw_buffer(size_t bytes): ptr(operator new(bytes, std::nothrow)) {}
    //! True if buffer was successfully obtained, zero otherwise.
    operator bool() const { return ptr != NULL; }
    //! Return pointer to buffer, or  NULL if buffer could not be obtained.
    void* get() const { return ptr; }
    //! Destroy buffer
    ~raw_buffer() { operator delete(ptr); }
};

template<typename F>
typename std::result_of<F()>::type except_handler(F f) {
    try {
        return f();
    }
    catch(const std::bad_alloc&) {
        throw; // re-throw bad_alloc according to 25.2.4.1 [algorithms.parallel.exceptions]
    }
    catch(...) {
        std::terminate(); // Good bye according to 25.2.4.2 [algorithms.parallel.exceptions]
    }
}

//! Unary operator that returns reference to its argument.
struct no_op {
    template<typename T>
    T& operator()(T& a) const { return a; }
};

//! Logical negation of a predicate
template<typename Pred>
class not_pred {
    Pred pred;
public:
    explicit not_pred( Pred pred_ ) : pred(pred_) {}

    template<typename ... Args>
    bool operator()( Args&& ... args ) const { return !pred(std::forward<Args>(args)...); }
};

template<typename Pred>
class reorder_pred {
    Pred pred;
public:
    explicit reorder_pred( Pred pred_ ) : pred(pred_) {}

    template<typename T>
    bool operator()(T&& a, T&& b) const { return pred(std::forward<T>(b), std::forward<T>(a)); }
};

//! "==" comparison.
/** Not called "equal" to avoid (possibly unfounded) concerns about accidental invocation via
    argument-dependent name lookup by code expecting to find the usual std::equal. */
class pstl_equal {
public:
    explicit pstl_equal() {}

    template<typename X, typename Y>
    bool operator()( X&& x, Y&& y ) const { return std::forward<X>(x)==std::forward<Y>(y); }
};

//! Like a polymorphic lambda for ==value
template<typename T>
class equal_value {
    const T& value;
public:
    explicit equal_value( const T& value_ ) : value(value_) {}

    template<typename Arg>
    bool operator()( Arg&& arg ) const { return std::forward<Arg>(arg)==value; }
};

//! Logical negation of ==value
template<typename T>
class not_equal_value {
    const T& value;
public:
    explicit not_equal_value( const T& value_ ) : value(value_) {}

    template<typename Arg>
    bool operator()( Arg&& arg ) const { return !(std::forward<Arg>(arg)==value); }
};

} /* namespace __icp_algorithm */

#endif /* __PSTL_common_H */
