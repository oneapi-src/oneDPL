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

#ifndef __PSTL_utils_H
#define __PSTL_utils_H

#include <new>
#include <iterator>

namespace pstl {
namespace internal {

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

template<typename F>
void invoke_if(std::true_type, F f) {
    f();
}

template<typename F>
void invoke_if(std::false_type, F f) {}

template<typename F>
void invoke_if_not(std::false_type, F f) {
    f();
}

template<typename F>
void invoke_if_not(std::true_type, F f) {}

template<typename F1, typename F2>
typename std::result_of<F1()>::type invoke_if_else(std::true_type, F1 f1, F2 f2) {
    return f1();
}

template<typename F1, typename F2>
typename std::result_of<F2()>::type invoke_if_else(std::false_type, F1 f1, F2 f2) {
    return f2();
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::pointer reduce_to_ptr(Iterator it) {
    return std::addressof(*it);
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
    bool operator()( Args&& ... args ) { return !pred(std::forward<Args>(args)...); }
};

template<typename Pred>
class reorder_pred {
    Pred pred;
public:
    explicit reorder_pred( Pred pred_ ) : pred(pred_) {}

    template<typename T>
    bool operator()(T&& a, T&& b) { return pred(std::forward<T>(b), std::forward<T>(a)); }
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

//! "<" comparison.
class pstl_less {
public:
    explicit pstl_less() {}

    template<typename X, typename Y>
    bool operator()(X&& x, Y&& y) const { return std::forward<X>(x) < std::forward<Y>(y); }
};

//! Like a polymorphic lambda for pred(...,value)
template<typename T, typename Predicate>
class equal_value_by_pred {
    const T& value;
    Predicate pred;
public:
    equal_value_by_pred(const T& value_, Predicate pred_) : value(value_), pred(pred_) {}

    template<typename Arg>
    bool operator()(Arg&& arg) { return pred(std::forward<Arg>(arg), value); }
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

template <typename ForwardIterator, typename Compare>
ForwardIterator cmp_iterators_by_values(ForwardIterator a, ForwardIterator b, Compare comp) {
    if(a < b) { // we should return closer iterator
        return comp(*b, *a) ? b : a;
    } else {
        return comp(*a, *b) ? a : b;
    }
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_utils_H */
