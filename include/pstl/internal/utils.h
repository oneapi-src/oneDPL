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

template<typename _Fp>
typename std::result_of<_Fp()>::type except_handler(_Fp __f) {
    try {
        return __f();
    }
    catch(const std::bad_alloc&) {
        throw; // re-throw bad_alloc according to the standard [algorithms.parallel.exceptions]
    }
    catch(...) {
        std::terminate(); // Good bye according to the standard [algorithms.parallel.exceptions]
    }
}

template<typename _Fp>
void invoke_if(std::true_type, _Fp __f) {
    __f();
}

template<typename _Fp>
void invoke_if(std::false_type, _Fp __f) {}

template<typename _Fp>
void invoke_if_not(std::false_type, _Fp __f) {
    __f();
}

template<typename _Fp>
void invoke_if_not(std::true_type, _Fp __f) {}

template<typename _F1, typename _F2>
typename std::result_of<_F1()>::type invoke_if_else(std::true_type, _F1 __f1, _F2 __f2) {
    return __f1();
}

template<typename _F1, typename _F2>
typename std::result_of<_F2()>::type invoke_if_else(std::false_type, _F1 __f1, _F2 __f2) {
    return __f2();
}

template<typename _Iterator>
typename std::iterator_traits<_Iterator>::pointer reduce_to_ptr(_Iterator __it) {
    return std::addressof(*__it);
}

//! Unary operator that returns reference to its argument.
struct no_op {
    template<typename _Tp>
    _Tp&& operator()(_Tp&& __a) const { return std::forward<_Tp>(__a); }
};

//! Logical negation of a predicate
template<typename _Pred>
class not_pred {
    _Pred _M_pred;

public:
    explicit not_pred( _Pred __pred ) : _M_pred(__pred) {}

    template<typename ... _Args>
    bool operator()( _Args&& ... __args ) { return !_M_pred(std::forward<_Args>(__args)...); }
};

template<typename _Pred>
class reorder_pred {
    _Pred _M_pred;
public:
    explicit reorder_pred( _Pred __pred ) : _M_pred(__pred) {}

    template<typename _Tp>
    bool operator()(_Tp&& __a, _Tp&& __b) { return _M_pred(std::forward<_Tp>(__b), std::forward<_Tp>(__a)); }
};

//! "==" comparison.
/** Not called "equal" to avoid (possibly unfounded) concerns about accidental invocation via
    argument-dependent name lookup by code expecting to find the usual std::equal. */
class pstl_equal {
public:
    explicit pstl_equal() {}

    template<typename _Xp, typename _Yp>
    bool operator()( _Xp&& __x, _Yp&& __y ) const { return std::forward<_Xp>(__x) == std::forward<_Yp>(__y); }
};

//! "<" comparison.
class pstl_less {
public:
    explicit pstl_less() {}

    template<typename _Xp, typename _Yp>
    bool operator()(_Xp&& __x, _Yp&& __y) const { return std::forward<_Xp>(__x) < std::forward<_Yp>(__y); }
};

//! Like a polymorphic lambda for pred(...,value)
template<typename _Tp, typename _Predicate>
class equal_value_by_pred {
    const _Tp& _M_value;
    _Predicate _M_pred;
public:
    equal_value_by_pred(const _Tp& __value, _Predicate __pred)
      : _M_value(__value)
      , _M_pred(__pred) {}

    template<typename _Arg>
    bool operator()(_Arg&& __arg) { return _M_pred(std::forward<_Arg>(__arg), _M_value); }
};

//! Like a polymorphic lambda for ==value
template<typename _Tp>
class equal_value {
    const _Tp& _M_value;
public:
    explicit equal_value( const _Tp& __value ) : _M_value(__value) {}

    template<typename _Arg>
    bool operator()( _Arg&& __arg ) const { return std::forward<_Arg>(__arg) == _M_value; }
};

//! Logical negation of ==value
template<typename _Tp>
class not_equal_value {
    const _Tp& _M_value;
public:
    explicit not_equal_value( const _Tp& __value ) : _M_value(__value) {}

    template<typename _Arg>
    bool operator()( _Arg&& __arg ) const { return !(std::forward<_Arg>(__arg) == _M_value); }
};

template <typename _ForwardIterator, typename _Compare>
_ForwardIterator cmp_iterators_by_values(_ForwardIterator __a, _ForwardIterator __b, _Compare __comp) {
    if(__a < __b) { // we should return closer iterator
        return __comp(*__b, *__a) ? __b : __a;
    } else {
        return __comp(*__a, *__b) ? __a : __b;
    }
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_utils_H */
