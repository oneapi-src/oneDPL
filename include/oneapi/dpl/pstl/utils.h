// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

#ifndef _ONEDPL_UTILS_H
#define _ONEDPL_UTILS_H

#include "pstl_config.h"

#include <new>
#include <iterator>
#include <type_traits>

#if _PSTL_CPP14_INTEGER_SEQUENCE_PRESENT
#    include <utility>
#endif

#if _PSTL_BACKEND_SYCL
#    include <CL/sycl.hpp>
#    include "hetero/dpcpp/sycl_iterator.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename Iterator>
using is_const_iterator =
    typename ::std::is_const<typename ::std::remove_pointer<typename ::std::iterator_traits<Iterator>::pointer>::type>;

template <typename _Fp>
auto
__except_handler(_Fp __f) -> decltype(__f())
{
    try
    {
        return __f();
    }
    catch (const ::std::bad_alloc&)
    {
        throw; // re-throw bad_alloc according to the standard [algorithms.parallel.exceptions]
    }
    catch (...)
    {
        ::std::terminate(); // Good bye according to the standard [algorithms.parallel.exceptions]
    }
}

template <typename _Fp>
void
__invoke_if(::std::true_type, _Fp __f)
{
    __f();
}

template <typename _Fp>
void
__invoke_if(::std::false_type, _Fp __f)
{
}

template <typename _Fp>
void
__invoke_if_not(::std::false_type, _Fp __f)
{
    __f();
}

template <typename _Fp>
void
__invoke_if_not(::std::true_type, _Fp __f)
{
}

template <typename _F1, typename _F2>
auto
__invoke_if_else(::std::true_type, _F1 __f1, _F2 __f2) -> decltype(__f1())
{
    return __f1();
}

template <typename _F1, typename _F2>
auto
__invoke_if_else(::std::false_type, _F1 __f1, _F2 __f2) -> decltype(__f2())
{
    return __f2();
}

template <typename _Op>
struct __invoke_unary_op
{
    mutable _Op __op;

    template <typename _Input, typename _Output>
    void
    operator()(_Input&& __x, _Output&& __y) const
    {
        __y = __op(::std::forward<_Input>(__x));
    }
};

//! Unary operator that returns reference to its argument.
struct __no_op
{
    template <typename _Tp>
    _Tp&&
    operator()(_Tp&& __a) const
    {
        return ::std::forward<_Tp>(__a);
    }
};

//! Logical negation of a predicate
template <typename _Pred>
class __not_pred
{
    _Pred _M_pred;

  public:
    explicit __not_pred(_Pred __pred) : _M_pred(__pred) {}

    template <typename... _Args>
    bool
    operator()(_Args&&... __args)
    {
        return !_M_pred(::std::forward<_Args>(__args)...);
    }
};

template <typename _Pred>
class __reorder_pred
{
    mutable _Pred _M_pred;

  public:
    explicit __reorder_pred(_Pred __pred) : _M_pred(__pred) {}

    template <typename _FTp, typename _STp>
    bool
    operator()(_FTp&& __a, _STp&& __b) const
    {
        return _M_pred(::std::forward<_STp>(__b), ::std::forward<_FTp>(__a));
    }
};

//! "==" comparison.
/** Not called "equal" to avoid (possibly unfounded) concerns about accidental invocation via
    argument-dependent name lookup by code expecting to find the usual ::std::equal. */
class __pstl_equal
{
  public:
    explicit __pstl_equal() {}

    template <typename _Xp, typename _Yp>
    bool
    operator()(_Xp&& __x, _Yp&& __y) const
    {
        return ::std::forward<_Xp>(__x) == ::std::forward<_Yp>(__y);
    }
};

//! "<" comparison.
class __pstl_less
{
  public:
    template <typename _Xp, typename _Yp>
    bool
    operator()(_Xp&& __x, _Yp&& __y) const
    {
        return ::std::forward<_Xp>(__x) < ::std::forward<_Yp>(__y);
    }
};

//! ">" comparison.
class __pstl_greater
{
  public:
    template <typename _Xp, typename _Yp>
    bool
    operator()(_Xp&& __x, _Yp&& __y) const
    {
        return ::std::forward<_Xp>(__x) > ::std::forward<_Yp>(__y);
    }
};

//! General "+" operation
class __pstl_plus
{
  public:
    template <typename _Xp, typename _Yp>
    auto
    operator()(_Xp&& __x, _Yp&& __y) const -> decltype(::std::forward<_Xp>(__x) + ::std::forward<_Yp>(__y))
    {
        return ::std::forward<_Xp>(__x) + ::std::forward<_Yp>(__y);
    }
};

//! min calculation.
class __pstl_min
{
  public:
    template <typename _Xp, typename _Yp>
    auto
    operator()(_Xp&& __x, _Yp&& __y) const
        -> decltype((::std::forward<_Xp>(__x) < ::std::forward<_Yp>(__y)) ? ::std::forward<_Xp>(__x)
                                                                          : ::std::forward<_Yp>(__y))
    {
        return (::std::forward<_Xp>(__x) < ::std::forward<_Yp>(__y)) ? ::std::forward<_Xp>(__x)
                                                                     : ::std::forward<_Yp>(__y);
    }
};

//! max calculation.
class __pstl_max
{
  public:
    template <typename _Xp, typename _Yp>
    auto
    operator()(_Xp&& __x, _Yp&& __y) const
        -> decltype((::std::forward<_Xp>(__x) > ::std::forward<_Yp>(__y)) ? ::std::forward<_Xp>(__x)
                                                                          : ::std::forward<_Yp>(__y))
    {
        return (::std::forward<_Xp>(__x) > ::std::forward<_Yp>(__y)) ? ::std::forward<_Xp>(__x)
                                                                     : ::std::forward<_Yp>(__y);
    }
};

//! Like a polymorphic lambda for pred(...,value)
template <typename _Tp, typename _Predicate>
class __equal_value_by_pred
{
    const _Tp _M_value;
    _Predicate _M_pred;

  public:
    __equal_value_by_pred(const _Tp& __value, _Predicate __pred) : _M_value(__value), _M_pred(__pred) {}

    template <typename _Arg>
    bool
    operator()(_Arg&& __arg)
    {
        return _M_pred(::std::forward<_Arg>(__arg), _M_value);
    }
};

//! Like a polymorphic lambda for ==value
template <typename _Tp>
class __equal_value
{
    const _Tp _M_value;

  public:
    explicit __equal_value(const _Tp& __value) : _M_value(__value) {}

    template <typename _Arg>
    bool
    operator()(_Arg&& __arg) const
    {
        return ::std::forward<_Arg>(__arg) == _M_value;
    }
};

//! Logical negation of ==value
template <typename _Tp>
class __not_equal_value
{
    const _Tp _M_value;

  public:
    explicit __not_equal_value(const _Tp& __value) : _M_value(__value) {}

    template <typename _Arg>
    bool
    operator()(_Arg&& __arg) const
    {
        return !(::std::forward<_Arg>(__arg) == _M_value);
    }
};

//! Like ::std::next, but with specialization for dpcpp case
template <typename _Iter>
_Iter
__pstl_next(_Iter __iter, typename ::std::iterator_traits<_Iter>::difference_type __n = 1)
{
    return ::std::next(__iter, __n);
}

#if _PSTL_BACKEND_SYCL
template <cl::sycl::access::mode _Mode, typename... _Params>
dpstd::__internal::sycl_iterator<_Mode, _Params...>
__pstl_next(
    dpstd::__internal::sycl_iterator<_Mode, _Params...> __iter,
    typename ::std::iterator_traits<dpstd::__internal::sycl_iterator<_Mode, _Params...>>::difference_type __n = 1)
{
    return __iter + __n;
}
#endif

template <typename _ForwardIterator, typename _Compare>
_ForwardIterator
__cmp_iterators_by_values(_ForwardIterator __a, _ForwardIterator __b, _Compare __comp)
{
    if (__a < __b)
    { // we should return closer iterator
        return __comp(*__b, *__a) ? __b : __a;
    }
    else
    {
        return __comp(*__a, *__b) ? __a : __b;
    }
}

template <typename _Acc, typename _Size1, typename _Value, typename _Compare>
_Size1
__pstl_lower_bound(_Acc __acc, _Size1 __first, _Size1 __last, const _Value& __value, _Compare __comp)
{
    auto __n = __last - __first;
    auto __cur = __n;
    _Size1 __it;
    while (__n > 0)
    {
        __it = __first;
        __cur = __n / 2;
        __it += __cur;
        if (__comp(__acc[__it], __value))
        {
            __n -= __cur + 1, __first = ++__it;
        }
        else
            __n = __cur;
    }
    return __first;
}

template <typename _Acc, typename _Size1, typename _Value, typename _Compare>
_Size1
__pstl_upper_bound(_Acc __acc, _Size1 __first, _Size1 __last, const _Value& __value, _Compare __comp)
{
    return __pstl_lower_bound(__acc, __first, __last, __value,
                              oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__reorder_pred<_Compare>>{
                                  oneapi::dpl::__internal::__reorder_pred<_Compare>{__comp}});
}

// Searching for the first element strongly greater than a passed value - right bound
template <typename _Buffer, typename _Index, typename _Value, typename _Compare>
_Index
__pstl_right_bound(_Buffer& __a, _Index __first, _Index __last, const _Value& __val, _Compare __comp)
{
    return __pstl_upper_bound(__a, __first, __last, __val, __comp);
}

template <typename _IntType, typename _Acc>
struct _ReverseCounter
{
    typedef typename ::std::make_signed<_IntType>::type difference_type;

    _IntType __my_cn;

    _ReverseCounter&
    operator++()
    {
        --__my_cn;
        return *this;
    }

    template <typename _DiffType>
    _ReverseCounter&
    operator+=(_DiffType __val)
    {
        __my_cn -= __val;
        return *this;
    }

    difference_type
    operator-(const _ReverseCounter& __a)
    {
        return __a.__my_cn - __my_cn;
    }

    operator _IntType() { return __my_cn; }

// TODO: Temporary hotfix. Investigate the necessity of _ReverseCounter
// Investigate potential user types convertible to integral
// This is the compile-time trick where we define the conversion operator to sycl::id
// conditionally. If we can call accessor::operator[] with the type that converts to the
// same integral type as _ReverseCounter (it means that we can call accessor::operator[]
// with the _ReverseCounter itself) then we don't need conversion operator to sycl::id.
// Otherwise, we define conversion operator to sycl::id.
#if _PSTL_BACKEND_SYCL
    struct __integral
    {
        operator _IntType();
    };

    template <typename _Tp>
    static auto
    __check_braces(int) -> decltype(::std::declval<_Tp>()[::std::declval<__integral>()], ::std::false_type{});

    template <typename _Tp>
    static auto
    __check_braces(...) -> ::std::true_type;

    class __private_class;

    operator typename ::std::conditional<decltype(__check_braces<_Acc>(0))::value, cl::sycl::id<1>,
                                         __private_class>::type()
    {
        return cl::sycl::id<1>(__my_cn);
    }
#endif
};

// Reverse searching for the first element strongly less than a passed value - left bound
template <typename _Buffer, typename _Index, typename _Value, typename _Compare>
_Index
__pstl_left_bound(_Buffer& __a, _Index __first, _Index __last, const _Value& __val, _Compare __comp)
{
    auto __beg = _ReverseCounter<_Index, _Buffer>{__last - 1};
    auto __end = _ReverseCounter<_Index, _Buffer>{__first - 1};

    return __pstl_upper_bound(__a, __beg, __end, __val, __reorder_pred<_Compare>{__comp});
}

#if _PSTL_CPP14_INTEGER_SEQUENCE_PRESENT

template <::std::size_t... _Sp>
using __index_sequence = ::std::index_sequence<_Sp...>;
template <::std::size_t _Np>
using __make_index_sequence = ::std::make_index_sequence<_Np>;

#else

template <::std::size_t... _Sp>
class __index_sequence
{
};

template <::std::size_t _Np, ::std::size_t... _Sp>
struct __make_index_sequence_impl : __make_index_sequence_impl<_Np - 1, _Np - 1, _Sp...>
{
};

template <::std::size_t... _Sp>
struct __make_index_sequence_impl<0, _Sp...>
{
    using type = __index_sequence<_Sp...>;
};

template <::std::size_t _Np>
using __make_index_sequence = typename oneapi::dpl::__internal::__make_index_sequence_impl<_Np>::type;

#endif /* _PSTL_CPP14_INTEGER_SEQUENCE_PRESENT */

// Required to support GNU libstdc++ below 5.x
template <typename _Tp>
using __has_trivial_copy_assignemnt =
#if _PSTL_CPP11_IS_TRIVIALLY_COPY_ASSIGNABLE_PRESENT
    ::std::is_trivially_copy_assignable<
#else
    ::std::has_trivial_copy_assign<
#endif /* _PSTL_CPP11_IS_TRIVIALLY_COPY_ASSIGNABLE_PRESENT */
        _Tp>;

// Aliases for adjacent_find compile-time dispatching
using __or_semantic = ::std::true_type;
using __first_semantic = ::std::false_type;

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_UTILS_H */
