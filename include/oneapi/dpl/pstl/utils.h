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

#ifndef _ONEDPL_UTILS_H
#define _ONEDPL_UTILS_H

#include "onedpl_config.h"

#include <new>
#include <iterator>
#include <type_traits>
#include <tuple>
#include <utility>
#include <climits>

#if _ONEDPL_BACKEND_SYCL
#    include "hetero/dpcpp/sycl_defs.h"
#    include "hetero/dpcpp/sycl_iterator.h"
#endif

#if __has_include(<bit>)
#    include <bit>
#endif

#if !(__cpp_lib_bit_cast >= 201806L)
#    ifndef __has_builtin
#        define __has_builtin(__x) 0
#    endif
#    include <cstring> // memcpy
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
    operator()(_Args&&... __args) const
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

//! custom assignment operator used in copy_if and other algorithms using predicates
class __pstl_assign
{
  public:
    // rvalue reference used for output parameter to allow assignment of std::tuple of references.
    // The output is the second argument because the output range is passed to the algorithm as the second range.
    template <typename _Xp, typename _Yp>
    void
    operator()(const _Xp& __x, _Yp&& __y) const
    {
        ::std::forward<_Yp>(__y) = __x;
    }
};

//! "==" comparison.
/** Not called "equal" to avoid (possibly unfounded) concerns about accidental invocation via
    argument-dependent name lookup by code expecting to find the usual ::std::equal. */
class __pstl_equal
{
  public:
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
    operator()(_Arg&& __arg) const
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

template <typename _Pred>
class __transform_functor
{
    _Pred _M_pred;

  public:
    explicit __transform_functor(_Pred __pred) : _M_pred(__pred) {}

    template <typename _Input1Type, typename _Input2Type, typename _OutputType>
    void
    operator()(const _Input1Type& x, const _Input2Type& y, _OutputType& z) const
    {
        z = _M_pred(x, y);
    }
};

template <typename _Tp, typename _Pred>
class __replace_functor
{
    const _Tp _M_value;
    _Pred _M_pred;

  public:
    __replace_functor(const _Tp& __value, _Pred __pred) : _M_value(__value), _M_pred(__pred) {}

    template <typename _OutputType>
    void
    operator()(_OutputType& __elem) const
    {
        if (_M_pred(__elem))
            __elem = _M_value;
    }
};

template <typename _Tp, typename _Pred>
class __replace_copy_functor
{
    const _Tp _M_value;
    _Pred _M_pred;

  public:
    __replace_copy_functor(const _Tp& __value, _Pred __pred) : _M_value(__value), _M_pred(__pred) {}

    template <typename _InputType, typename _OutputType>
    void
    operator()(const _InputType& __x, _OutputType& __y) const
    {
        __y = _M_pred(__x) ? _M_value : __x;
    }
};

//! Like ::std::next, but with specialization for dpcpp case
template <typename _Iter>
_Iter
__pstl_next(_Iter __iter, typename ::std::iterator_traits<_Iter>::difference_type __n = 1)
{
    return ::std::next(__iter, __n);
}

#if _ONEDPL_BACKEND_SYCL
template <sycl::access::mode _Mode, typename... _Params>
oneapi::dpl::__internal::sycl_iterator<_Mode, _Params...>
__pstl_next(
    oneapi::dpl::__internal::sycl_iterator<_Mode, _Params...> __iter,
    typename ::std::iterator_traits<oneapi::dpl::__internal::sycl_iterator<_Mode, _Params...>>::difference_type __n = 1)
{
    return __iter + __n;
}
#endif

template <typename _ForwardIterator, typename _Compare, typename _CompareIt>
_ForwardIterator
__cmp_iterators_by_values(_ForwardIterator __a, _ForwardIterator __b, _Compare __comp, _CompareIt __comp_it)
{
    if (__comp_it(__a, __b))
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
#if _ONEDPL_BACKEND_SYCL
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

    operator typename ::std::conditional<decltype(__check_braces<_Acc>(0))::value, sycl::id<1>, __private_class>::type()
    {
        return sycl::id<1>(__my_cn);
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

// Aliases for adjacent_find compile-time dispatching
using __or_semantic = ::std::true_type;
using __first_semantic = ::std::false_type;

// Define __void_type via this structure to handle redefinition issue.
// See CWG 1558 for information about it.
template <typename... _Ts>
struct __make_void_type
{
    using __type = void;
};

template <typename... _Ts>
using __void_type = typename __make_void_type<_Ts...>::__type;

// is_callable_object
template <typename _Tp, typename = void>
struct __is_callable_object : ::std::false_type
{
};

template <typename _Tp>
struct __is_callable_object<_Tp, __void_type<decltype(&_Tp::operator())>> : ::std::true_type
{
};

// is_pointer_to_const_member
template <typename _Tp>
struct __is_pointer_to_const_member_impl : ::std::false_type
{
};

template <typename _R, typename _U, typename... _Args>
struct __is_pointer_to_const_member_impl<_R (_U::*)(_Args...) const> : ::std::true_type
{
};

template <typename _R, typename _U, typename... _Args>
struct __is_pointer_to_const_member_impl<_R (_U::*)(_Args...) const noexcept> : ::std::true_type
{
};

template <typename _Tp, bool = __is_callable_object<_Tp>::value>
struct __is_pointer_to_const_member : ::std::false_type
{
};

template <typename _Tp>
struct __is_pointer_to_const_member<_Tp, true> : __is_pointer_to_const_member_impl<decltype(&_Tp::operator())>
{
};

// is_const_callable_object to check whether we call const or non-const object
template <typename _Tp>
using __is_const_callable_object =
    ::std::integral_constant<bool, __is_callable_object<_Tp>::value && __is_pointer_to_const_member<_Tp>::value>;

struct __next_to_last
{
    template <typename _Iterator>
    typename ::std::enable_if<::std::is_base_of<::std::random_access_iterator_tag,
                                                typename ::std::iterator_traits<_Iterator>::iterator_category>::value,
                              _Iterator>::type
    operator()(_Iterator __it, _Iterator __last, typename ::std::iterator_traits<_Iterator>::difference_type __n)
    {
        return __n > __last - __it ? __last : __it + __n;
    }

    template <typename _Iterator>
    typename ::std::enable_if<!::std::is_base_of<::std::random_access_iterator_tag,
                                                 typename ::std::iterator_traits<_Iterator>::iterator_category>::value,
                              _Iterator>::type
    operator()(_Iterator __it, _Iterator __last, typename ::std::iterator_traits<_Iterator>::difference_type __n)
    {
        for (; --__n >= 0 && __it != __last; ++__it)
            ;
        return __it;
    }
};

template <typename _T, class _Enable = void>
class __future;

template <typename... _Bs>
struct __conjunction : ::std::true_type
{
};

template <typename _B1>
struct __conjunction<_B1> : _B1
{
};

template <typename _B1, typename... _Bs>
struct __conjunction<_B1, _Bs...> : ::std::conditional<!bool(_B1::value), _B1, __conjunction<_Bs...>>::type
{
};

// empty base class for type erasure
struct __lifetime_keeper_base
{
    virtual ~__lifetime_keeper_base() {}
};

// derived class to keep temporaries (e.g. buffer) alive
template <typename... Ts>
struct __lifetime_keeper : public __lifetime_keeper_base
{
    ::std::tuple<Ts...> __my_tmps;
    __lifetime_keeper(Ts... __t) : __my_tmps(::std::make_tuple(__t...)) {}
};

//-----------------------------------------------------------------------
// Generic bit- and number-manipulation routines
//-----------------------------------------------------------------------

// Bitwise type casting, same as C++20 std::bit_cast
template <typename _Dst, typename _Src>
::std::enable_if_t<
    sizeof(_Dst) == sizeof(_Src) && ::std::is_trivially_copyable_v<_Dst> && ::std::is_trivially_copyable_v<_Src>, _Dst>
__dpl_bit_cast(const _Src& __src) noexcept
{
#if __cpp_lib_bit_cast >= 201806L
    return ::std::bit_cast<_Dst>(__src);
#elif _ONEDPL_BACKEND_SYCL && _ONEDPL_LIBSYCL_VERSION >= 50300
    return sycl::bit_cast<_Dst>(__src);
#elif __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(_Dst, __src);
#else
    _Dst __result;
    ::std::memcpy(&__result, &__src, sizeof(_Dst));
    return __result;
#endif
}

// The max power of 2 not exceeding the given value, same as C++20 std::bit_floor
template <typename _T>
::std::enable_if_t<::std::is_integral_v<_T> && ::std::is_unsigned_v<_T>, _T>
__dpl_bit_floor(_T __x) noexcept
{
    if (__x == 0)
        return 0;
#if __cpp_lib_int_pow2 >= 202002L
    return ::std::bit_floor(__x);
#elif _ONEDPL_BACKEND_SYCL
    // Use the count-leading-zeros function
    return _T{1} << (sizeof(_T) * CHAR_BIT - sycl::clz(__x) - 1);
#else
    // Fill all the lower bits with 1s
    __x |= (__x >> 1);
    __x |= (__x >> 2);
    __x |= (__x >> 4);
    if constexpr (sizeof(_T) > 1) __x |= (__x >> 8);
    if constexpr (sizeof(_T) > 2) __x |= (__x >> 16);
    if constexpr (sizeof(_T) > 4) __x |= (__x >> 32);
    __x += 1; // Now it equals to the next greater power of 2, or 0 in case of wraparound
    return (__x == 0) ? _T{1} << (sizeof(_T) * CHAR_BIT - 1) : __x >> 1;
#endif
}

// The max power of 2 not smaller than the given value, same as C++20 std::bit_ceil
template <typename _T>
::std::enable_if_t<::std::is_integral_v<_T> && ::std::is_unsigned_v<_T>, _T>
__dpl_bit_ceil(_T __x) noexcept
{
    return ((__x & (__x - 1)) != 0) ? __dpl_bit_floor(__x) << 1 : __x;
}

// rounded up result of (__number / __divisor)
template <typename _T1, typename _T2>
constexpr auto
__dpl_ceiling_div(_T1 __number, _T2 __divisor)
{
    return (__number - 1) / __divisor + 1;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UTILS_H
