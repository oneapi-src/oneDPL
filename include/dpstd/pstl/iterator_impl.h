// -*- C++ -*-
//===-- iterator_impl.h ---------------------------------------------------===//
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

#ifndef _PSTL_iterator_impl_H
#define _PSTL_iterator_impl_H

#include <iterator>
#include <tuple>
#include <cassert>

#include "utils.h"

namespace dpstd
{
namespace __internal
{
template <size_t _Np>
struct __tuple_util
{
    template <typename _TupleType, typename _DifferenceType>
    static void
    __increment(_TupleType& __it, _DifferenceType __forward)
    {
        std::get<_Np - 1>(__it) = std::get<_Np - 1>(__it) + __forward;
        __tuple_util<_Np - 1>::__increment(__it, __forward);
    }
    template <typename _TupleType>
    static void
    __pre_increment(_TupleType& __it)
    {
        ++std::get<_Np - 1>(__it);
        __tuple_util<_Np - 1>::__pre_increment(__it);
    }
};

template <>
struct __tuple_util<0>
{
    template <typename _TupleType, typename _DifferenceType>
    static void
    __increment(_TupleType&, _DifferenceType)
    {
    }
    template <typename _TupleType>
    static void
    __pre_increment(_TupleType&)
    {
    }
};

template <typename _TupleReturnType>
struct __make_references
{
    template <typename _TupleType, std::size_t... _Ip>
    _TupleReturnType
    operator()(const _TupleType& __t, dpstd::__internal::__index_sequence<_Ip...>)
    {
        return _TupleReturnType(*std::get<_Ip>(__t)...);
    }
};

//zip_iterator version for forward iterator
//== and != comparison is performed only on the first element of the tuple
template <typename... _Types>
class zip_forward_iterator
{
    static const std::size_t __num_types = sizeof...(_Types);
    typedef typename std::tuple<_Types...> __it_types;

  public:
    typedef typename std::make_signed<std::size_t>::type difference_type;
    typedef std::tuple<typename std::iterator_traits<_Types>::value_type...> value_type;
    typedef std::tuple<typename std::iterator_traits<_Types>::reference...> reference;
    typedef std::tuple<typename std::iterator_traits<_Types>::pointer...> pointer;
    typedef std::forward_iterator_tag iterator_category;

    zip_forward_iterator() : __my_it_() {}
    explicit zip_forward_iterator(_Types... __args) : __my_it_(std::make_tuple(__args...)) {}
    zip_forward_iterator(const zip_forward_iterator& __input) : __my_it_(__input.__my_it_) {}
    zip_forward_iterator&
    operator=(const zip_forward_iterator& __input)
    {
        __my_it_ = __input.__my_it_;
        return *this;
    }

    reference operator*() const
    {
        return __make_references<reference>()(__my_it_, __make_index_sequence<__num_types>());
    }

    zip_forward_iterator&
    operator++()
    {
        __tuple_util<__num_types>::__pre_increment(__my_it_);
        return *this;
    }
    zip_forward_iterator
    operator++(int)
    {
        zip_forward_iterator __it(*this);
        ++(*this);
        return __it;
    }

    bool
    operator==(const zip_forward_iterator& __it) const
    {
        return std::get<0>(__my_it_) == std::get<0>(__it.__my_it_);
    }
    bool
    operator!=(const zip_forward_iterator& __it) const
    {
        return !(*this == __it);
    }

    __it_types
    base() const
    {
        return __my_it_;
    }

  private:
    __it_types __my_it_;
};

// Maps an incoming type for tuplewrapper to simplify tuple-related handling.
// as it doesn't work well with rvalue refs.
// T& -> T&, T&& -> T, T -> T
template <typename _Tp, bool = std::is_lvalue_reference<_Tp>::value>
struct __lvref_or_val
{
    using __type = _Tp&&;
};

template <typename _Tp>
struct __lvref_or_val<_Tp, false>
{
    using __type = typename std::remove_reference<_Tp>::type;
};

// A simple wrapper over a tuple of references.
// The class is designed to hold a temporary tuple of reference
// after dereferencing a zip_iterator; in particular, it is needed
// to swap these rvalue tuples.
// Note: there is a special handling for iterators which don't return a reference
// (like counting_iterator). tuplewrapper still valid for such iterators, but
// swap is not supported for them.
template <typename... _Tp>
struct __tuplewrapper : public std::tuple<typename __lvref_or_val<_Tp>::__type...>
{
    // In the context of this class, T is a reference, so T&& is a "forwarding reference"
    typedef std::tuple<typename __lvref_or_val<_Tp>::__type...> __base_type;
    using __base_type::__base_type;

    // Construct from the result of std::tie
    __tuplewrapper(const __base_type& __t) : __base_type(__t) {}
#if __INTEL_COMPILER
    // ICC cannot generate copy ctor & assignment
    __tuplewrapper(const __tuplewrapper& __rhs) : __base_type(__rhs) {}
    __tuplewrapper&
    operator=(const __tuplewrapper& __rhs)
    {
        *this = __base_type(__rhs);
        return *this;
    }
#endif
    // Assign any tuple convertible to std::tuple<T&&...>: *it = a_tuple;
    template <typename... _Up>
    __tuplewrapper&
    operator=(const std::tuple<_Up...>& __other)
    {
        __base_type::operator=(__other);
        return *this;
    }
#if _LIBCPP_VERSION || _CPPLIB_VER
    // (Necessary for libc++ tuples) Convert to a tuple of values: v = *it;
    operator std::tuple<typename std::remove_reference<_Tp>::type...>() { return __base_type(*this); }
#endif
#if _CPPLIB_VER
    friend bool
    operator==(const __tuplewrapper& __lhs, const __tuplewrapper& __rhs)
    {
        return __base_type(__lhs) == __base_type(__rhs);
    }
    friend bool
    operator!=(const __tuplewrapper& __lhs, const __tuplewrapper& __rhs)
    {
        return __base_type(__lhs) != __base_type(__rhs);
    }
#endif
    // Swap rvalue tuples: swap(*it1,*it2);
    friend void
    swap(__tuplewrapper&& __a, __tuplewrapper&& __b)
    {
        __a.swap(__b);
    }
};
} // namespace __internal
} // namespace dpstd

#if _CPPLIB_VER
namespace std
{
template <size_t _Idx, typename... _Tp>
auto
get(const dpstd::__internal::__tuplewrapper<_Tp...>& __a)
    -> decltype(std::get<_Idx>(typename dpstd::__internal::__tuplewrapper<_Tp...>::__base_type(__a)))
{
    using __base = typename dpstd::__internal::__tuplewrapper<_Tp...>::__base_type;
    return std::get<_Idx>(__base(__a));
}
} // namespace std
#endif

namespace dpstd
{
template <typename _Ip>
class counting_iterator
{
    static_assert(std::is_integral<_Ip>::value, "Cannot instantiate counting_iterator with a non-integer type");

  public:
    typedef typename std::make_signed<_Ip>::type difference_type;
    typedef _Ip value_type;
    typedef const _Ip* pointer;
    // There is no storage behind the iterator, so we return a value instead of reference.
    typedef _Ip reference;
    typedef std::random_access_iterator_tag iterator_category;

    counting_iterator() : __my_counter_() {}
    explicit counting_iterator(_Ip __init) : __my_counter_(__init) {}

    reference operator*() const { return __my_counter_; }
    reference operator[](difference_type __i) const { return *(*this + __i); }

    difference_type
    operator-(const counting_iterator& __it) const
    {
        return __my_counter_ - __it.__my_counter_;
    }

    counting_iterator&
    operator+=(difference_type __forward)
    {
        __my_counter_ += __forward;
        return *this;
    }
    counting_iterator&
    operator-=(difference_type __backward)
    {
        return *this += -__backward;
    }
    counting_iterator&
    operator++()
    {
        return *this += 1;
    }
    counting_iterator&
    operator--()
    {
        return *this -= 1;
    }

    counting_iterator
    operator++(int)
    {
        counting_iterator __it(*this);
        ++(*this);
        return __it;
    }
    counting_iterator
    operator--(int)
    {
        counting_iterator __it(*this);
        --(*this);
        return __it;
    }

    counting_iterator
    operator-(difference_type __backward) const
    {
        return counting_iterator(__my_counter_ - __backward);
    }
    counting_iterator
    operator+(difference_type __forward) const
    {
        return counting_iterator(__my_counter_ + __forward);
    }
    friend counting_iterator
    operator+(difference_type __forward, const counting_iterator __it)
    {
        return __it + __forward;
    }

    bool
    operator==(const counting_iterator& __it) const
    {
        return *this - __it == 0;
    }
    bool
    operator!=(const counting_iterator& __it) const
    {
        return !(*this == __it);
    }
    bool
    operator<(const counting_iterator& __it) const
    {
        return *this - __it < 0;
    }
    bool
    operator>(const counting_iterator& __it) const
    {
        return __it < *this;
    }
    bool
    operator<=(const counting_iterator& __it) const
    {
        return !(*this > __it);
    }
    bool
    operator>=(const counting_iterator& __it) const
    {
        return !(*this < __it);
    }

  private:
    _Ip __my_counter_;
};

template <typename... _Types>
class zip_iterator
{
    static_assert(sizeof...(_Types) > 0, "Cannot instantiate zip_iterator with empty template parameter pack");
    static const std::size_t __num_types = sizeof...(_Types);
    typedef std::tuple<_Types...> __it_types;

  public:
    typedef typename std::make_signed<std::size_t>::type difference_type;
    typedef std::tuple<typename std::iterator_traits<_Types>::value_type...> value_type;
#if __INTEL_COMPILER && __INTEL_COMPILER < 1800 && _MSC_VER
    typedef std::tuple<typename std::iterator_traits<_Types>::reference...> reference;
#else
    typedef dpstd::__internal::__tuplewrapper<typename std::iterator_traits<_Types>::reference...> reference;
#endif
    typedef std::tuple<typename std::iterator_traits<_Types>::pointer...> pointer;
    typedef std::random_access_iterator_tag iterator_category;

    zip_iterator() : __my_it_() {}
    explicit zip_iterator(_Types... __args) : __my_it_(std::make_tuple(__args...)) {}
    zip_iterator(const zip_iterator& __input) : __my_it_(__input.__my_it_) {}
    zip_iterator&
    operator=(const zip_iterator& __input)
    {
        __my_it_ = __input.__my_it_;
        return *this;
    }

    reference operator*() const
    {
        return dpstd::__internal::__make_references<reference>()(
            __my_it_, dpstd::__internal::__make_index_sequence<__num_types>());
    }
    reference operator[](difference_type __i) const { return *(*this + __i); }

    difference_type
    operator-(const zip_iterator& __it) const
    {
        return std::get<0>(__my_it_) - std::get<0>(__it.__my_it_);
    }

    zip_iterator&
    operator+=(difference_type __forward)
    {
        dpstd::__internal::__tuple_util<__num_types>::__increment(__my_it_, __forward);
        return *this;
    }
    zip_iterator&
    operator-=(difference_type __backward)
    {
        return *this += -__backward;
    }
    zip_iterator&
    operator++()
    {
        return *this += 1;
    }
    zip_iterator&
    operator--()
    {
        return *this -= 1;
    }

    zip_iterator
    operator++(int)
    {
        zip_iterator __it(*this);
        ++(*this);
        return __it;
    }
    zip_iterator
    operator--(int)
    {
        zip_iterator __it(*this);
        --(*this);
        return __it;
    }

    zip_iterator
    operator-(difference_type __backward) const
    {
        zip_iterator __it(*this);
        return __it -= __backward;
    }
    zip_iterator
    operator+(difference_type __forward) const
    {
        zip_iterator __it(*this);
        return __it += __forward;
    }
    friend zip_iterator
    operator+(difference_type __forward, const zip_iterator& __it)
    {
        return __it + __forward;
    }

    bool
    operator==(const zip_iterator& __it) const
    {
        return *this - __it == 0;
    }
    __it_types
    base() const
    {
        return __my_it_;
    }

    bool
    operator!=(const zip_iterator& __it) const
    {
        return !(*this == __it);
    }
    bool
    operator<(const zip_iterator& __it) const
    {
        return *this - __it < 0;
    }
    bool
    operator>(const zip_iterator& __it) const
    {
        return __it < *this;
    }
    bool
    operator<=(const zip_iterator& __it) const
    {
        return !(*this > __it);
    }
    bool
    operator>=(const zip_iterator& __it) const
    {
        return !(*this < __it);
    }

  private:
    __it_types __my_it_;
};

template <typename... _Tp>
zip_iterator<_Tp...>
make_zip_iterator(_Tp... __args)
{
    return zip_iterator<_Tp...>(__args...);
}

template <typename _UnaryFunc, typename _Iter>
class transform_iterator
{
  public:
    typedef typename std::iterator_traits<_Iter>::difference_type difference_type;
#if _PSTL_CPP17_INVOKE_RESULT_PRESENT
    typedef typename std::invoke_result<_UnaryFunc, typename std::iterator_traits<_Iter>::reference>::type reference;
#else
    typedef typename std::result_of<_UnaryFunc(typename std::iterator_traits<_Iter>::reference)>::type reference;
#endif
    typedef typename std::remove_reference<reference>::type value_type;
    typedef typename std::iterator_traits<_Iter>::pointer pointer;
    typedef typename std::random_access_iterator_tag iterator_category;

    transform_iterator(_Iter __it, _UnaryFunc __unary_func) : __my_it_(__it), __my_unary_func_(__unary_func)
    {
        static_assert((std::is_same<typename std::iterator_traits<_Iter>::iterator_category,
                                    std::random_access_iterator_tag>::value),
                      "Random access iterator required.");
    }
    transform_iterator(const transform_iterator& __input)
        : __my_it_(__input.__my_it_), __my_unary_func_(__input.__my_unary_func_)
    {
    }
    transform_iterator&
    operator=(const transform_iterator& __input)
    {
        __my_it_ = __input.__my_it_;
        return *this;
    }
    reference operator*() const { return __my_unary_func_(*__my_it_); }
    reference operator[](difference_type __i) const { return *(*this + __i); }
    transform_iterator&
    operator++()
    {
        ++__my_it_;
        return *this;
    }
    transform_iterator&
    operator--()
    {
        --__my_it_;
        return *this;
    }
    transform_iterator
    operator++(int)
    {
        transform_iterator __it(*this);
        ++(*this);
        return __it;
    }
    transform_iterator
    operator--(int)
    {
        transform_iterator __it(*this);
        --(*this);
        return __it;
    }
    transform_iterator
    operator+(difference_type __forward) const
    {
        return {__my_it_ + __forward, __my_unary_func_};
    }
    transform_iterator
    operator-(difference_type __backward) const
    {
        return {__my_it_ - __backward, __my_unary_func_};
    }
    transform_iterator&
    operator+=(difference_type __forward)
    {
        __my_it_ += __forward;
        return *this;
    }
    transform_iterator&
    operator-=(difference_type __backward)
    {
        __my_it_ -= __backward;
        return *this;
    }
    friend transform_iterator
    operator+(difference_type __forward, const transform_iterator& __it)
    {
        return __it + __forward;
    }
    difference_type
    operator-(const transform_iterator& __it) const
    {
        return __my_it_ - __it.__my_it_;
    }
    bool
    operator==(const transform_iterator& __it) const
    {
        return *this - __it == 0;
    }
    bool
    operator!=(const transform_iterator& __it) const
    {
        return !(*this == __it);
    }
    bool
    operator<(const transform_iterator& __it) const
    {
        return *this - __it < 0;
    }
    bool
    operator>(const transform_iterator& __it) const
    {
        return __it < *this;
    }
    bool
    operator<=(const transform_iterator& __it) const
    {
        return !(*this > __it);
    }
    bool
    operator>=(const transform_iterator& __it) const
    {
        return !(*this < __it);
    }

    _Iter
    base() const
    {
        return __my_it_;
    }
    _UnaryFunc
    functor() const
    {
        return __my_unary_func_;
    }

  private:
    _Iter __my_it_;
    const _UnaryFunc __my_unary_func_;
};

template <typename _UnaryFunc, typename _Iter>
transform_iterator<_UnaryFunc, _Iter>
make_transform_iterator(_Iter __it, _UnaryFunc __unary_func)
{
    return transform_iterator<_UnaryFunc, _Iter>(__it, __unary_func);
}

} // namespace dpstd

#endif /* _PSTL_iterator_impl_H */
