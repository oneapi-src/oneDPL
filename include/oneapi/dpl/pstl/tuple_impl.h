// -*- C++ -*-
//===-- tuple_impl.h ---------------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#ifndef _ONEDPL_tuple_impl_H
#define _ONEDPL_tuple_impl_H

#include <iterator>
#include <tuple>
#include <cassert>
#include <type_traits>

#include "utils.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//custom tuple utilities

template <typename... T>
struct tuple;

template <typename... Size>
struct get_value_by_idx;

template <typename T1, typename... T, ::std::size_t... indices>
::std::tuple<T...>
get_tuple_tail_impl(const ::std::tuple<T1, T...>& t, const oneapi::dpl::__internal::__index_sequence<indices...>&)
{
    return ::std::tuple<T...>(::std::get<indices + 1>(t)...);
}

template <typename T1, typename... T>
::std::tuple<T...>
get_tuple_tail(const ::std::tuple<T1, T...>& other)
{
    return get_tuple_tail_impl(other, oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>());
}

// Maps an incoming type for tuplewrapper to simplify tuple-related handling.
// as it doesn't work well with rvalue refs.
// T& -> T&, T&& -> T, T -> T
template <typename _Tp, bool = ::std::is_lvalue_reference<_Tp>::value>
struct __lvref_or_val
{
    using __type = _Tp&&;
};

template <typename _Tp>
struct __lvref_or_val<_Tp, false>
{
    using __type = typename ::std::remove_reference<_Tp>::type;
};

template <typename T>
struct MapValue
{
    T id;
    template <typename T1>
    auto
    operator()(const T1& t1) -> decltype(t1[id]) const
    {
        return t1[id];
    }
};

// It serves the same purpose as tuplewrapper in iterator_impl.h, but in this case we don't need
// swap, so we can simply map it to tuple adjusting the types
template <typename... T>
using tuplewrapper = oneapi::dpl::__internal::tuple<typename oneapi::dpl::__internal::__lvref_or_val<T>::__type...>;

// __internal::make_tuple
template <typename... T>
constexpr tuple<T...>
make_tuple(T... args)
{
    return oneapi::dpl::__internal::tuple<T...>{args...};
}

// __internal::make_tuplewrapper
template <typename... T>
__internal::tuplewrapper<T&&...>
make_tuplewrapper(T&&... t)
{
    return {::std::forward<T>(t)...};
}

// __internal::tuple_element
template <::std::size_t N, typename T>
struct tuple_element;

template <::std::size_t N, typename T, typename... Rest>
struct tuple_element<N, tuple<T, Rest...>> : tuple_element<N - 1, tuple<Rest...>>
{
};

template <typename T, typename... Rest>
struct tuple_element<0, tuple<T, Rest...>>
{
    using type = T;
};

template <size_t N>
struct get_impl
{
    template <typename... T>
    constexpr auto
    operator()(tuple<T...>& t) const -> decltype(get_impl<N - 1>()(t.next))
    {
        return get_impl<N - 1>()(t.next);
    }

    template <typename... T>
    constexpr auto
    operator()(const tuple<T...>& t) const -> decltype(get_impl<N - 1>()(t.next))
    {
        return get_impl<N - 1>()(t.next);
    }
};

template <>
struct get_impl<0>
{
    template <typename... T>
    using ret_type = typename tuple_element<0, tuple<T...>>::type;

    template <typename... T>
    constexpr ret_type<T...>&
    operator()(tuple<T...>& t) const
    {
        return t.holder.value;
    }

    template <typename... T>
    constexpr ret_type<T...> const&
    operator()(const tuple<T...>& t) const
    {
        return t.holder.value;
    }
};

// __internal::map_tuple
template <size_t I, typename F, typename... T>
auto
apply_to_tuple(F f, T... in) -> decltype(f(get_impl<I>()(in)...))
{
    return f(get_impl<I>()(in)...);
}

struct make_inner_tuple_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const -> decltype(oneapi::dpl::__internal::make_tuple(::std::forward<Args>(args)...))
    {
        return oneapi::dpl::__internal::make_tuple(::std::forward<Args>(args)...);
    }
};

struct make_tuplewrapper_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const
        -> decltype(oneapi::dpl::__internal::make_tuplewrapper(::std::forward<Args>(args)...))
    {
        return oneapi::dpl::__internal::make_tuplewrapper(::std::forward<Args>(args)...);
    }
};

template <typename MakeTupleF, typename F, size_t... indices, typename... T>
auto
map_tuple_impl(MakeTupleF mtf, F f, oneapi::dpl::__internal::__index_sequence<indices...>, T... in)
    -> decltype(mtf(apply_to_tuple<indices>(f, in...)...))
{
    return mtf(apply_to_tuple<indices>(f, in...)...);
}

//
template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_tuple(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_inner_tuple_functor{}, f,
                               oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...))
{
    return map_tuple_impl(make_inner_tuple_functor{}, f, oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(),
                          in, rest...);
}

// Functions are needed to call get_value_by_idx: it requires to store in tuple wrapper
template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_tuplewrapper(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_tuplewrapper_functor{}, f,
                               oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...))
{
    return map_tuple_impl(make_tuplewrapper_functor{}, f,
                          oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...);
}

template <typename _Tp>
struct __value_holder
{
    __value_holder() = default;
    __value_holder(const _Tp& t) : value(t) {}
    _Tp value;
};

// Neccessary to make tuple trivially_copy_assignable. This type decided
// if it's needed to have user-defined operator=.
template <typename _Tp, bool = oneapi::dpl::__internal::__has_trivial_copy_assignemnt<__value_holder<_Tp>>::value>
struct __copy_assignable_holder : __value_holder<_Tp>
{
    using __value_holder<_Tp>::__value_holder;
};

template <typename _Tp>
struct __copy_assignable_holder<_Tp, false> : __value_holder<_Tp>
{
    using __value_holder<_Tp>::__value_holder;

    __copy_assignable_holder&
    operator=(const __copy_assignable_holder& other)
    {
        this->value = other.value;
        return *this;
    }
};

template <typename T1, typename... T>
struct tuple<T1, T...>
{
    __copy_assignable_holder<T1> holder;
    tuple<T...> next;

    using tuple_type = ::std::tuple<T1, T...>;

    tuple() = default;
    tuple(const tuple& other) = default;
    template <typename U1, typename... U>
    tuple(const tuple<U1, U...>& other) : holder(other.holder.value), next(other.next)
    {
    }

    tuple(const T1& _value, const T&... _next) : holder(_value), next(_next...) {}

    // required to convert ::std::tuple to inner tuple in user-provided functor
    tuple(const ::std::tuple<T1, T...>& other) : holder(::std::get<0>(other)), next(get_tuple_tail(other)) {}

    // conversion to ::std::tuple with the same template arguments
    operator ::std::tuple<T1, T...>() const
    {
        static constexpr ::std::size_t __tuple_size = sizeof...(T) + 1;
        return to_std_tuple(*this, oneapi::dpl::__internal::__make_index_sequence<__tuple_size>());
    }

    // conversion to ::std::tuple with the different template arguments
    template <typename U1, typename... U>
    operator ::std::tuple<U1, U...>() const
    {
        static constexpr ::std::size_t __tuple_size = sizeof...(T) + 1;
        return to_std_tuple(static_cast<tuple<U1, U...>>(*this),
                            oneapi::dpl::__internal::__make_index_sequence<__tuple_size>());
    }

    // non-const subscript operator with tuple argument
    template <typename Size1, typename... SizeRest>
    auto operator[](tuple<Size1, SizeRest...> tuple_size)
        -> decltype(__internal::get_value_by_idx<Size1, SizeRest...>()(*this, tuple_size))
    {
        return __internal::get_value_by_idx<Size1, SizeRest...>()(*this, tuple_size);
    }

    // const subscript operator with tuple argument
    template <typename Size1, typename... SizeRest>
    auto operator[](const tuple<Size1, SizeRest...> tuple_size) const
        -> decltype(__internal::get_value_by_idx<Size1, SizeRest...>()(*this, tuple_size))
    {
        return __internal::get_value_by_idx<Size1, SizeRest...>()(*this, tuple_size);
    }

    // non-const subscript operator with scalar argument
    template <typename Idx>
    auto operator[](Idx idx) -> decltype(map_tuplewrapper(MapValue<Idx>{idx}, *this))
    {
        return map_tuplewrapper(MapValue<Idx>{idx}, *this);
    }

    // const subscript operator with scalar argument
    template <typename Idx>
    auto operator[](Idx idx) const -> decltype(map_tuplewrapper(MapValue<Idx>{idx}, *this))
    {
        return map_tuplewrapper(MapValue<Idx>{idx}, *this);
    }

    template <typename U1, typename... U>
    tuple&
    operator=(const __internal::tuple<U1, U...>& other)
    {
        holder.value = other.holder.value;
        next = other.next;
        return *this;
    }

    // if T1 is deduced with reference, compiler generates deleted operator= and,
    // since "template operator=" is not considered as operator= overload
    // the deleted operator= has a preference during lookup
    tuple&
    operator=(const __internal::tuple<T1, T...>& other) = default;

    // for cases when we assign ::std::tuple to __internal::tuple
    template <typename U1, typename... U>
    tuple&
    operator=(const ::std::tuple<U1, U...>& other)
    {
        holder.value = ::std::get<0>(other);
        next = get_tuple_tail(other);
        return *this;
    }

    friend bool
    operator==(const tuple& __lhs, const tuple& __rhs)
    {
        return __lhs.holder.value == __rhs.holder.value && __lhs.next == __rhs.next;
    }
    friend bool
    operator!=(const tuple& __lhs, const tuple& __rhs)
    {
        return !(__lhs == __rhs);
    }

    template <typename U1, typename... U, ::std::size_t... _Ip>
    static ::std::tuple<U1, U...>
    to_std_tuple(const tuple<U1, U...>& __t, oneapi::dpl::__internal::__index_sequence<_Ip...>)
    {
        return ::std::tuple<U1, U...>(get_impl<_Ip>()(__t)...);
    }
};

template <>
struct tuple<>
{
    using tuple_type = ::std::tuple<>;
    // since compiler does not autogenerate ctors
    // if user defines its own, we have to define them too
    tuple() = default;
    tuple(const tuple& other) = default;

    tuple(const ::std::tuple<>&) {}

    tuple operator[](tuple) { return {}; }
    tuple operator[](const tuple&) const { return {}; }
    tuple&
    operator=(const tuple&) = default;
    tuple&
    operator=(const ::std::tuple<>& other)
    {
        return *this;
    }
    friend bool
    operator==(const tuple& __lhs, const tuple& __rhs)
    {
        return true;
    }
};

inline void
swap(tuple<>& __x, tuple<>& __y)
{
}

template <typename... _T>
void
swap(tuple<_T...>& __x, tuple<_T...>& __y)
{
    using ::std::swap;
    swap(__x.holder.value, __y.holder.value);
    swap(__x.next, __y.next);
}

template <typename... _T>
void
swap(tuple<_T...>&& __x, tuple<_T...>&& __y)
{
    using ::std::swap;
    swap(__x.holder.value, __y.holder.value);
    swap(__x.next, __y.next);
}

// Get corresponding ::std::tuple for our internal tuple(i.e. access tuple_type member
// which is ::std::tuple<Ts...> for internal::tuple<Ts...>).
// Do nothing for other types or if both operands are internal tuples.
template <class _T, class>
struct __get_tuple_type
{
    using __type = _T;
};

template <class... _Ts, class... _Us>
struct __get_tuple_type<oneapi::dpl::__internal::tuple<_Ts...>, oneapi::dpl::__internal::tuple<_Us...>>
{
    using __type = typename oneapi::dpl::__internal::tuple<_Ts...>;
};

template <class... _Ts, class _Other>
struct __get_tuple_type<oneapi::dpl::__internal::tuple<_Ts...>, _Other>
{
    using __type = typename oneapi::dpl::__internal::tuple<_Ts...>::tuple_type;
};

template <typename Size>
struct AddIndexes
{
    Size idx;

    template <typename T1>
    T1
    operator()(const T1& t1) const
    {
        return t1 + idx;
    }
};

template <typename Size>
struct SubTupleFromIndex
{
    Size idx;

    template <typename T1>
    T1
    operator()(const T1& t1) const
    {
        return idx - t1;
    }
};

template <typename Size>
struct SubIndexFromTuple
{
    Size idx;

    template <typename T1>
    T1
    operator()(const T1& t1) const
    {
        return t1 - idx;
    }
};

struct subscription_functor
{
    template <typename _A, typename _Index>
    auto
    operator()(_A __a, _Index __idx) -> decltype(__a[__idx]) const
    {
        return __a[__idx];
    }
};

template <typename... Size>
struct get_value_by_idx
{
    template <typename... Acc>
    auto
    operator()(oneapi::dpl::__internal::tuple<Acc...>& acc, oneapi::dpl::__internal::tuple<Size...>& idx)
        -> decltype(oneapi::dpl::__internal::map_tuplewrapper(subscription_functor{}, acc, idx))
    {
        return oneapi::dpl::__internal::map_tuplewrapper(subscription_functor{}, acc, idx);
    }
    template <typename... Acc>
    auto
    operator()(const oneapi::dpl::__internal::tuple<Acc...>& acc,
               const oneapi::dpl::__internal::tuple<Size...>& idx) const
        -> decltype(oneapi::dpl::__internal::map_tuplewrapper(subscription_functor{}, acc, idx))
    {
        return oneapi::dpl::__internal::map_tuplewrapper(subscription_functor{}, acc, idx);
    }
};

template <typename Size, typename... T1>
oneapi::dpl::__internal::tuple<T1...>
operator+(const oneapi::dpl::__internal::tuple<T1...>& tuple1, Size idx)
{
    return oneapi::dpl::__internal::map_tuple(AddIndexes<Size>{idx}, tuple1);
}

template <typename Size, typename... T1>
oneapi::dpl::__internal::tuple<T1...>
operator+(Size idx, const oneapi::dpl::__internal::tuple<T1...>& tuple1)
{
    return oneapi::dpl::__internal::map_tuple(AddIndexes<Size>{idx}, tuple1);
}

template <typename Size, typename... T1>
oneapi::dpl::__internal::tuple<T1...>
operator-(const oneapi::dpl::__internal::tuple<T1...>& tuple1, Size idx)
{
    return oneapi::dpl::__internal::map_tuple(SubIndexFromTuple<Size>{idx}, tuple1);
}

// required in scan implementation for false offset calculation
template <typename Size, typename... T1>
auto
operator-(Size idx, const oneapi::dpl::__internal::tuple<T1...>& tuple1)
    -> decltype(oneapi::dpl::__internal::map_tuple(SubTupleFromIndex<Size>{idx}, tuple1))
{
    return oneapi::dpl::__internal::map_tuple(SubTupleFromIndex<Size>{idx}, tuple1);
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

namespace std
{
template <size_t _Idx, typename... _Tp>
constexpr typename tuple_element<_Idx, tuple<_Tp...>>::type&
get(oneapi::dpl::__internal::tuple<_Tp...>& __a)
{
    return oneapi::dpl::__internal::get_impl<_Idx>()(__a);
}

template <size_t _Idx, typename... _Tp>
constexpr typename tuple_element<_Idx, tuple<_Tp...>>::type const&
get(const oneapi::dpl::__internal::tuple<_Tp...>& __a)
{
    return oneapi::dpl::__internal::get_impl<_Idx>()(__a);
}

} // namespace std

#endif /* _ONEDPL_tuple_impl_H */
