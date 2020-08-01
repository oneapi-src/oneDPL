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

#ifndef _PSTL_tuple_impl_H
#define _PSTL_tuple_impl_H

#include <iterator>
#include <tuple>
#include <cassert>

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

template <typename T1, typename... T, std::size_t... indices>
std::tuple<T...>
get_tuple_tail_impl(const std::tuple<T1, T...>& t, const oneapi::dpl::__internal::__index_sequence<indices...>&)
{
    return std::tuple<T...>(std::get<indices + 1>(t)...);
}

template <typename T1, typename... T>
std::tuple<T...>
get_tuple_tail(const std::tuple<T1, T...>& other)
{
    return get_tuple_tail_impl(other, oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>());
}

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

// Replacement for std::forward_as_tuple to avoid having tuple of rvalue references
template <class... Args>
auto
__forward_tuple(Args&&... args) -> decltype(
    std::tuple<typename oneapi::dpl::__internal::__lvref_or_val<Args>::__type...>(std::forward<Args>(args)...))
{
    return std::tuple<typename oneapi::dpl::__internal::__lvref_or_val<Args>::__type...>(std::forward<Args>(args)...);
}

struct make_std_tuple_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const -> decltype(__forward_tuple(std::forward<Args>(args)...))
    {
        // Use forward_as_tuple to correctly propagate references inside the tuple
        return __forward_tuple(std::forward<Args>(args)...);
    }
};

template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_stdtuple(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_std_tuple_functor{}, f,
                               oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...));

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
    return {std::forward<T>(t)...};
}

// __internal::tuple_element
template <std::size_t N, typename T>
struct tuple_element;

template <std::size_t N, typename T, typename... Rest>
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
    constexpr typename tuple_element<N, tuple<T...>>::type
    operator()(tuple<T...>& t) const
    {
        return get_impl<N - 1>()(t.next);
    }
    template <typename... T>
    constexpr typename tuple_element<N, tuple<T...>>::type const
    operator()(const tuple<T...>& t) const
    {
        return get_impl<N - 1>()(t.next);
    }
};

template <>
struct get_impl<0>
{
    template <typename... T>
    constexpr typename tuple_element<0, tuple<T...>>::type
    operator()(tuple<T...>& t) const
    {
        return t.value;
    }
    template <typename... T>
    constexpr typename tuple_element<0, tuple<T...>>::type const
    operator()(const tuple<T...>& t) const
    {
        return t.value;
    }
};

// __internal::get
// According to the standard, this should have overloads with type& and type&&
// But this produces some issues with determining the type of type's element
// because all of them becomes lvalue-references if the type is passed as lvalue.
// Removing & from the return type fixes the issue and doesn't seem to break
// anything else.
// TODO: investigate whether it's possible to keep the behavior and specify get
// according to the standard at the same time.
template <size_t N, typename... T>
constexpr typename tuple_element<N, tuple<T...>>::type
get(tuple<T...>& t)
{
    return get_impl<N>()(t);
}
template <size_t N, typename... T>
constexpr typename tuple_element<N, tuple<T...>>::type const
get(const tuple<T...>& t)
{
    return get_impl<N>()(t);
}

// __internal::map_tuple
template <size_t I, typename F, typename... T>
auto
apply_to_tuple(F f, T... in) -> decltype(f(get<I>(in)...))
{
    return f(get<I>(in)...);
}

struct make_inner_tuple_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const -> decltype(oneapi::dpl::__internal::make_tuple(std::forward<Args>(args)...))
    {
        return oneapi::dpl::__internal::make_tuple(std::forward<Args>(args)...);
    }
};

struct make_tuplewrapper_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const
        -> decltype(oneapi::dpl::__internal::make_tuplewrapper(std::forward<Args>(args)...))
    {
        return oneapi::dpl::__internal::make_tuplewrapper(std::forward<Args>(args)...);
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

// Required to repack any tuple to std::tuple to return to user
template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_stdtuple(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_std_tuple_functor{}, f,
                               oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...))
{
    return map_tuple_impl(make_std_tuple_functor{}, f, oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(),
                          in, rest...);
}

// Function can replace all above map_* functions,
// but requires from its user an additional functor
// that knows how to construct tuple of a certain type
template <typename MakeTupleF, typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_any_tuplelike_to(MakeTupleF mtf, F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(mtf, f, oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...))
{
    return map_tuple_impl(mtf, f, oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...);
}

template <typename T1, typename... T>
struct tuple<T1, T...>
{
    T1 value;
    tuple<T...> next;

    using tuple_type = std::tuple<T1, T...>;

    // since compiler does not autogenerate ctors
    // if user defines its own, we have to define them too
    tuple() = default;
    tuple(const tuple& other) = default;
    tuple(const T1& _value, const T&... _next) : value(_value), next(_next...) {}

    // required to convert std::tuple to inner tuple in user-provided functor
    tuple(const std::tuple<T1, T...>& other) : value(std::get<0>(other)), next(get_tuple_tail(other)) {}

    operator std::tuple<T1, T...>() const { return map_stdtuple(oneapi::dpl::__internal::__no_op{}, *this); }

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
        value = other.value;
        next = other.next;
        return *this;
    }

    // if T1 is deduced with reference, compiler generates deleted operator= and,
    // since "template operator=" is not considered as operator= overload
    // the deleted operator= has a preference during lookup
    tuple&
    operator=(const __internal::tuple<T1, T...>& other) = default;
};

// The only purpose of this specialization is to have explicitly
// defined operator= which otherwise(with = default) would be implicitly deleted.
// TODO: check if it's possible to remove duplication without complicated code.
template <typename _T1, typename... _T>
struct tuple<_T1&, _T&...>
{
    _T1& value;
    tuple<_T&...> next;

    using tuple_type = std::tuple<_T1&, _T&...>;

    // since compiler does not autogenerate ctors
    // if user defines its own, we have to define them too
    tuple() = default;
    tuple(const tuple& other) = default;
    tuple(_T1& _value, _T&... _next) : value(_value), next(_next...) {}

    // required to convert std::tuple to inner tuple in user-provided functor
    tuple(const std::tuple<_T1&, _T&...>& other) : value(std::get<0>(other)), next(get_tuple_tail(other)) {}

    operator std::tuple<_T1&, _T&...>() const { return map_stdtuple(oneapi::dpl::__internal::__no_op{}, *this); }

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
        value = other.value;
        next = other.next;
        return *this;
    }

    // if T1 is deduced with reference, compiler generates deleted operator= and,
    // since "template operator=" is not considered as operator= overload
    // the deleted operator= has a preference during lookup
    tuple&
    operator=(const __internal::tuple<_T1&, _T&...>& other)
    {
        value = other.value;
        next = other.next;
        return *this;
    }
};

template <>
struct tuple<>
{
    using tuple_type = std::tuple<>;
    // since compiler does not autogenerate ctors
    // if user defines its own, we have to define them too
    tuple() = default;
    tuple(const tuple& other) = default;

    tuple(const std::tuple<>&) {}

    tuple<> operator[](tuple<>) { return {}; }
    tuple<> operator[](const tuple<>&) const { return {}; }
    tuple<>&
    operator=(const tuple<>&) = default;
};

inline void
swap(tuple<>& __x, tuple<>& __y)
{
}

template <typename... _T>
void
swap(tuple<_T...>& __x, tuple<_T...>& __y)
{
    using std::swap;
    swap(__x.value, __y.value);
    swap(__x.next, __y.next);
}

// Get corresponding std::tuple for our internal tuple(i.e. access tuple_type member
// which is std::tuple<Ts...> for internal::tuple<Ts...>).
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

    template <typename... _Up>
    __tuplewrapper&
    operator=(const oneapi::dpl::__internal::tuple<_Up...>& __other)
    {
        __base_type::operator=(std::tuple<_Up...>(__other));
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
} // namespace dpl
} // namespace oneapi

#if _CPPLIB_VER
namespace std
{
template <size_t _Idx, typename... _Tp>
auto
get(const oneapi::dpl::__internal::__tuplewrapper<_Tp...>& __a)
    -> decltype(std::get<_Idx>(typename oneapi::dpl::__internal::__tuplewrapper<_Tp...>::__base_type(__a)))
{
    using __base = typename oneapi::dpl::__internal::__tuplewrapper<_Tp...>::__base_type;
    return std::get<_Idx>(__base(__a));
}
} // namespace std
#endif

#endif /* _PSTL_tuple_impl_H */
