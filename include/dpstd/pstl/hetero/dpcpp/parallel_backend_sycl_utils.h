// -*- C++ -*-
//===-- parallel_backend_sycl_utils.h -------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#ifndef _PSTL_parallel_backend_sycl_utils_H
#define _PSTL_parallel_backend_sycl_utils_H

//!!! NOTE: This file should be included under the macro _PSTL_BACKEND_SYCL
#include <CL/sycl.hpp>
#include <type_traits>
#include "../../iterator_impl.h"

#include "../../utils.h"

#define _PRINT_INFO_IN_DEBUG_MODE(...) dpstd::__par_backend_hetero::__internal::__print_device_debug_info(__VA_ARGS__)

namespace dpstd
{
namespace __par_backend_hetero
{

//-----------------------------------------------------------------------
// sycl::access::mode and sycl::access::target helpers
//-----------------------------------------------------------------------

// aliases for faster access to modes
using access_target = cl::sycl::access::target;
using access_mode = cl::sycl::access::mode;
static constexpr access_mode read = access_mode::read;
static constexpr access_mode write = access_mode::write;
static constexpr access_mode read_write = access_mode::read_write;
static constexpr access_mode discard_write = access_mode::discard_write;
static constexpr access_mode discard_read_write = access_mode::discard_read_write;

template <typename _T>
using __decay_t = typename std::decay<_T>::type;
template <bool __flag, typename _T = void>
using __enable_if_t = typename std::enable_if<__flag, _T>::type;

// function to hide zip_iterator creation
template <typename... T>
dpstd::zip_iterator<T...>
zip(T... args)
{
    return dpstd::zip_iterator<T...>(args...);
}

template <bool flag>
struct explicit_wait_if
{
    template <typename _ExecutionPolicy>
    void
    operator()(_ExecutionPolicy&&){};

    void operator()(cl::sycl::event){};
};

template <>
struct explicit_wait_if<true>
{
    template <typename _ExecutionPolicy>
    void
    operator()(_ExecutionPolicy&& __exec)
    {
        __exec.queue().wait_and_throw();
    };

    void
    operator()(cl::sycl::event __event)
    {
        __event.wait_and_throw();
    }
};

namespace __internal
{
namespace sycl = cl::sycl;

template <typename _Policy>
inline void __print_device_debug_info(_Policy, size_t = 0, size_t = 0)
{
}

template <typename... T>
struct tuple;

template <typename... Size>
struct get_value_by_idx;

template <typename T1, typename... T, std::size_t... indices>
std::tuple<T...>
get_tuple_tail_impl(const std::tuple<T1, T...>& t, const dpstd::__internal::__index_sequence<indices...>&)
{
    return std::tuple<T...>(std::get<indices + 1>(t)...);
}

template <typename T1, typename... T>
std::tuple<T...>
get_tuple_tail(const std::tuple<T1, T...>& other)
{
    return get_tuple_tail_impl(other, dpstd::__internal::__make_index_sequence<sizeof...(T)>());
}

// Replacement for std::forward_as_tuple to avoid having tuple of rvalue references
template <class... Args>
auto
__forward_tuple(Args&&... args)
    -> decltype(std::tuple<typename dpstd::__internal::__lvref_or_val<Args>::__type...>(std::forward<Args>(args)...))
{
    return std::tuple<typename dpstd::__internal::__lvref_or_val<Args>::__type...>(std::forward<Args>(args)...);
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
    -> decltype(map_tuple_impl(make_std_tuple_functor{}, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(),
                               in, rest...));

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

    operator std::tuple<T1, T...>() const { return map_stdtuple(dpstd::__internal::__no_op{}, *this); }

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

    operator std::tuple<_T1&, _T&...>() const { return map_stdtuple(dpstd::__internal::__no_op{}, *this); }

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

// It serves the same purpose as tuplewrapper in iterator_impl.h, but in this case we don't need
// swap, so we can simply map it to tuple adjusting the types
template <typename... T>
using tuplewrapper = __internal::tuple<typename dpstd::__internal::__lvref_or_val<T>::__type...>;

// __internal::make_tuple
template <typename... T>
constexpr tuple<T...>
make_tuple(T... args)
{
    return tuple<T...>{args...};
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

struct GetValue
{
    template <typename Acc, typename Size>
    auto
    operator()(Acc acc_i, Size idx_i) -> decltype(acc_i[idx_i]) const
    {
        return acc_i[idx_i];
    }
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
    operator()(Args&&... args) const -> decltype(__internal::make_tuple(std::forward<Args>(args)...))
    {
        return __internal::make_tuple(std::forward<Args>(args)...);
    }
};

struct make_tuplewrapper_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const -> decltype(__internal::make_tuplewrapper(std::forward<Args>(args)...))
    {
        return __internal::make_tuplewrapper(std::forward<Args>(args)...);
    }
};

struct make_zipiterator_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const -> decltype(dpstd::make_zip_iterator(std::forward<Args>(args)...))
    {
        return dpstd::make_zip_iterator(std::forward<Args>(args)...);
    }
};

template <typename MakeTupleF, typename F, size_t... indices, typename... T>
auto
map_tuple_impl(MakeTupleF mtf, F f, dpstd::__internal::__index_sequence<indices...>, T... in)
    -> decltype(mtf(apply_to_tuple<indices>(f, in...)...))
{
    return mtf(apply_to_tuple<indices>(f, in...)...);
}

//
template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_tuple(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_inner_tuple_functor{}, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(),
                               in, rest...))
{
    return map_tuple_impl(make_inner_tuple_functor{}, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(), in,
                          rest...);
}

// Functions are needed to call get_value_by_idx: it requires to store in tuple wrapper
template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_tuplewrapper(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_tuplewrapper_functor{}, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(),
                               in, rest...))
{
    return map_tuple_impl(make_tuplewrapper_functor{}, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(), in,
                          rest...);
}

// The functions are required because
// after applying a functor to each element of a tuple
// we may need to get a zip iterator

template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_zip(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_zipiterator_functor{}, dpstd::__internal::__make_index_sequence<sizeof...(T)>(), in,
                               rest...))
{
    return map_tuple_impl(make_zipiterator_functor{}, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(), in,
                          rest...);
}

// Required to repack any tuple to std::tuple to return to user
template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_stdtuple(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_std_tuple_functor{}, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(),
                               in, rest...))
{
    return map_tuple_impl(make_std_tuple_functor{}, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(), in,
                          rest...);
}

// Function can replace all above map_* functions,
// but requires from its user an additional functor
// that knows how to construct tuple of a certain type
template <typename MakeTupleF, typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_any_tuplelike_to(MakeTupleF mtf, F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(mtf, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...))
{
    return map_tuple_impl(mtf, f, dpstd::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...);
}

template <typename... Size>
struct get_value_by_idx
{
    template <typename... Acc>
    auto
    operator()(__internal::tuple<Acc...>& acc, __internal::tuple<Size...>& idx)
        -> decltype(map_tuplewrapper(GetValue(), acc, idx))
    {
        return map_tuplewrapper(GetValue(), acc, idx);
    }
    template <typename... Acc>
    auto
    operator()(const __internal::tuple<Acc...>& acc, const __internal::tuple<Size...>& idx) const
        -> decltype(map_tuplewrapper(GetValue(), acc, idx))
    {
        return map_tuplewrapper(GetValue(), acc, idx);
    }
};

template <typename Size, typename... T1>
__internal::tuple<T1...>
operator+(const __internal::tuple<T1...>& tuple1, Size idx)
{
    return map_tuple(AddIndexes<Size>{idx}, tuple1);
}

template <typename Size, typename... T1>
__internal::tuple<T1...>
operator+(Size idx, const __internal::tuple<T1...>& tuple1)
{
    return map_tuple(AddIndexes<Size>{idx}, tuple1);
}

template <typename Size, typename... T1>
__internal::tuple<T1...>
operator-(const __internal::tuple<T1...>& tuple1, Size idx)
{
    return map_tuple(SubIndexFromTuple<Size>{idx}, tuple1);
}

// required in scan implementation for false offset calculation
template <typename Size, typename... T1>
auto
operator-(Size idx, const __internal::tuple<T1...>& tuple1) -> decltype(map_tuple(SubTupleFromIndex<Size>{idx}, tuple1))
{
    return map_tuple(SubTupleFromIndex<Size>{idx}, tuple1);
}

// struct for checking if iterator is heterogeneous or not
template <typename Iter, typename Void = void> // for non-heterogeneous iterators
struct is_hetero_iterator : std::false_type
{
};

template <typename Iter> // for heterogeneous iterators
struct is_hetero_iterator<Iter, typename std::enable_if<Iter::is_hetero::value, void>::type> : std::true_type
{
};
// struct for checking if iterator should be passed directly to device or not
template <typename Iter, typename Void = void> // for iterators that should not be passed directly
struct is_passed_directly : std::false_type
{
};

template <typename Iter> // for iterators defined as direct pass
struct is_passed_directly<Iter, typename std::enable_if<Iter::is_passed_directly::value, void>::type> : std::true_type
{
};

template <typename Iter> // for pointers to objects on device
struct is_passed_directly<Iter, typename std::enable_if<std::is_pointer<Iter>::value, void>::type> : std::true_type
{
};

// functor to extract buffer from iterator or create temporary buffer to run on device

template <typename UnaryFunction, typename Iterator>
struct transform_buffer_wrapper;

// TODO: shifted_buffer can be removed when sub-buffers over sub-buffers will be supported.
template <typename T, typename StartIdx>
struct shifted_buffer
{
    using container_t = cl::sycl::buffer<T, 1>;
    using value_type = T;
    container_t buffer;
    StartIdx startIdx{};

    shifted_buffer(container_t&& buf) : buffer(std::move(buf)) {}

    shifted_buffer(const container_t& buf, StartIdx sId) : buffer(buf), startIdx(sId) {}
};

struct get_buffer
{
    // for heterogeneous iterator
    template <typename Iter>
    typename std::enable_if<is_hetero_iterator<Iter>::value,
                            shifted_buffer<typename std::iterator_traits<Iter>::value_type,
                                           typename std::iterator_traits<Iter>::difference_type>>::type
    operator()(Iter it, Iter)
    {
        return {it.get_buffer(), it - dpstd::begin(it.get_buffer())};
    }

    // for counting_iterator
    // To not multiply buffers without necessity it was decided to return counting_iterator
    // Counting_iterator already contains idx as dereferenced value. So idx should be 0
    template <typename T>
    dpstd::counting_iterator<T>
    operator()(dpstd::counting_iterator<T> it, dpstd::counting_iterator<T>)
    {
        return it;
    }
    // for zip_iterator
    template <typename... Iters>
    auto
    operator()(dpstd::zip_iterator<Iters...> it, dpstd::zip_iterator<Iters...> it2)
        -> decltype(map_tuple(*this, it.base(), it2.base()))
    {
        return map_tuple(*this, it.base(), it2.base());
    }

    // for transform_iterator
    template <typename UnaryFunction, typename Iterator>
    transform_buffer_wrapper<UnaryFunction, Iterator>
    operator()(dpstd::transform_iterator<UnaryFunction, Iterator> it1,
               dpstd::transform_iterator<UnaryFunction, Iterator> it2)
    {
        return {operator()(it1.base(), it2.base()), it1.functor()};
    }

    // the function is needed to create buffer over iterators depending on const or non-const iterators
    // for non-const iterators
    template <typename Iter>
    typename std::enable_if<!dpstd::__internal::is_const_iterator<Iter>::value,
                            cl::sycl::buffer<typename std::iterator_traits<Iter>::value_type, 1>>::type
    get_buffer_from_iters(Iter first, Iter last)
    {
        auto temp_buf = cl::sycl::buffer<typename std::iterator_traits<Iter>::value_type, 1>(first, last);
        temp_buf.set_final_data(first);
        return temp_buf;
    }

    // for const iterators
    template <typename Iter>
    typename std::enable_if<dpstd::__internal::is_const_iterator<Iter>::value,
                            cl::sycl::buffer<typename std::iterator_traits<Iter>::value_type, 1>>::type
    get_buffer_from_iters(Iter first, Iter last)
    {
        return sycl::buffer<typename std::iterator_traits<Iter>::value_type, 1>(first, last);
    }

    // for host iterator
    template <typename Iter>
    typename std::enable_if<!is_hetero_iterator<Iter>::value && !is_passed_directly<Iter>::value,
                            shifted_buffer<typename std::iterator_traits<Iter>::value_type,
                                           typename std::iterator_traits<Iter>::difference_type>>::type
    operator()(Iter first, Iter last)
    {
        using T = typename std::iterator_traits<Iter>::value_type;

        if (first == last)
        {
            //If the sequence is empty we return a dummy buffer
            return {sycl::buffer<T, 1>(sycl::range<1>(1)), typename std::iterator_traits<Iter>::difference_type{}};
        }
        else
        {
            return {get_buffer_from_iters(first, last), typename std::iterator_traits<Iter>::difference_type{}};
        }
    }

    // for raw pointers and direct pass objects
    template <typename Iter>
    typename std::enable_if<is_passed_directly<Iter>::value, Iter>::type
    operator()(Iter first, Iter)
    {
        return first;
    }

    template <typename Iter>
    typename std::enable_if<is_passed_directly<Iter>::value, const Iter>::type
    operator()(const Iter first, const Iter) const
    {
        return first;
    }
};

template <typename Iterator>
using iterator_buffer_type = decltype(std::declval<get_buffer>()(std::declval<Iterator>(), std::declval<Iterator>()));

template <typename UnaryFunction, typename Iterator>
struct transform_buffer_wrapper
{
    iterator_buffer_type<Iterator> iterator_buffer;
    UnaryFunction functor;
};

// get_access_mode
template <typename Iter, typename Void = void>
struct get_access_mode
{
    static constexpr auto mode = cl::sycl::access::mode::read_write;
};

template <typename Iter> // for any const iterators
struct get_access_mode<Iter, typename std::enable_if<dpstd::__internal::is_const_iterator<Iter>::value, void>::type>
{

    static constexpr auto mode = cl::sycl::access::mode::read;
};

template <typename Iter> // for heterogeneous and non-const iterators
struct get_access_mode<
    Iter, typename std::enable_if<is_hetero_iterator<Iter>::value && !dpstd::__internal::is_const_iterator<Iter>::value,
                                  void>::type>
{

    static constexpr auto mode = Iter::mode;
};
template <typename... Iters> // for zip_iterators
struct get_access_mode<dpstd::zip_iterator<Iters...>>
{
    static constexpr auto mode = std::make_tuple(get_access_mode<Iters>::mode...);
};
// TODO: for counting_iterator

struct ApplyFunc
{
    template <typename Func, typename... Elem>
    auto
    operator()(Func func_i, Elem... elem_i) -> decltype(func_i(elem_i...)) const
    {
        return func_i(elem_i...);
    }
};

//------------------------------------------------------------------------
// functor to get accessor from buffer
// (or to get tuple of accessors from tuple of buffers)
//------------------------------------------------------------------------
template <typename UnaryFunction, typename Iterator>
struct transform_accessor_wrapper;

template <typename BaseIter>
struct get_access
{
  private:
    sycl::handler& cgh;

  public:
    get_access(sycl::handler& cgh_) : cgh(cgh_) {}
    // for common buffers
    template <typename T, typename StartIdx, typename LocalIter = BaseIter>
    sycl::accessor<T, 1, get_access_mode<LocalIter>::mode, sycl::access::target::global_buffer>
    operator()(shifted_buffer<T, StartIdx> buf)
    {
        //std::cout << (unsigned int) get_access_mode<BaseIter>::mode << std::endl;
        return buf.buffer.template get_access<get_access_mode<BaseIter>::mode>(cgh, buf.buffer.get_range(),
                                                                               buf.startIdx);
    }
    // for counting_iterator
    template <typename T>
    dpstd::counting_iterator<T>
    operator()(dpstd::counting_iterator<T> it)
    {
        return it;
    }

    // for transform_iterator
    template <typename UnaryFunction, typename Iterator>
    transform_accessor_wrapper<UnaryFunction, Iterator>
    operator()(const transform_buffer_wrapper<UnaryFunction, Iterator>& buf)
    {
        return {get_access<Iterator>(cgh)(buf.iterator_buffer), buf.functor};
    }

    // for raw pointers and direct pass objects
    template <typename Iter>
    typename std::enable_if<is_passed_directly<Iter>::value, Iter>::type
    operator()(Iter first)
    {
        return first;
    }

    template <typename Iter>
    typename std::enable_if<is_passed_directly<Iter>::value, const Iter>::type
    operator()(const Iter first) const
    {
        return first;
    }
};

template <typename... BaseIters>
struct get_access<dpstd::zip_iterator<BaseIters...>>
{
  private:
    sycl::handler& cgh;

  public:
    get_access(sycl::handler& cgh_) : cgh(cgh_) {}
    // for tuple of buffers
    template <typename... Buffers, typename... StartIdx>
    auto
    operator()(tuple<Buffers...> buf)
        -> decltype(map_tuple(ApplyFunc(), tuple<get_access<BaseIters>...>{get_access<BaseIters>(cgh)...}, buf))
    {
        return map_tuple(ApplyFunc(), tuple<get_access<BaseIters>...>{get_access<BaseIters>(cgh)...}, buf);
    }
};

template <typename Iterator>
using iterator_accessor_type =
    decltype(std::declval<get_access<Iterator>>()(std::declval<iterator_buffer_type<Iterator>>()));

template <typename UnaryFunction, typename Iterator>
struct transform_accessor_wrapper
{
    iterator_accessor_type<Iterator> iterator_accessor;
    UnaryFunction functor;

    template <typename ID>
    typename std::iterator_traits<dpstd::transform_iterator<UnaryFunction, Iterator>>::reference operator[](ID id) const
    {
        return functor(iterator_accessor[id]);
    }
};

//-----------------------------------------------------------------------
// type traits for comparators
//-----------------------------------------------------------------------

// traits for ascending functors
template <typename _Comp>
struct __is_comp_ascending
{
    static constexpr bool value = false;
};
template <typename _T>
struct __is_comp_ascending<std::less<_T>>
{
    static constexpr bool value = true;
};
template <>
struct __is_comp_ascending<dpstd::__internal::__pstl_less>
{
    static constexpr bool value = true;
};

// traits for descending functors
template <typename _Comp>
struct __is_comp_descending
{
    static constexpr bool value = false;
};
template <typename _T>
struct __is_comp_descending<std::greater<_T>>
{
    static constexpr bool value = true;
};
template <>
struct __is_comp_descending<dpstd::__internal::__pstl_greater>
{
    static constexpr bool value = true;
};

//-----------------------------------------------------------------------
// temporary "buffer" constructed over specified container type
//-----------------------------------------------------------------------

template <typename _Unknown>
struct __local_buffer;

template <int __dim, typename _AllocT, typename _T>
struct __local_buffer<cl::sycl::buffer<_T, __dim, _AllocT>>
{
    using type = cl::sycl::buffer<_T, __dim, _AllocT>;
};

//if we take std::tuple as a type for buffer we should convert to internal::tuple
template <int __dim, typename _AllocT, typename... _T>
struct __local_buffer<cl::sycl::buffer<std::tuple<_T...>, __dim, _AllocT>>
{
    using type = cl::sycl::buffer<__internal::tuple<_T...>, __dim, _AllocT>;
};

// __buffer defaulted to sycl::buffer<_T, 1, ...>
template <typename _ExecutionPolicy, typename _T, typename _Container = cl::sycl::buffer<_T, 1>>
struct __buffer;

// impl for sycl::buffer<...>
template <typename _ExecutionPolicy, typename _T, typename _BValueT, int __dim, typename _AllocT>
struct __buffer<_ExecutionPolicy, _T, cl::sycl::buffer<_BValueT, __dim, _AllocT>>
{
  private:
    using __exec_policy_t = __decay_t<_ExecutionPolicy>;
    using __container_t = typename __local_buffer<cl::sycl::buffer<_T, __dim, _AllocT>>::type;

    __container_t __container;

  public:
    __buffer(_ExecutionPolicy /*__exec*/, std::size_t __n_elements) : __container{cl::sycl::range<1>(__n_elements)} {}

    auto
    get() -> decltype(dpstd::begin(__container)) const
    {
        return dpstd::begin(__container);
    }

    __container_t
    get_buffer() const
    {
        return __container;
    }
};

// impl for shifted_buffer
template <typename _ExecutionPolicy, typename _T, typename _StartIdx>
struct __buffer<_ExecutionPolicy, _T, __internal::shifted_buffer<_T, _StartIdx>>
{
  private:
    using __exec_policy_t = __decay_t<_ExecutionPolicy>;
    using __container_t = __internal::shifted_buffer<_T, _StartIdx>;
    using __buf_t = typename __container_t::container_t;

    __container_t __container;

  public:
    __buffer(_ExecutionPolicy /*__exec*/, std::size_t __n_elements)
        : __container{__buf_t{cl::sycl::range<1>(__n_elements)}}
    {
    }

    __container_t
    get_buffer() const
    {
        return __container;
    }
};

template <typename _ExecutionPolicy, typename _T>
struct __sycl_usm_free
{
    _ExecutionPolicy __exec;

    void
    operator()(_T* __memory) const
    {
        cl::sycl::free(__memory, __exec.queue().get_context());
    }
};

template <typename _ExecutionPolicy, typename _T, cl::sycl::usm::alloc __alloc_t>
struct __sycl_usm_alloc
{
    _ExecutionPolicy __exec;

    _T*
    operator()(std::size_t __elements) const
    {
        const auto& __queue = __exec.queue();
        return (_T*)cl::sycl::malloc(sizeof(_T) * __elements, __queue.get_device(), __queue.get_context(), __alloc_t);
    }
};

// impl for USM pointer
template <typename _ExecutionPolicy, typename _T, typename _BValueT>
struct __buffer<_ExecutionPolicy, _T, _BValueT*>
{
  private:
    using __exec_policy_t = __decay_t<_ExecutionPolicy>;
    using __container_t = std::unique_ptr<_T, __sycl_usm_free<__exec_policy_t, _T>>;
    using __alloc_t = cl::sycl::usm::alloc;

    __container_t __container;

  public:
    __buffer(_ExecutionPolicy __exec, std::size_t __n_elements)
        : __container(__sycl_usm_alloc<__exec_policy_t, _T, __alloc_t::shared>{__exec}(__n_elements),
                      __sycl_usm_free<__exec_policy_t, _T>{__exec})
    {
    }

    _T*
    get() const
    {
        return __container.get();
    }

    _T*
    get_buffer() const
    {
        return __container.get();
    }
};

//-----------------------------------------------------------------------
// type traits for objects granting access to some value objects
//-----------------------------------------------------------------------

template <typename _ContainerOrIterator>
struct __memobj_traits
{
    using value_type = typename _ContainerOrIterator::value_type;
};

template <typename _T>
struct __memobj_traits<_T*>
{
    using value_type = _T;
};

} // namespace __internal

template <typename T>
struct __repacked_tuple
{
    using type = T;
};

template <typename... Args>
struct __repacked_tuple<std::tuple<Args...>>
{
    using type = __internal::tuple<Args...>;
};

template <typename T>
using __repacked_tuple_t = typename __repacked_tuple<T>::type;

template <typename _ContainerOrIterable>
using __value_t = typename __internal::__memobj_traits<_ContainerOrIterable>::value_type;

} // namespace __par_backend_hetero
} // namespace dpstd

#endif //_PSTL_parallel_backend_sycl_utils_H
