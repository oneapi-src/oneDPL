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

#ifndef _ONEDPL_parallel_backend_sycl_utils_H
#define _ONEDPL_parallel_backend_sycl_utils_H

//!!! NOTE: This file should be included under the macro _PSTL_BACKEND_SYCL
#include <CL/sycl.hpp>
#include <type_traits>
#include "../../iterator_impl.h"
#include "sycl_iterator.h"

#include "../../utils.h"

#define _PRINT_INFO_IN_DEBUG_MODE(...)                                                                                 \
    oneapi::dpl::__par_backend_hetero::__internal::__print_device_debug_info(__VA_ARGS__)

//TODO: The dpcpp extensions depend on the internal interface, so the code below is to keep the backward compatibility
namespace std
{
template <size_t _Idx, typename... _Tp>
_DPSTD_DEPRECATED auto
get(oneapi::dpl::__ranges::zip_view<_Tp...>& __a) -> decltype(::std::get<_Idx>(__a.tuple()))
{
    return ::std::get<_Idx>(__a.tuple());
}
} // namespace std

namespace oneapi
{
namespace dpl
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
using __decay_t = typename ::std::decay<_T>::type;
template <bool __flag, typename _T = void>
using __enable_if_t = typename ::std::enable_if<__flag, _T>::type;

// function to hide zip_iterator creation
template <typename... T>
oneapi::dpl::zip_iterator<T...>
zip(T... args)
{
    return oneapi::dpl::zip_iterator<T...>(args...);
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

// struct for checking if iterator is heterogeneous or not
template <typename Iter, typename Void = void> // for non-heterogeneous iterators
struct is_hetero_iterator : ::std::false_type
{
};

template <typename Iter> // for heterogeneous iterators
struct is_hetero_iterator<Iter, typename ::std::enable_if<Iter::is_hetero::value, void>::type> : ::std::true_type
{
};
// struct for checking if iterator should be passed directly to device or not
template <typename Iter, typename Void = void> // for iterators that should not be passed directly
struct is_passed_directly : ::std::false_type
{
};

template <typename Iter> // for iterators defined as direct pass
struct is_passed_directly<Iter, typename ::std::enable_if<Iter::is_passed_directly::value, void>::type>
    : ::std::true_type
{
};

template <typename Iter> // for pointers to objects on device
struct is_passed_directly<Iter, typename ::std::enable_if<::std::is_pointer<Iter>::value, void>::type>
    : ::std::true_type
{
};

// functor to extract buffer from iterator or create temporary buffer to run on device

template <typename Iterator, typename UnaryFunction>
struct transform_buffer_wrapper;

// TODO: shifted_buffer can be removed when sub-buffers over sub-buffers will be supported.
template <typename T, typename StartIdx>
struct shifted_buffer
{
    using container_t = cl::sycl::buffer<T, 1>;
    using value_type = T;
    container_t buffer;
    StartIdx startIdx{};

    shifted_buffer(container_t&& buf) : buffer(::std::move(buf)) {}

    shifted_buffer(const container_t& buf, StartIdx sId) : buffer(buf), startIdx(sId) {}
};

struct get_buffer
{
    // for heterogeneous iterator
    template <typename Iter>
    typename ::std::enable_if<is_hetero_iterator<Iter>::value,
                              shifted_buffer<typename ::std::iterator_traits<Iter>::value_type,
                                             typename ::std::iterator_traits<Iter>::difference_type>>::type
    operator()(Iter it, Iter)
    {
        return {it.get_buffer(), it - oneapi::dpl::begin(it.get_buffer())};
    }

    // for counting_iterator
    // To not multiply buffers without necessity it was decided to return counting_iterator
    // Counting_iterator already contains idx as dereferenced value. So idx should be 0
    template <typename T>
    oneapi::dpl::counting_iterator<T>
    operator()(oneapi::dpl::counting_iterator<T> it, oneapi::dpl::counting_iterator<T>)
    {
        return it;
    }
    // for zip_iterator
    template <typename... Iters>
    auto
    operator()(oneapi::dpl::zip_iterator<Iters...> it, oneapi::dpl::zip_iterator<Iters...> it2)
        -> decltype(oneapi::dpl::__internal::map_tuple(*this, it.base(), it2.base()))
    {
        return oneapi::dpl::__internal::map_tuple(*this, it.base(), it2.base());
    }

    // for transform_iterator
    template <typename Iterator, typename UnaryFunction>
    transform_buffer_wrapper<Iterator, UnaryFunction>
    operator()(oneapi::dpl::transform_iterator<Iterator, UnaryFunction> it1,
               oneapi::dpl::transform_iterator<Iterator, UnaryFunction> it2)
    {
        return {operator()(it1.base(), it2.base()), it1.functor()};
    }

    // the function is needed to create buffer over iterators depending on const or non-const iterators
    // for non-const iterators
    template <typename Iter>
    typename ::std::enable_if<!oneapi::dpl::__internal::is_const_iterator<Iter>::value,
                              cl::sycl::buffer<typename ::std::iterator_traits<Iter>::value_type, 1>>::type
    get_buffer_from_iters(Iter first, Iter last)
    {
        auto temp_buf = cl::sycl::buffer<typename ::std::iterator_traits<Iter>::value_type, 1>(first, last);
        temp_buf.set_final_data(first);
        return temp_buf;
    }

    // for const iterators
    template <typename Iter>
    typename ::std::enable_if<oneapi::dpl::__internal::is_const_iterator<Iter>::value,
                              cl::sycl::buffer<typename ::std::iterator_traits<Iter>::value_type, 1>>::type
    get_buffer_from_iters(Iter first, Iter last)
    {
        return sycl::buffer<typename ::std::iterator_traits<Iter>::value_type, 1>(first, last);
    }

    // for host iterator
    template <typename Iter>
    typename ::std::enable_if<!is_hetero_iterator<Iter>::value && !is_passed_directly<Iter>::value,
                              shifted_buffer<typename ::std::iterator_traits<Iter>::value_type,
                                             typename ::std::iterator_traits<Iter>::difference_type>>::type
    operator()(Iter first, Iter last)
    {
        using T = typename ::std::iterator_traits<Iter>::value_type;

        if (first == last)
        {
            //If the sequence is empty we return a dummy buffer
            return {sycl::buffer<T, 1>(sycl::range<1>(1)), typename ::std::iterator_traits<Iter>::difference_type{}};
        }
        else
        {
            return {get_buffer_from_iters(first, last), typename ::std::iterator_traits<Iter>::difference_type{}};
        }
    }

    // for raw pointers and direct pass objects
    template <typename Iter>
    typename ::std::enable_if<is_passed_directly<Iter>::value, Iter>::type
    operator()(Iter first, Iter)
    {
        return first;
    }

    template <typename Iter>
    typename ::std::enable_if<is_passed_directly<Iter>::value, const Iter>::type
    operator()(const Iter first, const Iter) const
    {
        return first;
    }
};

template <typename Iterator>
using iterator_buffer_type =
    decltype(::std::declval<get_buffer>()(::std::declval<Iterator>(), ::std::declval<Iterator>()));

template <typename Iterator, typename UnaryFunction>
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
struct get_access_mode<Iter,
                       typename ::std::enable_if<oneapi::dpl::__internal::is_const_iterator<Iter>::value, void>::type>
{

    static constexpr auto mode = cl::sycl::access::mode::read;
};

template <typename Iter> // for heterogeneous and non-const iterators
struct get_access_mode<
    Iter, typename ::std::enable_if<
              is_hetero_iterator<Iter>::value && !oneapi::dpl::__internal::is_const_iterator<Iter>::value, void>::type>
{

    static constexpr auto mode = Iter::mode;
};
template <typename... Iters> // for zip_iterators
struct get_access_mode<oneapi::dpl::zip_iterator<Iters...>>
{
    static constexpr auto mode = ::std::make_tuple(get_access_mode<Iters>::mode...);
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
template <typename Iterator, typename UnaryFunction>
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
        //::std::cout << (unsigned int) get_access_mode<BaseIter>::mode << ::std::endl;
        return buf.buffer.template get_access<get_access_mode<BaseIter>::mode>(cgh, buf.buffer.get_range(),
                                                                               buf.startIdx);
    }
    // for counting_iterator
    template <typename T>
    oneapi::dpl::counting_iterator<T>
    operator()(oneapi::dpl::counting_iterator<T> it)
    {
        return it;
    }

    // for transform_iterator
    template <typename Iterator, typename UnaryFunction>
    transform_accessor_wrapper<Iterator, UnaryFunction>
    operator()(const transform_buffer_wrapper<Iterator, UnaryFunction>& buf)
    {
        return {get_access<Iterator>(cgh)(buf.iterator_buffer), buf.functor};
    }

    // for raw pointers and direct pass objects
    template <typename Iter>
    typename ::std::enable_if<is_passed_directly<Iter>::value, Iter>::type
    operator()(Iter first)
    {
        return first;
    }

    template <typename Iter>
    typename ::std::enable_if<is_passed_directly<Iter>::value, const Iter>::type
    operator()(const Iter first) const
    {
        return first;
    }
};

template <typename... BaseIters>
struct get_access<oneapi::dpl::zip_iterator<BaseIters...>>
{
  private:
    sycl::handler& cgh;

  public:
    get_access(sycl::handler& cgh_) : cgh(cgh_) {}
    // for tuple of buffers
    template <typename... Buffers, typename... StartIdx>
    auto
    operator()(oneapi::dpl::__internal::tuple<Buffers...> buf) -> decltype(oneapi::dpl::__internal::map_tuple(
        ApplyFunc(), oneapi::dpl::__internal::tuple<get_access<BaseIters>...>{get_access<BaseIters>(cgh)...}, buf))
    {
        return oneapi::dpl::__internal::map_tuple(
            ApplyFunc(), oneapi::dpl::__internal::tuple<get_access<BaseIters>...>{get_access<BaseIters>(cgh)...}, buf);
    }
};

template <typename Iterator>
using iterator_accessor_type =
    decltype(::std::declval<get_access<Iterator>>()(::std::declval<iterator_buffer_type<Iterator>>()));

template <typename Iterator, typename UnaryFunction>
struct transform_accessor_wrapper
{
    iterator_accessor_type<Iterator> iterator_accessor;
    UnaryFunction functor;

    template <typename ID>
    typename ::std::iterator_traits<oneapi::dpl::transform_iterator<Iterator, UnaryFunction>>::reference
    operator[](ID id) const
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
struct __is_comp_ascending<::std::less<_T>>
{
    static constexpr bool value = true;
};
template <>
struct __is_comp_ascending<oneapi::dpl::__internal::__pstl_less>
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
struct __is_comp_descending<::std::greater<_T>>
{
    static constexpr bool value = true;
};
template <>
struct __is_comp_descending<oneapi::dpl::__internal::__pstl_greater>
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

//if we take ::std::tuple as a type for buffer we should convert to internal::tuple
template <int __dim, typename _AllocT, typename... _T>
struct __local_buffer<cl::sycl::buffer<::std::tuple<_T...>, __dim, _AllocT>>
{
    using type = cl::sycl::buffer<oneapi::dpl::__internal::tuple<_T...>, __dim, _AllocT>;
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
    __buffer(_ExecutionPolicy /*__exec*/, ::std::size_t __n_elements) : __container{cl::sycl::range<1>(__n_elements)} {}

    auto
    get() -> decltype(oneapi::dpl::begin(__container)) const
    {
        return oneapi::dpl::begin(__container);
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
    __buffer(_ExecutionPolicy /*__exec*/, ::std::size_t __n_elements)
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
    operator()(::std::size_t __elements) const
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
    using __container_t = ::std::unique_ptr<_T, __sycl_usm_free<__exec_policy_t, _T>>;
    using __alloc_t = cl::sycl::usm::alloc;

    __container_t __container;

  public:
    __buffer(_ExecutionPolicy __exec, ::std::size_t __n_elements)
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
struct __repacked_tuple<::std::tuple<Args...>>
{
    using type = oneapi::dpl::__internal::tuple<Args...>;
};

template <typename T>
using __repacked_tuple_t = typename __repacked_tuple<T>::type;

template <typename _ContainerOrIterable>
using __value_t = typename __internal::__memobj_traits<_ContainerOrIterable>::value_type;

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_parallel_backend_sycl_utils_H
