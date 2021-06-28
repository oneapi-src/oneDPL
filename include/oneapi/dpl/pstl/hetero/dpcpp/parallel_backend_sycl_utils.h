// -*- C++ -*-
//===-- parallel_backend_sycl_utils.h -------------------------------------===//
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

#ifndef _ONEDPL_parallel_backend_sycl_utils_H
#define _ONEDPL_parallel_backend_sycl_utils_H

//!!! NOTE: This file should be included under the macro _ONEDPL_BACKEND_SYCL
#include <CL/sycl.hpp>
#include <type_traits>
#include "../../iterator_impl.h"
#include "sycl_iterator.h"

#include "../../utils.h"

#define _PRINT_INFO_IN_DEBUG_MODE(...)                                                                                 \
    oneapi::dpl::__par_backend_hetero::__internal::__print_device_debug_info(__VA_ARGS__)

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
using access_target = sycl::access::target;
using access_mode = sycl::access_mode;

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

    void operator()(sycl::event){};
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
    operator()(sycl::event __event)
    {
        __event.wait_and_throw();
    }
};

template <typename Op, ::std::size_t CallNumber>
struct __unique_kernel_name;

template <typename Policy, int idx>
using __new_kernel_name = __unique_kernel_name<typename ::std::decay<Policy>::type, idx>;

// function is needed to wrap kernel name into another class
template <template <typename> class _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy)
    -> decltype(oneapi::dpl::execution::make_device_policy<_NewKernelName<typename __decay_t<_Policy>::kernel_name>>(
        ::std::forward<_Policy>(__policy)))
{
    return oneapi::dpl::execution::make_device_policy<_NewKernelName<typename __decay_t<_Policy>::kernel_name>>(
        ::std::forward<_Policy>(__policy));
}

#if _ONEDPL_FPGA_DEVICE
template <template <typename> class _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy)
    -> decltype(oneapi::dpl::execution::make_fpga_policy<__decay_t<_Policy>::unroll_factor,
                                                         _NewKernelName<typename __decay_t<_Policy>::kernel_name>>(
        ::std::forward<_Policy>(__policy)))
{
    return oneapi::dpl::execution::make_fpga_policy<__decay_t<_Policy>::unroll_factor,
                                                    _NewKernelName<typename __decay_t<_Policy>::kernel_name>>(
        ::std::forward<_Policy>(__policy));
}
#endif

namespace __internal
{

// extract the deepest kernel name when we have a policy wrapper that might hide the default name
template <typename _CustomName>
struct _HasDefaultName
{
    static constexpr bool value = std::is_same<_CustomName, oneapi::dpl::execution::DefaultKernelName>::value;
};

template <template <typename...> class _ExternalName, typename _InternalName>
struct _HasDefaultName<_ExternalName<_InternalName>>
{
    static constexpr bool value = _HasDefaultName<_InternalName>::value;
};

template <template <typename...> class _BaseName, typename _CustomName, typename... _Args>
using _KernelName_t =
#if __SYCL_UNNAMED_LAMBDA__
    typename std::conditional<_HasDefaultName<_CustomName>::value, _BaseName<_CustomName, _Args...>,
                              _BaseName<_CustomName>>::type;
#else
    _BaseName<_CustomName>;
#endif

#if _ONEDPL_DEBUG_SYCL
template <typename _Policy>
inline void
// Passing policy by value should be enough for debugging
__print_device_debug_info(_Policy __policy, size_t __wg_size = 0, size_t __mcu = 0)
{
    ::std::cout << "Device info" << ::std::endl;
    ::std::cout << " > device name:         " << oneapi::dpl::__internal::__device_info(__policy) << ::std::endl;
    ::std::cout << " > max compute units:   "
                << (__mcu ? __mcu : oneapi::dpl::__internal::__max_compute_units(__policy)) << ::std::endl;
    ::std::cout << " > max work-group size: "
                << (__wg_size ? __wg_size : oneapi::dpl::__internal::__max_work_group_size(__policy)) << ::std::endl;
}
#else
template <typename _Policy>
inline void __print_device_debug_info(_Policy, size_t = 0, size_t = 0)
{
}
#endif

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
struct __local_buffer<sycl::buffer<_T, __dim, _AllocT>>
{
    using type = sycl::buffer<_T, __dim, _AllocT>;
};

//if we take ::std::tuple as a type for buffer we should convert to internal::tuple
template <int __dim, typename _AllocT, typename... _T>
struct __local_buffer<sycl::buffer<::std::tuple<_T...>, __dim, _AllocT>>
{
    using type = sycl::buffer<oneapi::dpl::__internal::tuple<_T...>, __dim, _AllocT>;
};

// __buffer defaulted to sycl::buffer<_T, 1, ...>
template <typename _ExecutionPolicy, typename _T, typename _Container = sycl::buffer<_T, 1>>
struct __buffer;

// impl for sycl::buffer<...>
template <typename _ExecutionPolicy, typename _T, typename _BValueT, int __dim, typename _AllocT>
struct __buffer<_ExecutionPolicy, _T, sycl::buffer<_BValueT, __dim, _AllocT>>
{
  private:
    using __exec_policy_t = __decay_t<_ExecutionPolicy>;
    using __container_t = typename __local_buffer<sycl::buffer<_T, __dim, _AllocT>>::type;

    __container_t __container;

  public:
    __buffer(_ExecutionPolicy /*__exec*/, ::std::size_t __n_elements) : __container{sycl::range<1>(__n_elements)} {}

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

template <typename _ExecutionPolicy, typename _T>
struct __sycl_usm_free
{
    _ExecutionPolicy __exec;

    void
    operator()(_T* __memory) const
    {
        sycl::free(__memory, __exec.queue().get_context());
    }
};

template <typename _ExecutionPolicy, typename _T, sycl::usm::alloc __alloc_t>
struct __sycl_usm_alloc
{
    _ExecutionPolicy __exec;

    _T*
    operator()(::std::size_t __elements) const
    {
        const auto& __queue = __exec.queue();
        return (_T*)sycl::malloc(sizeof(_T) * __elements, __queue.get_device(), __queue.get_context(), __alloc_t);
    }
};

// impl for USM pointer
template <typename _ExecutionPolicy, typename _T, typename _BValueT>
struct __buffer<_ExecutionPolicy, _T, _BValueT*>
{
  private:
    using __exec_policy_t = __decay_t<_ExecutionPolicy>;
    using __container_t = ::std::unique_ptr<_T, __sycl_usm_free<__exec_policy_t, _T>>;
    using __alloc_t = sycl::usm::alloc;

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

//-----------------------------------------------------------------------
// future and helper classes for async pattern/algorithm
//-----------------------------------------------------------------------

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

// TODO: towards higher abstraction and generic future. implementation specific sycl::event should be hidden
struct __future_base
{
    sycl::event __my_event;

    __future_base() = default;
    __future_base(sycl::event __e) : __my_event(__e) {}
    void
    wait()
    {
#if !ONEDPL_ALLOW_DEFERRED_WAITING
        __my_event.wait_and_throw();
#endif
    }
    operator sycl::event() const { return __my_event; }
};

template <typename _T>
class __future : public __future_base
{
    ::std::size_t __result_idx;
    sycl::buffer<_T> __data;

  public:
    __future(sycl::event __e, size_t __o, sycl::buffer<_T> __b)
        : __par_backend_hetero::__future_base(__e), __data(__b), __result_idx(__o)
    {
    }

    _T
    get()
    {
        return __data.template get_access<access_mode::read>()[__result_idx];
    }
    template <class _Tp, class _Enable>
    friend class oneapi::dpl::__internal::__future;
};

template <>
class __future<void> : public __future_base
{
    ::std::unique_ptr<__lifetime_keeper_base> __tmps;

  public:
    template <typename... _Ts>
    __future(sycl::event __e, _Ts... __t) : __future_base(__e)
    {
        if (sizeof...(__t) != 0)
            __tmps = ::std::unique_ptr<__lifetime_keeper<_Ts...>>(new __lifetime_keeper<_Ts...>(__t...));
    }
    void
    get()
    {
        this->wait();
    }
    template <class _Tp, class _Enable>
    friend class oneapi::dpl::__internal::__future;
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_parallel_backend_sycl_utils_H
