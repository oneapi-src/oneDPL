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
#include <type_traits>
#include <tuple>

#if (__cpluslplus >= 202002L || _MSVC_LANG >= 202002L) && __has_include(<bit>)
#    include <bit>
#else
#    include <cstring> // memcpy
#endif

#include "../../iterator_impl.h"

#include "sycl_defs.h"
#include "sycl_iterator.h"
#include "../../utils.h"

#if _ONEDPL_DEBUG_SYCL
#    include <iostream>
#endif

#define _PRINT_INFO_IN_DEBUG_MODE(...)                                                                                 \
    oneapi::dpl::__par_backend_hetero::__internal::__print_device_debug_info(__VA_ARGS__)

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// aliases for faster access to modes
using access_mode = sycl::access_mode;

// substitution for C++17 convenience types
template <typename _T>
using __decay_t = typename ::std::decay<_T>::type;
template <bool __flag, typename _T = void>
using __enable_if_t = typename ::std::enable_if<__flag, _T>::type;

// Bitwise type casting, same as C++20 std::bit_cast
template <typename _Dst, typename _Src>
__enable_if_t<
    sizeof(_Dst) == sizeof(_Src) && ::std::is_trivially_copyable_v<_Dst> && ::std::is_trivially_copyable_v<_Src>, _Dst>
__dpl_bit_cast(const _Src& __src) noexcept
{
#if SYCL_LANGUAGE_VERSION >= 2020
    return sycl::bit_cast<_Dst>(__src);
#elif (__cpluslplus >= 202002L || _MSVC_LANG >= 202002L) && __has_include(<bit>)
    return ::std::bit_cast<_Dst>(__src);
#elif defined(__has_builtin) && __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(_Dst, __src);
#else
    _Dst __result;
    ::std::memcpy(&__result, &__src, sizeof(_Dst));
    return __result;
#endif
}

// The max power of 2 not exceeding the given value, same as C++20 std::bit_floor
template <typename _T>
__enable_if_t<::std::is_integral<_T>::value && ::std::is_unsigned<_T>::value, _T>
__dpl_bit_floor(_T __x) noexcept
{
    if (__x == 0) return 0;
#if SYCL_LANGUAGE_VERSION >= 2020
    // Use the count-leading-zeros function
    return 1 << (sycl::clz(_T{0}) - sycl::clz(__x) - 1);
#elif (__cpluslplus >= 202002L || _MSVC_LANG >= 202002L) && __has_include(<bit>)
    return ::std::bit_floor(__x);
#else
    // Fill all the lower bits with 1s
    __x |= (__x >> 1);
    __x |= (__x >> 2);
    __x |= (__x >> 4);
    if constexpr (sizeof(_T) > 1) __x |= (__x >> 8);
    if constexpr (sizeof(_T) > 2) __x |= (__x >> 16);
    if constexpr (sizeof(_T) > 4) __x |= (__x >> 32);
    __x += 1; // Now it equals to the next greater power of 2, or 0 in case of wraparound
    return (__x == 0) ? 1 << (sizeof(_T) * 8 - 1) : __x >> 1;
#endif
}

// rounded up result of (__number / __divisor)
template <typename _T1, typename _T2>
constexpr auto
__ceiling_div(_T1 __number, _T2 __divisor) -> decltype((__number - 1) / __divisor + 1)
{
    return (__number - 1) / __divisor + 1;
}

// function to hide zip_iterator creation
template <typename... T>
oneapi::dpl::zip_iterator<T...>
zip(T... args)
{
    return oneapi::dpl::zip_iterator<T...>(args...);
}

// function is needed to wrap kernel name into another policy class
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
    static constexpr bool value = ::std::is_same<_CustomName, oneapi::dpl::execution::DefaultKernelName>::value
#if _ONEDPL_FPGA_DEVICE
                                  || ::std::is_same<_CustomName, oneapi::dpl::execution::DefaultKernelNameFPGA>::value
#endif
        ;
};

template <template <typename...> class _ExternalName, typename _InternalName>
struct _HasDefaultName<_ExternalName<_InternalName>>
{
    static constexpr bool value = _HasDefaultName<_InternalName>::value;
};

template <typename... _Name>
struct __optional_kernel_name;

template <typename _CustomName>
using __kernel_name_provider =
#if __SYCL_UNNAMED_LAMBDA__
    typename ::std::conditional<_HasDefaultName<_CustomName>::value, __optional_kernel_name<>,
                                __optional_kernel_name<_CustomName>>::type;
#else
    __optional_kernel_name<_CustomName>;
#endif

template <typename _KernelName, char...>
struct __composite
{
};

// Compose kernel name by transforming the constexpr string to the sequence of chars
// and instantiate template with variadic non-type template parameters.
// This approach is required to get reliable work group size when kernel is unnamed
#if _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT
template <typename _KernelName, typename _Tp>
class __kernel_name_composer
{
    static constexpr auto __name = __builtin_sycl_unique_stable_name(_Tp);
    static constexpr ::std::size_t __name_size = __builtin_strlen(__name);

    template <::std::size_t... _Is>
    static __composite<_KernelName, __name[_Is]...>
    __compose_kernel_name(::std::index_sequence<_Is...>);

  public:
    using type = decltype(__compose_kernel_name(::std::make_index_sequence<__name_size>{}));
};
#endif // _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT

template <template <typename...> class _BaseName, typename _CustomName, typename... _Args>
using __kernel_name_generator =
#if __SYCL_UNNAMED_LAMBDA__
    ::std::conditional_t<_HasDefaultName<_CustomName>::value,
#    if _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT
                         typename __kernel_name_composer<_BaseName<>, _BaseName<_CustomName, _Args...>>::type,
#    else // _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT
                         _BaseName<_CustomName, _Args...>,
#    endif
                         _BaseName<_CustomName>>;
#else // __SYCL_UNNAMED_LAMBDA__
    _BaseName<_CustomName>;
#endif

template <typename... _KernelNames>
class __kernel_compiler
{
    static constexpr ::std::size_t __kernel_count = sizeof...(_KernelNames);
    using __kernel_array_type = ::std::array<sycl::kernel, __kernel_count>;

    static_assert(__kernel_count > 0, "At least one kernel name should be provided");

  public:
#if _ONEDPL_KERNEL_BUNDLE_PRESENT
    template <typename _Exec>
    static auto
    __compile(_Exec&& __exec)
    {
        ::std::vector<sycl::kernel_id> __kernel_ids{sycl::get_kernel_id<_KernelNames>()...};

        auto __kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            __exec.queue().get_context(), {__exec.queue().get_device()}, __kernel_ids);

        if constexpr (__kernel_count > 1)
            return __make_kernels_array(__kernel_bundle, __kernel_ids, ::std::make_index_sequence<__kernel_count>());
        else
            return __kernel_bundle.template get_kernel(__kernel_ids[0]);
    }

  private:
    template <typename _KernelBundle, typename _KernelIds, ::std::size_t... _Ip>
    static auto
    __make_kernels_array(_KernelBundle __kernel_bundle, _KernelIds& __kernel_ids, ::std::index_sequence<_Ip...>)
    {
        return __kernel_array_type{__kernel_bundle.template get_kernel(__kernel_ids[_Ip])...};
    }
#else
    template <typename _Exec>
    static auto
    __compile(_Exec&& __exec)
    {
        sycl::program __program(__exec.queue().get_context());

        using __return_type = std::conditional_t<(__kernel_count > 1), __kernel_array_type, sycl::kernel>;
        return __return_type{
            (__program.build_with_kernel_type<_KernelNames>(), __program.get_kernel<_KernelNames>())...};
    }
#endif
};

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

//A contract for future class: <sycl::event or other event, a value or sycl::buffers...>
//Impl details: inheretance (private) instead of aggregation for enabling the empty base optimization.
template <typename _Event, typename... _Args>
class __future : private std::tuple<_Args...>
{
    _Event __my_event;

    template <typename _T>
    constexpr auto
    __wait_and_get_value(sycl::buffer<_T>& __buf)
    {
        //according to a contract, returned value is one-element sycl::buffer
        return __buf.template get_access<access_mode::read>()[0];
    }

    template <typename _T>
    constexpr auto
    __wait_and_get_value(_T& __val)
    {
        wait();
        return __val;
    }

public:
    __future(_Event __e, _Args... __args) : std::tuple<_Args...>(__args...), __my_event(__e) {}
    __future(_Event __e, std::tuple<_Args...> __t) : std::tuple<_Args...>(__t), __my_event(__e) {}

    auto
    event() const
    {
        return __my_event;
    }
    operator _Event() const { return event(); }
    void
    wait()
    {
#if !ONEDPL_ALLOW_DEFERRED_WAITING
        __my_event.wait_and_throw();
#endif
    }

    auto
    get()
    {
        if constexpr (sizeof...(_Args) > 0)
        {
            auto& __val = std::get<0>(*this);
            return __wait_and_get_value(__val);
        }
        else
            wait();
    }

    //The internal API. There are cases where the implementation specifies return value  "higher" than SYCL backend,
    //where a future is created.
    template <typename _T>
    auto
    __make_future(_T __t) const
    {
        auto new_val = std::tuple<_T>(__t);
        auto new_tuple = std::tuple_cat(new_val, (std::tuple<_Args...>)*this);
        return __future<_Event, _T, _Args...>(__my_event, new_tuple);
    }
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_parallel_backend_sycl_utils_H
