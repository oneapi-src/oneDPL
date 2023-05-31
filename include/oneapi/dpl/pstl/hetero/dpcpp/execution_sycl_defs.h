// -*- C++ -*-
//===-- execution_sycl_defs.h ---------------------------------------------===//
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

#ifndef _ONEDPL_EXECUTION_SYCL_DEFS_H
#define _ONEDPL_EXECUTION_SYCL_DEFS_H

#include "../../onedpl_config.h"
#include "../../execution_defs.h"

#include "sycl_defs.h"

#include <type_traits>

namespace oneapi
{
namespace dpl
{
namespace execution
{
inline namespace __dpl
{

struct DefaultKernelName;

//We can create device_policy object:
// 1. from sycl::queue
// 2. from sycl::device_selector (implicitly through sycl::queue)
// 3. from sycl::device
// 4. from other device_policy encapsulating the same queue type
template <typename KernelName = DefaultKernelName>
class device_policy
{
  public:
    using kernel_name = KernelName;

    device_policy() = default;
    template <typename OtherName>
    device_policy(const device_policy<OtherName>& other) : q(other.queue())
    {
    }
    explicit device_policy(sycl::queue q_) : q(q_) {}
    explicit device_policy(sycl::device d_) : q(d_) {}
    operator sycl::queue() const { return q; }
    sycl::queue
    queue() const
    {
        return q;
    }

    // For internal use only
    static constexpr ::std::true_type
    __allow_unsequenced()
    {
        return ::std::true_type{};
    }
    // __allow_vector is needed for __is_vectorization_preferred
    static constexpr ::std::true_type
    __allow_vector()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::true_type
    __allow_parallel()
    {
        return ::std::true_type{};
    }

  private:
    sycl::queue q;
};

#if _ONEDPL_FPGA_DEVICE
struct DefaultKernelNameFPGA;
template <unsigned int factor = 1, typename KernelName = DefaultKernelNameFPGA>
class fpga_policy : public device_policy<KernelName>
{
    using base = device_policy<KernelName>;

  public:
    static constexpr unsigned int unroll_factor = factor;

    fpga_policy()
        : base(sycl::queue(
#    if _ONEDPL_FPGA_EMU
              __dpl_sycl::__fpga_emulator_selector()
#    else
              __dpl_sycl::__fpga_selector()
#    endif // _ONEDPL_FPGA_EMU
              ))
    {
    }

    template <unsigned int other_factor, typename OtherName>
    fpga_policy(const fpga_policy<other_factor, OtherName>& other) : base(other.queue()){};
    explicit fpga_policy(sycl::queue q) : base(q) {}
    explicit fpga_policy(sycl::device d) : base(d) {}
};

#endif // _ONEDPL_FPGA_DEVICE

// 2.8, Execution policy objects
#if _ONEDPL_PREDEFINED_POLICIES

// In order to be useful oneapi::dpl::execution::dpcpp_default.queue() from one translation unit should be equal to
// oneapi::dpl::execution::dpcpp_default.queue() from another TU.
// Starting with c++17 we can simply define sycl as inline variable.
#    if _ONEDPL___cplusplus >= 201703L

inline device_policy<> dpcpp_default{};
#        if _ONEDPL_FPGA_DEVICE
inline fpga_policy<> dpcpp_fpga{};
#        endif // _ONEDPL_FPGA_DEVICE

#    endif // _ONEDPL___cplusplus >= 201703L

#endif // _ONEDPL_PREDEFINED_POLICIES

// make_policy functions
template <typename KernelName = DefaultKernelName>
device_policy<KernelName>
make_device_policy(sycl::queue q)
{
    return device_policy<KernelName>(q);
}

template <typename KernelName = DefaultKernelName>
device_policy<KernelName>
make_device_policy(sycl::device d)
{
    return device_policy<KernelName>(d);
}

template <typename NewKernelName, typename OldKernelName = DefaultKernelName>
device_policy<NewKernelName>
make_device_policy(const device_policy<OldKernelName>& policy
#if _ONEDPL_PREDEFINED_POLICIES
                   = dpcpp_default
#endif // _ONEDPL_PREDEFINED_POLICIES
)
{
    return device_policy<NewKernelName>(policy);
}

template <typename NewKernelName, typename OldKernelName = DefaultKernelName>
device_policy<NewKernelName>
make_hetero_policy(const device_policy<OldKernelName>& policy)
{
    return device_policy<NewKernelName>(policy);
}

#if _ONEDPL_FPGA_DEVICE
template <unsigned int unroll_factor = 1, typename KernelName = DefaultKernelNameFPGA>
fpga_policy<unroll_factor, KernelName>
make_fpga_policy(sycl::queue q)
{
    return fpga_policy<unroll_factor, KernelName>(q);
}

template <unsigned int unroll_factor = 1, typename KernelName = DefaultKernelNameFPGA>
fpga_policy<unroll_factor, KernelName>
make_fpga_policy(sycl::device d)
{
    return fpga_policy<unroll_factor, KernelName>(d);
}

template <unsigned int new_unroll_factor, typename NewKernelName, unsigned int old_unroll_factor = 1,
          typename OldKernelName = DefaultKernelNameFPGA>
fpga_policy<new_unroll_factor, NewKernelName>
make_fpga_policy(const fpga_policy<old_unroll_factor, OldKernelName>& policy
#    if _ONEDPL_PREDEFINED_POLICIES
                 = dpcpp_fpga
#    endif // _ONEDPL_PREDEFINED_POLICIES
)
{
    return fpga_policy<new_unroll_factor, NewKernelName>(policy);
}

template <unsigned int new_unroll_factor, typename NewKernelName, unsigned int old_unroll_factor = 1,
          typename OldKernelName = DefaultKernelNameFPGA>
fpga_policy<new_unroll_factor, NewKernelName>
make_hetero_policy(const fpga_policy<old_unroll_factor, OldKernelName>& policy)
{
    return fpga_policy<new_unroll_factor, NewKernelName>(policy);
}
#endif // _ONEDPL_FPGA_DEVICE

} // namespace __dpl

inline namespace v1
{

// 2.3, Execution policy type trait
template <typename... PolicyParams>
struct is_execution_policy<__dpl::device_policy<PolicyParams...>> : ::std::true_type
{
};

#if _ONEDPL_FPGA_DEVICE
template <unsigned int unroll_factor, typename... PolicyParams>
struct is_execution_policy<__dpl::fpga_policy<unroll_factor, PolicyParams...>> : ::std::true_type
{
};
#endif

} // namespace v1
} // namespace execution

namespace __internal
{

// Extension: hetero execution policy type trait
template <typename _T>
struct __is_hetero_execution_policy : ::std::false_type
{
};

template <typename... PolicyParams>
struct __is_hetero_execution_policy<execution::device_policy<PolicyParams...>> : ::std::true_type
{
};

template <typename _T>
struct __is_device_execution_policy : ::std::false_type
{
};

template <typename... PolicyParams>
struct __is_device_execution_policy<execution::device_policy<PolicyParams...>> : ::std::true_type
{
};

template <typename _T>
struct __is_fpga_execution_policy : ::std::false_type
{
};

#if _ONEDPL_FPGA_DEVICE
template <unsigned int unroll_factor, typename... PolicyParams>
struct __is_hetero_execution_policy<execution::fpga_policy<unroll_factor, PolicyParams...>> : ::std::true_type
{
};

template <unsigned int unroll_factor, typename... PolicyParams>
struct __is_fpga_execution_policy<execution::fpga_policy<unroll_factor, PolicyParams...>> : ::std::true_type
{
};

template <typename _T, unsigned int unroll_factor, typename... PolicyParams>
struct __ref_or_copy_impl<execution::fpga_policy<unroll_factor, PolicyParams...>, _T>
{
    using type = _T;
};
#endif

template <typename _T, typename... PolicyParams>
struct __ref_or_copy_impl<execution::device_policy<PolicyParams...>, _T>
{
    using type = _T;
};

// Extension: check if parameter pack is convertible to events
template <class... _Ts>
inline constexpr bool __is_convertible_to_event = (::std::is_convertible_v<::std::decay_t<_Ts>, sycl::event> && ...);

template <typename _T, typename... _Ts>
using __enable_if_convertible_to_events = ::std::enable_if_t<__is_convertible_to_event<_Ts...>, _T>;

// Extension: execution policies type traits
template <typename _ExecPolicy, typename _T, typename... _Events>
using __enable_if_device_execution_policy =
    ::std::enable_if_t<oneapi::dpl::__internal::__is_device_execution_policy<::std::decay_t<_ExecPolicy>>::value &&
                           oneapi::dpl::__internal::__is_convertible_to_event<_Events...>,
                       _T>;

template <typename _ExecPolicy, typename _T>
using __enable_if_hetero_execution_policy =
    ::std::enable_if_t<oneapi::dpl::__internal::__is_hetero_execution_policy<::std::decay_t<_ExecPolicy>>::value, _T>;

template <typename _ExecPolicy, typename _T>
using __enable_if_fpga_execution_policy =
    ::std::enable_if_t<oneapi::dpl::__internal::__is_fpga_execution_policy<::std::decay_t<_ExecPolicy>>::value, _T>;

template <typename _ExecPolicy, typename _T, typename _Op1, typename... _Events>
using __enable_if_device_execution_policy_single_no_default =
    ::std::enable_if_t<oneapi::dpl::__internal::__is_device_execution_policy<::std::decay_t<_ExecPolicy>>::value &&
                           !::std::is_convertible<_Op1, sycl::event>::value &&
                           oneapi::dpl::__internal::__is_convertible_to_event<_Events...>,
                       _T>;

template <typename _ExecPolicy, typename _T, typename _Op1, typename _Op2, typename... _Events>
using __enable_if_device_execution_policy_double_no_default =
    ::std::enable_if_t<oneapi::dpl::__internal::__is_device_execution_policy<::std::decay_t<_ExecPolicy>>::value &&
                           !::std::is_convertible<_Op1, sycl::event>::value &&
                           !::std::is_convertible<_Op2, sycl::event>::value &&
                           oneapi::dpl::__internal::__is_convertible_to_event<_Events...>,
                       _T>;

} // namespace __internal

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXECUTION_SYCL_DEFS_H
