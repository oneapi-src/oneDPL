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

#ifndef _ONEDPL_execution_sycl_defs_H
#define _ONEDPL_execution_sycl_defs_H

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
              __dpl_sycl::__fpga_emulator_selector {}
#    else
              __dpl_sycl::__fpga_selector {}
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
template <bool...>
struct __is_true_helper
{
};

template <bool... _Ts>
using __is_all_true = ::std::is_same<__is_true_helper<_Ts..., true>, __is_true_helper<true, _Ts...>>;

template <class... _Ts>
using __is_convertible_to_event =
    __is_all_true<::std::is_convertible<typename ::std::decay<_Ts>::type, sycl::event>::value...>;

template <typename _T, typename... _Events>
using __enable_if_convertible_to_events =
    typename ::std::enable_if<oneapi::dpl::__internal::__is_convertible_to_event<_Events...>::value, _T>::type;

// Extension: execution policies type traits
template <typename _ExecPolicy, typename _T, typename... _Events>
using __enable_if_device_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value &&
        oneapi::dpl::__internal::__is_convertible_to_event<_Events...>::value,
    _T>::type;

template <typename _ExecPolicy, typename _T>
using __enable_if_hetero_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

template <typename _ExecPolicy, typename _T>
using __enable_if_fpga_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_fpga_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

//-----------------------------------------------------------------------------
// Device run-time information helpers
//-----------------------------------------------------------------------------

#if _ONEDPL_DEBUG_SYCL
template <typename _ExecutionPolicy>
::std::string
__device_info(_ExecutionPolicy&& __policy)
{
    return __policy.queue().get_device().template get_info<sycl::info::device::name>();
}
#endif

template <typename _ExecutionPolicy>
::std::size_t
__max_work_group_size(_ExecutionPolicy&& __policy)
{
    return __policy.queue().get_device().template get_info<sycl::info::device::max_work_group_size>();
}

template <typename _ExecutionPolicy, typename _Size>
_Size
__max_local_allocation_size(_ExecutionPolicy&& __policy, _Size __type_size, _Size __local_allocation_size)
{
    return ::std::min(__policy.queue().get_device().template get_info<sycl::info::device::local_mem_size>() /
                          __type_size,
                      __local_allocation_size);
}

#if _USE_SUB_GROUPS
template <typename _ExecutionPolicy>
::std::size_t
__max_sub_group_size(_ExecutionPolicy&& __policy)
{
    auto __supported_sg_sizes = __policy.queue().get_device().template get_info<sycl::info::device::sub_group_sizes>();

    //The result of get_info<sycl::info::device::sub_group_sizes>() can be empty - the function returns 0;
    return __supported_sg_sizes.empty() ? 0 : __supported_sg_sizes.back();
}
#endif

template <typename _ExecutionPolicy>
auto
__max_compute_units(_ExecutionPolicy&& __policy)
    -> decltype(__policy.queue().get_device().template get_info<sycl::info::device::max_compute_units>())
{
    return __policy.queue().get_device().template get_info<sycl::info::device::max_compute_units>();
}

//-----------------------------------------------------------------------------
// Kernel run-time information helpers
//-----------------------------------------------------------------------------

// 20201214 value corresponds to Intel(R) oneAPI C++ Compiler Classic 2021.1.2 Patch release
#define _USE_KERNEL_DEVICE_SPECIFIC_API (__SYCL_COMPILER_VERSION > 20201214) || (_ONEDPL_LIBSYCL_VERSION >= 50700)

template <typename _ExecutionPolicy>
::std::size_t
__kernel_work_group_size(_ExecutionPolicy&& __policy, const sycl::kernel& __kernel)
{
    const sycl::device& __device = __policy.queue().get_device();
    const ::std::size_t __max_wg_size =
#if _USE_KERNEL_DEVICE_SPECIFIC_API
        __kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(__device);
#else
        __kernel.template get_work_group_info<sycl::info::kernel_work_group::work_group_size>(__device);
#endif
    // The variable below is needed to achieve better performance on CPU devices.
    // Experimentally it was found that the most common divisor is 4 with all patterns.
    // TODO: choose the divisor according to specific pattern.
    ::std::size_t __cpu_divisor = 1;
    if (__device.is_cpu() && __max_wg_size >= 4)
        __cpu_divisor = 4;

    return __max_wg_size / __cpu_divisor;
}

template <typename _ExecutionPolicy>
::std::uint32_t
__kernel_sub_group_size(_ExecutionPolicy&& __policy, const sycl::kernel& __kernel)
{
    const sycl::device& __device = __policy.queue().get_device();
    const ::std::size_t __wg_size = __kernel_work_group_size(::std::forward<_ExecutionPolicy>(__policy), __kernel);
    const ::std::uint32_t __sg_size =
#if _USE_KERNEL_DEVICE_SPECIFIC_API
        __kernel.template get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
            __device
#    if _ONEDPL_LIBSYCL_VERSION < 60000
            ,
            sycl::range<3> { __wg_size, 1, 1 }
#    endif
        );
#else
        __kernel.template get_sub_group_info<sycl::info::kernel_sub_group::max_sub_group_size>(
            __device, sycl::range<3>{__wg_size, 1, 1});
#endif
    return __sg_size;
}

} // namespace __internal

} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_execution_sycl_defs_H */
