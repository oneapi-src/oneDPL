// -*- C++ -*-
//===-- execution_sycl_defs.h ---------------------------------------------===//
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

#ifndef _ONEDPL_execution_sycl_defs_H
#define _ONEDPL_execution_sycl_defs_H

#include <CL/sycl.hpp>
#include "../../dpstd_config.h"
#include "../../execution_defs.h"
#if _PSTL_FPGA_DEVICE
#    include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

namespace oneapi
{
namespace dpl
{
namespace execution
{
inline namespace __dpstd
{

struct DefaultKernelName
{
};

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
    explicit device_policy(cl::sycl::queue q_) : q(q_) {}
    explicit device_policy(cl::sycl::device d_) : q(d_) {}
    operator cl::sycl::queue() const { return q; }
    cl::sycl::queue
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
    cl::sycl::queue q;
};

template <typename DevicePolicy = parallel_unsequenced_policy, typename KernelName = DefaultKernelName>
using sycl_policy _POLICY_DEPRECATED = device_policy<KernelName>;

#if _PSTL_FPGA_DEVICE
template <unsigned int factor = 1, typename KernelName = DefaultKernelName>
class fpga_policy : public device_policy<KernelName>
{
    using base = device_policy<KernelName>;

  public:
    static constexpr unsigned int unroll_factor = factor;

    fpga_policy()
        : base(cl::sycl::queue(
#    if _PSTL_FPGA_EMU
              cl::sycl::INTEL::fpga_emulator_selector {}
#    else
              cl::sycl::INTEL::fpga_selector {}
#    endif
              ))
    {
    }

    template <unsigned int other_factor, typename OtherName>
    fpga_policy(const fpga_policy<other_factor, OtherName>& other) : base(other.queue()){};
    explicit fpga_policy(cl::sycl::queue q) : base(q) {}
    explicit fpga_policy(cl::sycl::device d) : base(d) {}
};

template <typename KernelName = DefaultKernelName, int factor = 1, typename DevicePolicy = parallel_unsequenced_policy>
using fpga_device_policy _POLICY_DEPRECATED = fpga_policy<factor, KernelName>;

#endif

// 2.8, Execution policy objects
// In order to be useful oneapi::dpl::execution::dpcpp_default.queue() from one translation unit should be equal to
// oneapi::dpl::execution::dpcpp_default.queue() from another TU.
// Starting with c++17 we can simply define sycl as inline variable.
// But for c++11 we need to simulate this feature using local static variable and inline function to achieve
// a single definition across all TUs. As it's required for underlying sycl's queue to behave in the same way
// as it's copy, we simply copy-construct a static variable from a reference to that object.
#if __cplusplus >= 201703L

_POLICY_DEPRECATED inline device_policy<> sycl{};
_POLICY_DEPRECATED inline device_policy<> default_policy{};
inline device_policy<> dpcpp_default{};
#    if _PSTL_FPGA_DEVICE
inline fpga_policy<> dpcpp_fpga{};
#    endif

#else

template <typename DeviceSelector>
inline device_policy<>&
__get_default_policy_object(DeviceSelector selector)
{
    static device_policy<> __sycl_obj(selector);
    return __sycl_obj;
}
_POLICY_DEPRECATED static device_policy<> sycl{__get_default_policy_object(cl::sycl::default_selector{})};
_POLICY_DEPRECATED static device_policy<> default_policy{__get_default_policy_object(cl::sycl::default_selector{})};
static device_policy<> dpcpp_default{__get_default_policy_object(cl::sycl::default_selector{})};

#    if _PSTL_FPGA_DEVICE
inline fpga_policy<>&
__get_fpga_policy_object()
{
    static fpga_policy<> __sycl_obj{};
    return __sycl_obj;
}
static fpga_policy<> dpcpp_fpga{__get_fpga_policy_object()};
#    endif

#endif

// make_policy functions
template <typename KernelName, typename DevicePolicy, typename OldKernelName>
_POLICY_DEPRECATED sycl_policy<DevicePolicy, KernelName>
make_sycl_policy(const sycl_policy<DevicePolicy, OldKernelName>& policy)
{
    return sycl_policy<DevicePolicy, KernelName>(policy);
}

template <typename KernelName = DefaultKernelName>
_POLICY_DEPRECATED sycl_policy<parallel_unsequenced_policy, KernelName>
make_sycl_policy(const cl::sycl::queue& q)
{
    return sycl_policy<parallel_unsequenced_policy, KernelName>(q);
}

template <typename KernelName = DefaultKernelName>
_POLICY_DEPRECATED sycl_policy<parallel_unsequenced_policy, KernelName>
make_sycl_policy(const cl::sycl::device& device)
{
    return sycl_policy<parallel_unsequenced_policy, KernelName>(device);
}

template <typename KernelName = DefaultKernelName>
device_policy<KernelName>
make_device_policy(cl::sycl::queue q)
{
    return device_policy<KernelName>(q);
}

template <typename KernelName = DefaultKernelName>
device_policy<KernelName>
make_device_policy(cl::sycl::device d)
{
    return device_policy<KernelName>(d);
}

template <typename NewKernelName, typename OldKernelName>
device_policy<NewKernelName>
make_device_policy(const device_policy<OldKernelName>& policy = dpcpp_default)
{
    return device_policy<NewKernelName>(policy);
}

#if _PSTL_FPGA_DEVICE
template <unsigned int unroll_factor = 1, typename KernelName = DefaultKernelName>
fpga_policy<unroll_factor, KernelName>
make_fpga_policy(cl::sycl::queue q)
{
    return fpga_policy<unroll_factor, KernelName>(q);
}

template <unsigned int unroll_factor = 1, typename KernelName = DefaultKernelName>
fpga_policy<unroll_factor, KernelName>
make_fpga_policy(cl::sycl::device d)
{
    return fpga_policy<unroll_factor, KernelName>(d);
}

template <unsigned int new_unroll_factor, typename NewKernelName, unsigned int old_unroll_factor,
          typename OldKernelName>
fpga_policy<new_unroll_factor, NewKernelName>
make_fpga_policy(const fpga_policy<old_unroll_factor, OldKernelName>& policy = dpcpp_fpga)
{
    return fpga_policy<new_unroll_factor, NewKernelName>(policy);
}
#endif

} // namespace __dpstd

inline namespace v1
{

// 2.3, Execution policy type trait
template <typename... PolicyParams>
struct is_execution_policy<__dpstd::device_policy<PolicyParams...>> : ::std::true_type
{
};

#if _PSTL_FPGA_DEVICE
template <unsigned int unroll_factor, typename... PolicyParams>
struct is_execution_policy<__dpstd::fpga_policy<unroll_factor, PolicyParams...>> : ::std::true_type
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

#if _PSTL_FPGA_DEVICE
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
    using type = const _T;
};
#endif

template <typename _T, typename... PolicyParams>
struct __ref_or_copy_impl<execution::device_policy<PolicyParams...>, _T>
{
    using type = const _T;
};

// Extension: execution policies type traits
template <typename _ExecPolicy, typename _T>
using __enable_if_device_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

template <typename _ExecPolicy, typename _T>
using __enable_if_hetero_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

template <typename _ExecPolicy, typename _T>
using __enable_if_fpga_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_fpga_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

//-----------------------------------------------------------------------------
// Device run-time information helpers
//-----------------------------------------------------------------------------

template <typename _ExecutionPolicy>
::std::size_t
__max_work_group_size(_ExecutionPolicy&& __policy)
{
    return __policy.queue().get_device().template get_info<cl::sycl::info::device::max_work_group_size>();
}

template <typename _ExecutionPolicy, typename _T>
cl::sycl::cl_ulong
__max_local_allocation_size(_ExecutionPolicy&& __policy, const cl::sycl::cl_ulong& __local_allocation_size)
{
    const auto __local_mem_size =
        __policy.queue().get_device().template get_info<cl::sycl::info::device::local_mem_size>();
    return ::std::min(__local_mem_size / sizeof(_T), __local_allocation_size);
}

#if _USE_SUB_GROUPS
template <typename _ExecutionPolicy>
::std::size_t
__max_sub_group_size(_ExecutionPolicy&& __policy)
{
    // TODO: can get_info<sycl::info::device::sub_group_sizes>() return zero-size vector?
    //       Spec does not say anything about that.
    cl::sycl::vector_class<::std::size_t> __supported_sg_sizes =
        __policy.queue().get_device().template get_info<cl::sycl::info::device::sub_group_sizes>();

    // TODO: Since it is unknown if sycl::vector_class returned
    //       by get_info<sycl::info::device::sub_group_sizes>() can be empty,
    //       at() is used instead of operator[] for out of bound check
    return __supported_sg_sizes.at(__supported_sg_sizes.size() - 1);
}
#endif

template <typename _ExecutionPolicy>
cl::sycl::cl_uint
__max_compute_units(_ExecutionPolicy&& __policy)
{
    return __policy.queue().get_device().template get_info<cl::sycl::info::device::max_compute_units>();
}

//-----------------------------------------------------------------------------
// Kernel run-time information helpers
//-----------------------------------------------------------------------------

template <typename _ExecutionPolicy>
::std::size_t
__kernel_work_group_size(_ExecutionPolicy&& __policy, const cl::sycl::kernel& __kernel)
{
    const auto& __device = __policy.queue().get_device();
    auto __max_wg_size =
        __kernel.template get_work_group_info<cl::sycl::info::kernel_work_group::work_group_size>(__device);
    // The variable below is needed to achieve better performance on CPU devices.
    // Experimentally it was found that the most common divisor is 4 with all patterns.
    // TODO: choose the divisor according to specific pattern.
    const ::std::size_t __cpu_divisor = __device.is_cpu() ? 4 : 1;

    return __max_wg_size / __cpu_divisor;
}

template <typename _ExecutionPolicy>
long
__kernel_sub_group_size(_ExecutionPolicy&& __policy, const cl::sycl::kernel& __kernel)
{
    auto __device = __policy.queue().get_device();
    auto __wg_size = __kernel_work_group_size(::std::forward<_ExecutionPolicy>(__policy), __kernel);
    const ::std::size_t __sg_size =
        __kernel.template get_sub_group_info<sycl::info::kernel_sub_group::max_sub_group_size>(
            __device, sycl::range<3>{__wg_size, 1, 1});
    return __sg_size;
}

} // namespace __internal

} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_execution_sycl_defs_H */
