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

#ifndef _PSTL_execution_sycl_defs_H
#define _PSTL_execution_sycl_defs_H

#include <CL/sycl.hpp>
#include "../../dpstd_config.h"
#include "../../execution_defs.h"
#if _PSTL_FPGA_DEVICE
#    include <CL/sycl/intel/fpga_extensions.hpp>
#endif

namespace dpstd
{
namespace execution
{
inline namespace __dpstd
{

struct DefaultKernelName
{
};

//We can create device_policy object:
// 1. from sycl::queue or sycl::ordered_queue
// 2. from sycl::device_selector (implicitly through sycl::queue)
// 3. from sycl::device
// 4. from other device_policy encapsulating the same queue type
template <typename DevicePolicy, typename KernelName = DefaultKernelName>
class device_policy : public DevicePolicy
{
  public:
    using kernel_name = KernelName;

    device_policy() = default;
    template <typename OtherPolicy, typename OtherName>
    device_policy(const device_policy<OtherPolicy, OtherName>& other) : q(other.queue())
    {
    }
    device_policy(const cl::sycl::queue& q_) : q(q_) {}
    device_policy(cl::sycl::queue&& q_) : q(std::move(q_)) {}
    device_policy(const cl::sycl::device& device_) : q(device_) {}
    cl::sycl::queue
    queue() const
    {
        return q;
    }

  private:
    cl::sycl::queue q;
};

template <typename DevicePolicy, typename KernelName = DefaultKernelName>
using sycl_policy _POLICY_DEPRECATED = device_policy<DevicePolicy, KernelName>;

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

template <typename KernelName, typename DevicePolicy, typename OldKernelName>
device_policy<DevicePolicy, KernelName>
make_device_policy(const device_policy<DevicePolicy, OldKernelName>& policy)
{
    return device_policy<DevicePolicy, KernelName>(policy);
}

template <typename KernelName = DefaultKernelName>
device_policy<parallel_unsequenced_policy, KernelName>
make_device_policy(const cl::sycl::queue& q)
{
    return device_policy<parallel_unsequenced_policy, KernelName>(q);
}

template <typename KernelName = DefaultKernelName>
device_policy<parallel_unsequenced_policy, KernelName>
make_device_policy(const cl::sycl::device& device)
{
    return device_policy<parallel_unsequenced_policy, KernelName>(device);
}

#if _PSTL_FPGA_DEVICE
template <typename KernelName = DefaultKernelName, int factor = 1, typename DevicePolicy = parallel_unsequenced_policy>
class fpga_device_policy : public device_policy<DevicePolicy, KernelName>
{
    using base = device_policy<DevicePolicy, KernelName>;

  public:
    static constexpr int unroll_factor = factor;

    fpga_device_policy()
        : base(cl::sycl::queue(
#    if _PSTL_FPGA_EMU
              cl::sycl::intel::fpga_emulator_selector {}
#    else
              cl::sycl::intel::fpga_selector {}
#    endif
              ))
    {
    }

    template <typename OtherName, int OtherFactor, typename OtherPolicy>
    fpga_device_policy(const fpga_device_policy<OtherName, OtherFactor, OtherPolicy>& other) : base(other.queue()){};
    fpga_device_policy(const cl::sycl::queue& q_) : base(q_) {}
    fpga_device_policy(cl::sycl::queue&& q_) : base(std::move(q_)) {}
    fpga_device_policy(const cl::sycl::device& device_) : base(device_) {}
};

template <typename KernelName = DefaultKernelName, int factor = 1, typename DevicePolicy = parallel_unsequenced_policy>
fpga_device_policy<KernelName, factor, DevicePolicy>
make_fpga_policy()
{
    return fpga_device_policy<KernelName, factor, DevicePolicy>{};
}

template <typename KernelName = DefaultKernelName, int factor = 1, typename DevicePolicy, typename OldKernelName,
          int old_factor>
fpga_device_policy<KernelName, factor, DevicePolicy>
make_fpga_policy(const fpga_device_policy<OldKernelName, old_factor, DevicePolicy>& policy)
{
    return fpga_device_policy<KernelName, factor, DevicePolicy>(policy);
}

template <typename KernelName = DefaultKernelName, int factor = 1, typename DevicePolicy = parallel_unsequenced_policy>
fpga_device_policy<KernelName, factor, DevicePolicy>
make_fpga_policy(const cl::sycl::queue& q)
{
    return fpga_device_policy<KernelName, factor, DevicePolicy>(q);
}

template <typename KernelName = DefaultKernelName, int factor = 1, typename DevicePolicy = parallel_unsequenced_policy>
fpga_device_policy<KernelName, factor, DevicePolicy>
make_fpga_policy(const cl::sycl::device& device)
{
    return fpga_device_policy<KernelName, factor, DevicePolicy>(device);
}
#endif

// 2.8, Execution policy objects
// In order to be useful dpstd::execition::default_policy.queue() from one translation unit should be equal to
// dpstd::execution::default_policy.queue() from another TU.
// Starting with c++17 we can simply define sycl as inline variable.
// But for c++11 we need to simulate this feature using local static variable and inline function to achieve
// a single definition across all TUs. As it's required for underlying sycl's queue to behave in the same way
// as it's copy, we simply copy-construct a static variable from a reference to that object.
#if __cplusplus >= 201703L

_POLICY_DEPRECATED inline device_policy<parallel_unsequenced_policy> sycl{};
inline device_policy<parallel_unsequenced_policy> default_policy{};
#    if _PSTL_FPGA_DEVICE
inline fpga_device_policy<> fpga_policy{};
#    endif

#else

inline device_policy<parallel_unsequenced_policy>&
__get_default_policy_object()
{
    static device_policy<parallel_unsequenced_policy> __sycl_obj{};
    return __sycl_obj;
}
_POLICY_DEPRECATED static device_policy<parallel_unsequenced_policy> sycl{__get_default_policy_object()};
static device_policy<parallel_unsequenced_policy> default_policy{__get_default_policy_object()};

#    if _PSTL_FPGA_DEVICE
inline fpga_device_policy<>&
__get_fpga_policy_object()
{
    static fpga_device_policy<> __sycl_obj{};
    return __sycl_obj;
}
static fpga_device_policy<> fpga_policy{__get_fpga_policy_object()};
#    endif

#endif
} // namespace __dpstd

inline namespace v1
{

// 2.3, Execution policy type trait
template <typename... PolicyParams>
struct is_execution_policy<__dpstd::device_policy<PolicyParams...>> : std::true_type
{
};

#if _PSTL_FPGA_DEVICE
template <typename KernelName, int factor, typename... PolicyParams>
struct is_execution_policy<__dpstd::fpga_device_policy<KernelName, factor, PolicyParams...>> : std::true_type
{
};
#endif

} // namespace v1
} // namespace execution

namespace __internal
{

// Extension: hetero execution policy type trait
template <typename _T>
struct __is_hetero_execution_policy : std::false_type
{
};

template <typename... PolicyParams>
struct __is_hetero_execution_policy<execution::device_policy<PolicyParams...>> : std::true_type
{
};

template <typename _T>
struct __is_device_execution_policy : std::false_type
{
};

template <typename... PolicyParams>
struct __is_device_execution_policy<execution::device_policy<PolicyParams...>> : std::true_type
{
};

template <typename _T>
struct __is_fpga_execution_policy : std::false_type
{
};

#if _PSTL_FPGA_DEVICE
template <typename KernelName, int factor, typename... PolicyParams>
struct __is_hetero_execution_policy<execution::fpga_device_policy<KernelName, factor, PolicyParams...>> : std::true_type
{
};

template <typename KernelName, int factor, typename... PolicyParams>
struct __is_fpga_execution_policy<execution::fpga_device_policy<KernelName, factor, PolicyParams...>> : std::true_type
{
};

template <typename _T, typename KernelName, int factor, typename... PolicyParams>
struct __ref_or_copy_impl<execution::fpga_device_policy<KernelName, factor, PolicyParams...>, _T>
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
using __enable_if_device_execution_policy = typename std::enable_if<
    dpstd::__internal::__is_device_execution_policy<typename std::decay<_ExecPolicy>::type>::value, _T>::type;

template <typename _ExecPolicy, typename _T>
using __enable_if_hetero_execution_policy = typename std::enable_if<
    dpstd::__internal::__is_hetero_execution_policy<typename std::decay<_ExecPolicy>::type>::value, _T>::type;

template <typename _ExecPolicy, typename _T>
using __enable_if_fpga_execution_policy = typename std::enable_if<
    dpstd::__internal::__is_fpga_execution_policy<typename std::decay<_ExecPolicy>::type>::value, _T>::type;

//-----------------------------------------------------------------------------
// Device run-time information helpers
//-----------------------------------------------------------------------------

template <typename _ExecutionPolicy>
std::size_t
__max_work_group_size(_ExecutionPolicy&& __policy)
{
    return __policy.queue().get_device().template get_info<cl::sycl::info::device::max_work_group_size>();
}

#if _USE_SUB_GROUPS
template <typename _ExecutionPolicy>
std::size_t
__max_sub_group_size(_ExecutionPolicy&& __policy)
{
    // TODO: can get_info<sycl::info::device::sub_group_sizes>() return zero-size vector?
    //       Spec does not say anything about that.
    sycl::vector_class<std::size_t> __supported_sg_sizes =
        __policy.queue().get_device().template get_info<cl::sycl::info::device::sub_group_sizes>();

    // TODO: Since it is unknown if sycl::vector_class returned
    //       by get_info<sycl::info::device::sub_group_sizes>() can be empty,
    //       at() is used instead of operator[] for out of bound check
    return __supported_sg_sizes.at(__supported_sg_sizes.size() - 1);
}
#endif

template <typename _ExecutionPolicy>
cl_uint
__max_compute_units(_ExecutionPolicy&& __policy)
{
    return __policy.queue().get_device().template get_info<cl::sycl::info::device::max_compute_units>();
}

//-----------------------------------------------------------------------------
// Kernel run-time information helpers
//-----------------------------------------------------------------------------

template <typename _ExecutionPolicy>
std::size_t
__kernel_work_group_size(_ExecutionPolicy&& __policy, const sycl::kernel& __kernel)
{
    const auto& __device = __policy.queue().get_device();

    const std::size_t __wg_size =
        __kernel.template get_work_group_info<sycl::info::kernel_work_group::work_group_size>(__device);
    // The variable below is needed to divide __wgroup_size_kernel on CPU because
    // work group size getting from kernel is not enough to allow execution on CPU.
    // It causes CL_OUT_OF_RESOURCES error in runtime on CPU.
    // Experimentally it was found that minimal divisor is 4.
    const std::size_t __cpu_divisor = __device.is_cpu() ? 4 : 1;

    return __wg_size / __cpu_divisor;
}

template <typename _ExecutionPolicy>
long
__kernel_sub_group_size(_ExecutionPolicy&& __policy, const sycl::kernel& __kernel)
{
    auto __device = __policy.queue().get_device();

    // const std::size_t __sg_size = get_sub_group_info<sycl::info::kernel_sub_group::max_sub_group_size_for_ndrange>(
    //    __device, sycl::range<3>{256, 256, 256});
    const std::size_t __sg_size =
        __kernel.template get_sub_group_info<sycl::info::kernel_sub_group::local_size_for_sub_group_count>(__device,
                                                                                                           1)[0];

    return (__device.is_cpu() ? 1 : __sg_size);
}

} // namespace __internal

} // namespace dpstd

#endif /* _PSTL_execution_sycl_defs_H */
