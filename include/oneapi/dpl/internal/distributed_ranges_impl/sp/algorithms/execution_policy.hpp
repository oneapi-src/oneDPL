// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _ONEDPL_DR_SP_EXECUTION_POLICY_HPP
#define _ONEDPL_DR_SP_EXECUTION_POLICY_HPP

#include <span>
#include <sycl/sycl.hpp>
#include <vector>

namespace oneapi::dpl::experimental::dr::sp
{

struct distributed_device_policy
{
    distributed_device_policy(sycl::device device) : devices_({device}) {}
    distributed_device_policy(sycl::queue queue) : devices_({queue.get_device()}) {}

    distributed_device_policy() : devices_({sycl::queue{}.get_device()}) {}

    template <stdrng::range R>
    requires(std::is_same_v<stdrng::range_value_t<R>, sycl::device>) distributed_device_policy(R&& devices)
        : devices_(stdrng::begin(devices), stdrng::end(devices))
    {
    }

    std::span<sycl::device>
    get_devices() noexcept
    {
        return devices_;
    }

    std::span<const sycl::device>
    get_devices() const noexcept
    {
        return devices_;
    }

  private:
    std::vector<sycl::device> devices_;
};

} // namespace oneapi::dpl::experimental::dr::sp

#endif /* _ONEDPL_DR_SP_EXECUTION_POLICY_HPP */