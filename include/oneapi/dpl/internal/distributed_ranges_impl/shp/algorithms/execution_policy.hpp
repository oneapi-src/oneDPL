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

#pragma once

#include <span>
#include <sycl/sycl.hpp>
#include <vector>

namespace oneapi::dpl::experimental::dr::shp
{

struct device_policy
{
    device_policy(sycl::device device) : devices_({device}) {}
    device_policy(sycl::queue queue) : devices_({queue.get_device()}) {}

    device_policy() : devices_({sycl::queue{}.get_device()}) {}

    template <rng::range R>
    requires(std::is_same_v<rng::range_value_t<R>, sycl::device>) device_policy(R&& devices)
        : devices_(rng::begin(devices), rng::end(devices))
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

} // namespace oneapi::dpl::experimental::dr::shp
