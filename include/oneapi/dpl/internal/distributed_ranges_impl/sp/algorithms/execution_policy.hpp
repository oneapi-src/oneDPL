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

namespace oneapi::dpl::experimental::dr::sp
{

struct sycl_device_collection
{
    sycl_device_collection(sycl::device device) : devices_({device}) {}
    sycl_device_collection(sycl::queue queue) : devices_({queue.get_device()}) {}

    sycl_device_collection() : devices_({sycl::queue{}.get_device()}) {}

    template <stdrng::range R>
    requires(std::is_same_v<stdrng::range_value_t<R>, sycl::device>) sycl_device_collection(R&& devices)
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
