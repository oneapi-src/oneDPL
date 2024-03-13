// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <span>
#include <sycl/sycl.hpp>
#include <vector>

namespace dr::shp {

struct device_policy {
  device_policy(sycl::device device) : devices_({device}) {}
  device_policy(sycl::queue queue) : devices_({queue.get_device()}) {}

  device_policy() : devices_({sycl::queue{}.get_device()}) {}

  template <rng::range R>
    requires(std::is_same_v<rng::range_value_t<R>, sycl::device>)
  device_policy(R &&devices)
      : devices_(rng::begin(devices), rng::end(devices)) {}

  std::span<sycl::device> get_devices() noexcept { return devices_; }

  std::span<const sycl::device> get_devices() const noexcept {
    return devices_;
  }

private:
  std::vector<sycl::device> devices_;
};

} // namespace dr::shp
