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

#include "support/concurrent_queue.h"
#include "oneapi/dpl/dynamic_selection/ds_properties.h"
#include "oneapi/dpl/internal/dynamic_selection/scoring_policy_defs.h"
#include "oneapi/dpl/internal/dynamic_selection/scheduler_defs.h"

#include <ostream>
#include <vector>
#include <atomic>

struct int_inline_scheduler_t {
  using native_resource_t = int;
  using native_sync_t = int;
  using execution_resource_t = oneapi::dpl::experimental::nop_execution_resource_t<native_resource_t>;
  using native_universe_container_t = std::vector<native_resource_t>;
  using universe_container_t = std::vector<execution_resource_t>;

  class async_wait_t {
  public:
    virtual void wait_for_all() = 0;
    virtual native_sync_t get_native() const = 0;
    virtual ~async_wait_t() {}
  };
  using waiter_container_t = Queue<async_wait_t *>;

  template<typename PropertyHandle>
  class async_wait_impl_t : public async_wait_t {
    PropertyHandle p_;
    native_sync_t w_;
    std::shared_ptr<std::atomic<bool>> wait_reported_;
  public:

    async_wait_impl_t(PropertyHandle p, native_sync_t w) : p_(p), w_(w),
                                                           wait_reported_{std::make_shared<std::atomic<bool>>(false)} { };

    native_sync_t get_native() const override {
      return w_;
    }

    void wait_for_all() override {
      if (wait_reported_->exchange(true) == false) {
        if constexpr (PropertyHandle::should_report_task_completion) {
          oneapi::dpl::experimental::property::report(p_, oneapi::dpl::experimental::property::task_completion);
        }
        if constexpr (PropertyHandle::should_report_task_execution_time) {
          oneapi::dpl::experimental::property::report(p_, oneapi::dpl::experimental::property::task_execution_time, w_);
        }
      }
    }
  };

  universe_container_t universe_;
  waiter_container_t waiters_;

  int_inline_scheduler_t() {
    for (int i = 1; i < 4; ++i)
      universe_.push_back(execution_resource_t{i});
  }

  int_inline_scheduler_t(const native_universe_container_t& u) {
    for (const auto& e : u)
      universe_.push_back(execution_resource_t{e});
  }

  template<typename SelectionHandle, typename Function, typename ...Args>
  auto submit(SelectionHandle h, Function&& f, Args&&... args) {
    using PropertyHandle = typename SelectionHandle::property_handle_t;
    if constexpr (PropertyHandle::should_report_task_submission) {
      oneapi::dpl::experimental::property::report(h.get_property_handle(), oneapi::dpl::experimental::property::task_submission);
    }
    auto w = new async_wait_impl_t<PropertyHandle>(h.get_property_handle(), std::forward<Function>(f)(h.get_native(), std::forward<Args>(args)...));
    waiters_.push(w);
    return *w;
  }

  void wait_for_all() {
    async_wait_t *w;
    waiters_.pop(w);
    w->wait_for_all();
    delete w;

  }

  //
  // Support for property queries
  //

  universe_container_t query(oneapi::dpl::experimental::property::universe_t) const noexcept {
    return universe_;
  }

  auto query(oneapi::dpl::experimental::property::universe_size_t)  const noexcept {
     return universe_.size();
  }

  auto  query(oneapi::dpl::experimental::property::is_device_available_t, execution_resource_t e) const noexcept {
    native_resource_t i = e.get_native();
    for (auto j : universe_) {
      if (j == i) return true;
    }
    return false;
  }

  auto report(oneapi::dpl::experimental::property::universe_t, const universe_container_t &u) noexcept {
    universe_ = u;
  }

  friend std::ostream& operator<<(std::ostream &os, const int_inline_scheduler_t& s);
};

std::ostream& operator<<(std::ostream &os, const int_inline_scheduler_t&) {
  return os << "int_inline_scheduler";
}

std::ostream& operator<<(std::ostream &os, const int_inline_scheduler_t::execution_resource_t &e) {
  return os << e.get_native();
}
inline int_inline_scheduler_t int_inline_scheduler;

