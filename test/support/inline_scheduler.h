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

#ifndef _ONEDPL_INLINE_SCHEDULER_H
#define _ONEDPL_INLINE_SCHEDULER_H

#include "oneapi/dpl/dynamic_selection"
#include "oneapi/dpl/internal/dynamic_selection_impl/scheduler_defs.h"
#include "support/concurrent_queue.h"

#include <vector>
#include <atomic>

namespace TestUtils {
struct int_inline_scheduler_t {
  using native_resource_t = int;
  using native_sync_t = int;
  using execution_resource_t = oneapi::dpl::experimental::basic_execution_resource_t<native_resource_t>;
  using native_universe_container_t = std::vector<native_resource_t>;
  using universe_container_t = std::vector<execution_resource_t>;

  class async_wait_t {
  public:
    virtual void wait() = 0;
    virtual native_sync_t get_native() const = 0;
    virtual ~async_wait_t() {}
  };
  using waiter_container_t = concurrent_queue<async_wait_t *>;

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
    void wait() override {
      if (wait_reported_->exchange(true) == false) {
        if constexpr (PropertyHandle::should_report_task_completion) {
          oneapi::dpl::experimental::property::report(p_, oneapi::dpl::experimental::property::task_completion);
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
    if constexpr (!std::is_same_v <SelectionHandle, native_resource_t> || !std::is_same_v <SelectionHandle, execution_resource_t>) {
      using PropertyHandle = typename SelectionHandle::property_handle_t;
      auto w = new async_wait_impl_t<PropertyHandle>(h.get_property_handle(), std::forward<Function>(f)(h.get_native(), std::forward<Args>(args)...));
      waiters_.push(w);
      return *w;
    } else {
      return;
    }
  }

  auto get_wait_list(){
    std::list<async_wait_t*> wlist;
    waiters_.pop_all(wlist);
    return wlist;
  }

  void wait() {
    while(!waiters_.is_empty()){
      async_wait_t *w;
      waiters_.pop(w);
      w->wait();
      delete w;
    }
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

  auto report(oneapi::dpl::experimental::property::universe_t, const universe_container_t &u) noexcept {
    universe_ = u;
  }
};

inline int_inline_scheduler_t int_inline_scheduler;

} //namespace TestUtils

#endif /* _ONEDPL_INLINE_SCHEDULER_H */
