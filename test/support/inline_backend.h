// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_INLINE_SCHEDULER_H
#define _ONEDPL_INLINE_SCHEDULER_H

#include "oneapi/dpl/dynamic_selection"
#include "oneapi/dpl/internal/dynamic_selection_impl/submission_group.h"
#include "support/concurrent_queue.h"

#include <vector>
#include <atomic>

namespace TestUtils {
struct int_inline_backend_t {
  using resource_type = int;
  using wait_type = int;
  using execution_resource_t = oneapi::dpl::experimental::basic_execution_resource_t<resource_type>;
  using native_resource_container_t = std::vector<resource_type>;
  using resource_container_t = std::vector<execution_resource_t>;

  class async_wait_t {
  public:
    virtual void wait() = 0;
    virtual wait_type unwrap() const = 0;
    virtual ~async_wait_t() {}
  };
  using waiter_container_t = oneapi::dpl::experimental::submission_group_t<async_wait_t *>;

  template<typename Selection>
  class async_wait_impl_t : public async_wait_t {
    Selection s_;
    wait_type w_;
    std::shared_ptr<std::atomic<bool>> wait_reported_;
  public:

    async_wait_impl_t(Selection s, wait_type w) : s_(s), w_(w),
                                                           wait_reported_{std::make_shared<std::atomic<bool>>(false)} { };

    wait_type unwrap() const override {
      return w_;
    }
    void wait() override {
      if (wait_reported_->exchange(true) == false) {
        if constexpr (oneapi::dpl::experimental::report_info_v<Selection, oneapi::dpl::experimental::execution_info::task_completion_t>) {
          oneapi::dpl::experimental::report(s_, oneapi::dpl::experimental::execution_info::task_completion);
        }
      }
    }
  };

  resource_container_t resources_;
  waiter_container_t waiters_;

  int_inline_backend_t() {
    for (int i = 1; i < 4; ++i)
      resources_.push_back(execution_resource_t{i});
  }

  int_inline_backend_t(const native_resource_container_t& u) {
    for (const auto& e : u)
      resources_.push_back(execution_resource_t{e});
  }

  template<typename SelectionHandle, typename Function, typename ...Args>
  auto submit(SelectionHandle s, Function&& f, Args&&... args) {
    std::chrono::high_resolution_clock::time_point t0;
    if constexpr (oneapi::dpl::experimental::report_value_v<SelectionHandle, oneapi::dpl::experimental::execution_info::task_time_t>) {
      t0 = std::chrono::high_resolution_clock::now();
    }
    if constexpr(oneapi::dpl::experimental::report_info_v<SelectionHandle, oneapi::dpl::experimental::execution_info::task_submission_t>){
      s.report(oneapi::dpl::experimental::execution_info::task_submission);
    }
    auto w = new async_wait_impl_t<SelectionHandle>(s,
                                                    std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s),
                                                    std::forward<Args>(args)...));
    if constexpr (oneapi::dpl::experimental::report_value_v<SelectionHandle, oneapi::dpl::experimental::execution_info::task_time_t>) {
      report(s, oneapi::dpl::experimental::execution_info::task_time, (std::chrono::high_resolution_clock::now()-t0).count());
    }
    waiters_.add_submission(w);
    return *w;
  }

  auto get_submission_group(){
      return waiters_;
  }

  void wait() {
      waiters_.wait();
  }

  resource_container_t get_resources() const noexcept {
    return resources_;
  }

  auto get_resources_size()  const noexcept {
     return resources_.size();
  }

  template<typename V>
  auto initialize(const V& u) noexcept {
    resources_.clear();
    for (const auto& e : u)
      resources_.push_back(execution_resource_t{e});
  }
};

inline int_inline_backend_t int_inline_backend;

} //namespace TestUtils

#endif /* _ONEDPL_INLINE_SCHEDULER_H */
