// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SYCL_SCHEDULER_IMPL_H
#define _ONEDPL_SYCL_SCHEDULER_IMPL_H

#include <CL/sycl.hpp>
#include "oneapi/dpl/internal/dynamic_selection.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scheduler_defs.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/concurrent_queue.h"

#include <atomic>
#include <vector>
#include <memory>
#include <list>

namespace oneapi {
namespace dpl {
namespace experimental {

  struct sycl_scheduler {

    using resource_type = sycl::queue;
    using wait_type = sycl::event;

    using universe_container_t = std::vector<resource_type>;

    class async_wait_t {
    public:
      virtual void wait() = 0;
      virtual wait_type get_native() const = 0;
      virtual ~async_wait_t() {}
    };
    using waiter_container_t = util::concurrent_queue<async_wait_t *>;

    template<typename PropertyHandle>
    class async_wait_impl_t : public async_wait_t {
      PropertyHandle p_;
      wait_type w_;
      std::shared_ptr<std::atomic<bool>> wait_reported_;
    public:

      async_wait_impl_t(PropertyHandle p, sycl::event e) : p_(p), w_(e),
                                                           wait_reported_{std::make_shared<std::atomic<bool>>(false)} { };

      wait_type get_native() const override {
        return w_;
      }

      void wait() override {
        w_.wait();
        if constexpr (PropertyHandle::should_report_task_completion) {
            property::report(p_, property::task_completion);
        }
      }
    };

    std::mutex global_rank_mutex_;
    universe_container_t global_rank_;
    waiter_container_t waiters_;

    sycl_scheduler() = default;

    sycl_scheduler(const sycl_scheduler& v) = delete;

    template<typename NativeUniverseVector, typename ...Args>
    sycl_scheduler(const NativeUniverseVector& v, Args&&... args) {
      for (auto e : v) {
        global_rank_.push_back(e);
      }
    }


    template<typename SelectionHandle, typename Function, typename ...Args>
    auto submit(SelectionHandle h, Function&& f, Args&&... args) {
      using PropertyHandle = typename SelectionHandle::property_handle_t;
      auto prop = h.get_property_handle();
      if constexpr (PropertyHandle::should_report_task_submission) {
        oneapi::dpl::experimental::property::report(prop, oneapi::dpl::experimental::property::task_submission);
      }
      auto w = new async_wait_impl_t<PropertyHandle>(h.get_property_handle(), f(h.get_native(), std::forward<Args>(args)...));
      waiters_.push(w);
      return *w;
    }

    auto get_submission_group(){
       std::list<async_wait_t*> wlist;
       waiters_.pop_all(wlist);
       return wlist;
    }
    void wait() {
      while(!waiters_.empty()){
        async_wait_t *w;
        waiters_.pop(w);
        w->wait();
        delete w;
      }
    }

    auto get_universe()  noexcept {
      std::unique_lock<std::mutex> l(global_rank_mutex_);
      if (global_rank_.empty()) {
        auto devices = sycl::device::get_devices();
        for(auto x : devices){
          global_rank_.push_back(sycl::queue{x});
        }
      }
      return global_rank_;
    }

    auto get_universe_size() noexcept {
      if (global_rank_.empty()) {
        global_rank_=get_universe();
      }
      {
        std::unique_lock<std::mutex> l(global_rank_mutex_);
        return global_rank_.size();
      }
    }

    auto initialize(const universe_container_t &gr) noexcept {
      std::unique_lock<std::mutex> l(global_rank_mutex_);
      global_rank_ = gr;
    }

  };

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SYCL_SCHEDULER_IMPL_H*/
