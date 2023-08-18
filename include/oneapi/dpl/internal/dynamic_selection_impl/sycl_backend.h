// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SYCL_BACKEND_IMPL_H
#define _ONEDPL_SYCL_BACKEND_IMPL_H

#include <CL/sycl.hpp>
#include "oneapi/dpl/internal/dynamic_selection.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/backend_defs.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/concurrent_queue.h"

#include <atomic>
#include <vector>
#include <memory>
#include <list>

namespace oneapi {
namespace dpl {
namespace experimental {

  struct sycl_backend {

    using resource_type = sycl::queue;
    using wait_type = sycl::event;

    using execution_resource_t = oneapi::dpl::experimental::basic_execution_resource_t<resource_type>;
    using resource_container_t = std::vector<execution_resource_t>;

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
        if (wait_reported_->exchange(true) == false) {
          if constexpr (PropertyHandle::should_report_task_completion) {
            property::report(p_, property::task_completion);
          }
        }

      }
    };

    std::mutex global_rank_mutex_;
    resource_container_t global_rank_;
    waiter_container_t waiters_;

    sycl_backend() = default;

    sycl_backend(const sycl_backend& v) = delete;

    template<typename NativeUniverseVector, typename ...Args>
    sycl_backend(const NativeUniverseVector& v, Args&&... args) {
      for (auto e : v) {
        global_rank_.push_back(e);
      }
    }


    template<typename SelectionHandle, typename Function, typename ...Args>
    auto submit(SelectionHandle h, Function&& f, Args&&... args) {
      using PropertyHandle = typename SelectionHandle::property_handle_t;
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

    auto get_resources()  noexcept {
      std::unique_lock<std::mutex> l(global_rank_mutex_);
      if (global_rank_.empty()) {
        auto devices = sycl::device::get_devices();
        for(auto x : devices){
          global_rank_.push_back(sycl::queue{x});
        }
      }
      return global_rank_;
    }

    auto set_universe(const resource_container_t &gr) noexcept {
      std::unique_lock<std::mutex> l(global_rank_mutex_);
      global_rank_ = gr;
    }

  };

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SYCL_BACKEND_IMPL_H*/
