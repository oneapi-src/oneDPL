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
      virtual wait_type unwrap() const = 0;
      virtual ~async_wait_t() {}
    };
    using waiter_container_t = util::concurrent_queue<async_wait_t *>;

    template<typename PropertyHandle>
    class async_wait_impl_t : public async_wait_t {
      wait_type w_;
    public:
      async_wait_impl_t(sycl::event e) : w_(e) { };
      wait_type unwrap() const override { return w_; }
      void wait() override { w_.wait(); }
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
    auto submit(SelectionHandle s, Function&& f, Args&&... args) {
      auto q = unwrap(s);
      if constexpr (report_execution_info_v<SelectionHandle, execution_info::task_submission_t>) {
        report(s, execution_info::task_submission);
      }
      if constexpr(report_execution_info_v<SelectionHandle, execution_info::task_completion_t>) {
        auto e1 = f(q, std::forward<Args>(args)...);
        auto e2 = q.submit([=](sycl::handler& h){
            h.depends_on(e1);
            h.host_task([=](){
              report(s, execution_info::task_completion);
            });
        });
        auto w = new async_wait_impl_t<SelectionHandle>(e2);
        waiters_.push(w);
        return *w;
      } else {
        auto w = new async_wait_impl_t<SelectionHandle>(f(unwrap(s), std::forward<Args>(args)...));
        waiters_.push(w);
        return *w;
      }
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

    template<typename NativeUniverseVector>
    auto set_universe(const NativeUniverseVector &gr) noexcept {
      std::unique_lock<std::mutex> l(global_rank_mutex_);
      global_rank_.clear();
      for (auto e : gr) {
        global_rank_.push_back(e);
      }
    }

  };

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SYCL_BACKEND_IMPL_H*/
