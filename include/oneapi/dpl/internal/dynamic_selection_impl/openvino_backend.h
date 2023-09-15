// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_openvino_backend_IMPL_H
#define _ONEDPL_openvino_backend_IMPL_H

#include <inference_engine.hpp>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"

#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

#include <chrono>
#include <vector>
#include <memory>
#include <utility>

namespace oneapi {
namespace dpl {
namespace experimental {

  class openvino_backend {
  public:
    using resource_type = InferenceEngine::ExecutableNetwork;
    using wait_type = InferenceEngine::InferRequest;
    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<execution_resource_t>;

  private:

    class async_waiter {
      wait_type e_;
      public:
        async_waiter(wait_type e) : e_(e) {}
        wait_type unwrap() { return e_; }
        void wait() { e_.wait(); }
    };

    class submission_group {
      resource_container_t resources_;
    public:
      submission_group(const resource_container_t& v) : resources_(v) { }

      void wait() {
        for (auto& r : resources_) {
          unwrap(r).wait();
        }
      }
    };

  public:

    openvino_backend(const openvino_backend& v) = delete;
    openvino_backend& operator=(const openvino_backend&) = delete;

    openvino_backend() {
      initialize_default_resources();
      sgroup_ptr_ = std::make_unique<submission_group>(global_rank_);
    }

    template<typename NativeUniverseVector>
    openvino_backend(const NativeUniverseVector& v) {
      global_rank_.reserve(v.size());
      for (auto e : v) {
        global_rank_.push_back(e);
      }
      sgroup_ptr_ = std::make_unique<submission_group>(global_rank_);
    }

    template<typename SelectionHandle, typename Function, typename ...Args>
    auto submit(SelectionHandle s, Function&& f, Args&&... args) {
      auto q = unwrap(s);
      if constexpr (report_info_v<SelectionHandle, execution_info::task_submission_t>) {
        report(s, execution_info::task_submission);
      }
      if constexpr(report_info_v<SelectionHandle, execution_info::task_completion_t>
                   || report_value_v<SelectionHandle, execution_info::task_time_t>) {
        std::chrono::steady_clock::time_point t0;
        if constexpr (report_value_v<SelectionHandle, execution_info::task_time_t>) {
          t0 = std::chrono::steady_clock::now();
        }
        auto e1 = f(q, std::forward<Args>(args)...);
        auto e2 = q.submit([=](sycl::handler& h){
            h.depends_on(e1);
            h.host_task([=](){
              if constexpr(report_info_v<SelectionHandle, execution_info::task_completion_t>){
                s.report(execution_info::task_completion);
              }
              if constexpr(report_value_v<SelectionHandle, execution_info::task_time_t>)
                s.report(execution_info::task_time, (std::chrono::steady_clock::now() - t0).count());
            });
        });
        return async_waiter{e2};
      } else {
        return async_waiter{f(unwrap(s), std::forward<Args>(args)...)};
      }
    }

    auto get_submission_group(){
      return *sgroup_ptr_;
    }

    auto get_resources() {
      return global_rank_;
    }

  private:
    resource_container_t global_rank_;
    std::unique_ptr<submission_group> sgroup_ptr_;

    void initialize_default_resources() {
      auto devices = sycl::device::get_devices();
      for(auto x : devices){
        global_rank_.push_back(sycl::queue{x});
      }
    }
  };

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_openvino_backend_IMPL_H*/
