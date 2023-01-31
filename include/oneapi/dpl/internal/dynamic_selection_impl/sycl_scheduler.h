/*
    Copyright 2021 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/

#pragma once

#include <CL/sycl.hpp>
#include "tbb/concurrent_queue.h"
#include "oneapi/dpl/dynamic_selection/ds_properties.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scheduler_defs.h"

#include <mutex>
#include <ostream>
#include <stdlib.h>

namespace ds {

  struct sycl_scheduler {

    using native_resource_t = sycl::queue;
    using native_sync_t = sycl::event;
    using execution_resource_t = ds::nop_execution_resource_t<native_resource_t>; 
    using universe_container_t = std::vector<execution_resource_t>;

    class async_wait_t {
    public:
      virtual void wait_for_all() = 0;
      virtual native_sync_t get_native() const = 0;
      virtual ~async_wait_t() {}
    };
    using waiter_container_t = tbb::concurrent_queue<async_wait_t *>;

    template<typename PropertyHandle>
    class async_wait_impl_t : public async_wait_t {
      PropertyHandle p_;
      native_sync_t w_;
      std::chrono::high_resolution_clock::time_point t0_;
      std::shared_ptr<std::atomic<bool>> wait_reported_;
    public:

      async_wait_impl_t(PropertyHandle p, sycl::event e) : p_(p), w_(e), t0_(std::chrono::high_resolution_clock::now()),
                                                           wait_reported_{std::make_shared<std::atomic<bool>>(false)} { };

      native_sync_t get_native() const override {
        return w_;
      }

      void wait_for_all() override {
        w_.wait();
        if (wait_reported_->exchange(true) == false) {
          if constexpr (PropertyHandle::should_report_task_completion) {
            property::report(p_, property::task_completion);
          }
          if constexpr (PropertyHandle::should_report_task_execution_time) {
            property::report(p_, property::task_execution_time, (std::chrono::high_resolution_clock::now()-t0_).count());
          }
        }
      }
    };

    std::mutex global_rank_mutex_;
    universe_container_t global_rank_;
    waiter_container_t waiters_;

    sycl_scheduler() { }

    sycl_scheduler(const sycl_scheduler& v) : global_rank_(v.global_rank_), waiters_(v.waiters_) { }

    template<typename NativeUniverseVector, typename ...Args>
    sycl_scheduler(const NativeUniverseVector& v, Args&&... args) { 
      for (auto e : v) {
        global_rank_.push_back(e);
      } 
    }

    template<typename SelectionHandle, typename Function, typename ...Args>
    auto submit(SelectionHandle h, Function&& f, Args&&... args) {
      using PropertyHandle = typename SelectionHandle::property_handle_t;
      if constexpr (PropertyHandle::should_report_task_submission) {
        property::report(h.get_property_handle(), property::task_submission);
      }
      auto w = new async_wait_impl_t<PropertyHandle>(h.get_property_handle(), f(h.get_native(), std::forward<Args>(args)...));
      waiters_.push(w);
      return *w;
    }

    void wait_for_all() {
      async_wait_t *w;
      while (waiters_.try_pop(w)) {
        w->wait_for_all();
        delete w;
      }
    }

    //
    // Support for property queries
    //

    auto query(ds::property::universe_t) noexcept {
      std::unique_lock<std::mutex> l(global_rank_mutex_);
      if (global_rank_.empty()) {
        auto devices = sycl::device::get_devices();
        for(auto x : devices){
          global_rank_.push_back(sycl::queue{x});
        }
      }
      return global_rank_;
    }

    auto query(ds::property::universe_size_t) noexcept {
      if (global_rank_.empty()) {
        query(ds::property::universe); 
      }
      {
        std::unique_lock<std::mutex> l(global_rank_mutex_);
        return global_rank_.size();
      }
    }

    auto  query(ds::property::is_device_available_t, execution_resource_t e) const noexcept {
      auto device_=e.get_native().get_device();
      return device_.get_info<sycl::info::device::is_available>();
    }

    auto report(ds::property::universe_t, const universe_container_t &gr) noexcept {
      std::unique_lock<std::mutex> l(global_rank_mutex_);
      global_rank_ = gr;
    }

    friend std::ostream& operator<<(std::ostream &os, const sycl_scheduler& s);
  };

  std::ostream& operator<<(std::ostream &os, const sycl_scheduler& s) {
    return os << "sycl_scheduler\n";
  }

  std::ostream& operator<<(std::ostream &os, const sycl_scheduler::execution_resource_t& e) {
    auto device = e.get_native().get_info<sycl::info::queue::device>();
    if (device.is_cpu()) {
      os << "cpu\n";
    } else if (device.is_gpu()) {
      os << "gpu\n";
    } else {
      os << "other\n";
    }
    return os;
  }
}

namespace sycl {
  std::ostream& operator<<(std::ostream &os, const sycl::queue& q) {
    auto device = q.get_info<sycl::info::queue::device>();
    if (device.is_cpu()) {
      os << "cpu\n";
    } else if (device.is_gpu()) {
      os << "gpu\n";
    } else {
      os << "other\n";
    }
    return os;
  }
}

