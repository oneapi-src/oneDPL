// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_AUTO_TUNE_POLICY_H
#define _ONEDPL_AUTO_TUNE_POLICY_H

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"

namespace oneapi {
namespace dpl {
namespace experimental {

  template <typename Backend=sycl_backend, typename... KeyArgs>
  class auto_tune_policy {

    static constexpr double never_resample = 0.0;
    static constexpr int deferred_initialization = -1;
    static constexpr int use_best_resource = -1;

    using wrapped_resource_t = typename std::decay<Backend>::type::execution_resource_t;
    using size_type = typename std::vector<typename Backend::resource_type>::size_type;

    using timing_t = uint64_t;

    struct resource_with_offset_t {
      wrapped_resource_t r_;
      size_type offset_; 
    };

    struct time_data_t {
      uint64_t num_timings_ = 0;
      timing_t value_ = 0;
    };

    struct tuner_t {
      std::mutex m_;

      std::chrono::high_resolution_clock::time_point t0_;

      timing_t best_timing_ = std::numeric_limits<timing_t>::max();
      resource_with_offset_t best_resource_;

      const size_type max_resource_to_profile_;
      uint64_t next_resource_to_profile_ = 0; // as offset in resources

      using time_by_offset_t = std::map<size_type, time_data_t>;
      time_by_offset_t time_by_offset_;

      double resample_time_ = 0;

      tuner_t(resource_with_offset_t br, size_type resources_size, double rt) 
        : t0_(std::chrono::high_resolution_clock::now()), 
          best_resource_(br), 
          max_resource_to_profile_(resources_size),
          resample_time_(rt) {}

      size_type get_resource_to_profile() {
        std::unique_lock<std::mutex> l(m_);
        if (next_resource_to_profile_ < max_resource_to_profile_) {
          return next_resource_to_profile_++;
        } else if (resample_time_ == never_resample) {
          return use_best_resource;
        } else {
          auto now = std::chrono::high_resolution_clock::now();
          auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now-t0_).count();
          if (ms < resample_time_) {
            return use_best_resource;
          } else {
            t0_ = now;
            return next_resource_to_profile_ = 0;
          }
        }
      }

      // called to add new profile info
      void add_new_timing(resource_with_offset_t r, timing_t t) {
        std::unique_lock<std::mutex> l(m_);
        auto offset = r.offset_;
        timing_t new_value = t;
        if (time_by_offset_.count(offset) == 0) {
          time_by_offset_[offset] = time_data_t{1, t};
        } else {
          auto &td = time_by_offset_[offset];
          auto n = td.num_timings_;
          new_value = (n*td.value_+t)/(n+1);
          td.num_timings_ = n+1;
          td.value_ = new_value;
        }
        if (new_value < best_timing_) {
          best_timing_ = new_value;
          best_resource_ = r;
        }
      } 

    };

   public:

    // Needed by Policy Traits
    using resource_type = decltype(unwrap(std::declval<wrapped_resource_t>()));
    using wait_type = typename Backend::wait_type;

    class selection_type {
      resource_with_offset_t resource_;
      std::shared_ptr<tuner_t> tuner_;

    public:
      selection_type(resource_with_offset_t r, std::shared_ptr<tuner_t> t) 
        : resource_(r), tuner_(t) {}

      auto unwrap() { return ::oneapi::dpl::experimental::unwrap(resource_.r_); }

      //template<typename Info> void report(const Info&) { }

      void report(const execution_info::task_time_t&, const typename execution_info::task_time_t::value_type& v) {
        tuner_->add_new_timing(resource_, v);
      }
    };

    auto_tune_policy(double resample_time=never_resample) {
      if (resample_time != deferred_initialization) {
        initialize(resample_time);
      }
    }

    auto_tune_policy(const std::vector<resource_type>& u, double resample_time=never_resample) {
      if (resample_time != deferred_initialization) {
        initialize(u, resample_time);
      }
    }
   
    void initialize(double resample_time=never_resample) {
      backend_ = std::make_shared<Backend>();
      initialize_impl(resample_time);
    }

    void initialize(const std::vector<resource_type>& u, double resample_time=never_resample) {
      backend_ = std::make_shared<Backend>(u);
      initialize_impl(resample_time);
    }

    template<typename Function, typename ...Args>
    selection_type select(Function&& f, Args&&...args) {
      auto k =  make_task_key(std::forward<Function>(f), std::forward<Args>(args)...);
      auto t  = tuner_by_key_[k];
      auto offset = t->get_resource_to_profile();
      if (offset == use_best_resource) {
        return selection_type{t->best_resource_, t}; 
      } else {
        auto r = resources_with_offset_[offset];
        return selection_type{r, t}; 
      } 
    }

    template<typename Function, typename ...Args>
    auto submit(selection_type e, Function&& f, Args&&... args) {
      return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    auto get_resources() {
       return backend_->get_resources();
    }

    auto get_submission_group() {
      return backend_->get_submission_group();
    }

  private:

    //
    // types
    //


    using task_key_t = std::tuple<void *, KeyArgs...>;
    using tuner_by_key_t = std::map<task_key_t, std::shared_ptr<tuner_t>>;

    //
    // member variables
    //

    std::shared_ptr<Backend> backend_;

    std::vector<resource_with_offset_t> resources_with_offset_;

    resource_with_offset_t best_resource_;

    tuner_by_key_t tuner_by_key_;

    double resample_time_ = 0.0;

    //
    // private member functions
    //

    void initialize_impl(double resample_time=never_resample) {
      resample_time_ = resample_time;
      auto u = get_resources();
      for (size_type i = 0; i < u.size(); ++i) {
        resources_with_offset_.push_back(resource_with_offset_t{u[i], i});
      }
    }

    template<typename Function, typename... Args>
    task_key_t make_task_key(Function&& f, Args&&... args) {
      task_key_t k = std::tuple_cat(std::tuple<void *>(&f), std::make_tuple(std::forward<Args>(args)...));
      if (tuner_by_key_.count(k) == 0) {
        tuner_by_key_[k] = std::make_shared<tuner_t>(resources_with_offset_[0], resources_with_offset_.size(), resample_time_);
      }
      return k;
    }


  };

}
}
}

#endif //_ONEDPL_AUTO_TUNE_POLICY_H


