// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_OPENVINO_BACKEND_IMPL_H
#define _ONEDPL_OPENVINO_BACKEND_IMPL_H

#include <inference_engine.hpp>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

#include <chrono>
#include <vector>
#include <memory>
#include <utility>
#include <iostream>

namespace oneapi {
namespace dpl {
namespace experimental {

  class openvino_backend {
  public:
    using resource_type = ov::CompiledModel;
    using wait_type = ov::InferRequest;
    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<execution_resource_t>;

  private:
    class async_waiter {
      wait_type ir_;
      public:
        async_waiter(wait_type ir) : ir_(ir) {}
        wait_type unwrap() {
	  return ir_; 
	}
        void wait() { 
	  ir_.wait(); 
      	}
    };
    
    class submission_group {
      std::vector<async_waiter> aw_;
    public:
      submission_group(const std::vector<async_waiter>& aw) : aw_(aw) { }

      void wait() {
        for (auto& w : aw_) {
          w.wait(); 
        }
      }
    };

  public:

    openvino_backend(const openvino_backend& v) = delete;
    openvino_backend& operator=(const openvino_backend&) = delete;

    template<typename CompiledModelVector>
    openvino_backend(const CompiledModelVector& compiled_models) {
      global_rank_.reserve(compiled_models.size());
      for (auto i : compiled_models) {
        global_rank_.push_back(i);
      }
    }

    template<typename SelectionHandle, typename Function, typename ...Args>
    auto submit(SelectionHandle s, Function&& f, Args&&... args) {
      auto selected_compiled_model = unwrap(s); 
      std::chrono::high_resolution_clock::time_point t0_(std::chrono::high_resolution_clock::now());
      if constexpr (report_info_v<SelectionHandle, execution_info::task_submission_t>) {
        report(s, execution_info::task_submission);
      }

      //create the infer request
      auto infer_request = selected_compiled_model.create_infer_request();
 
      //set the input to the infer request
      infer_request.set_input_tensor(std::forward<Args>(args)...);;

      //create the callback for reporting once inference is complete
      infer_request.set_callback([&](std::exception_ptr ex) { 
      if (ex) {
          //TODO: Do something or handle exception
      } else {
        //async computation is done at this point
        if constexpr(report_info_v<SelectionHandle, execution_info::task_completion_t>){
          s.report(execution_info::task_completion);
        }
        if constexpr(report_value_v<SelectionHandle, execution_info::task_time_t>) {
          s.report(execution_info::task_time, (std::chrono::high_resolution_clock::now() - t0_).count());
	}
      }
      });

     //start async computation after setting up the callback
     infer_request.start_async();

     waiters_.push_back(async_waiter{infer_request});
     return async_waiter{infer_request};
    }

    auto get_submission_group(){
      sgroup_ptr_ = std::make_unique<submission_group>(waiters_);  //TODO:Empty submission group after returning or after waiting?
      return *sgroup_ptr_;
    }

    auto get_resources() {
      return global_rank_;
    }

  private:
    resource_container_t global_rank_;
    std::unique_ptr<submission_group> sgroup_ptr_;
    std::vector<async_waiter> waiters_;
  };

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_OPENVINO_BACKEND_IMPL_H*/
