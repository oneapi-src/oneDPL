// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SUBMISSION_GROUP_H
#define _ONEDPL_SUBMISSION_GROUP_H

#include "oneapi/dpl/internal/dynamic_selection_impl/concurrent_queue.h"
#include <vector>

namespace oneapi {
namespace dpl {
namespace experimental {
  template<typename T>
  class submission_group_t {
    using submission_container_t = util::concurrent_queue<T>;
    using submissions_t = std::vector<T>;
    submission_container_t submission_container;
    submissions_t submissions;

    public:
    submission_group_t() = default;
    submission_group_t(const submission_group_t&) = delete;
    submission_group_t& operator=(const submission_group_t&) = delete;

    //add to submission group after submit
    void add_submission(T w) { 
      submission_container.push_back(w);	
    }	

    //wait on all submissions in the list
    void wait() {
      while(!submission_container.empty()){
        T w;  
        submission_container.pop(w);
        w->wait();
        delete w;
      }
    }
  };	
} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SUBMISSION_GROUP_H*/
