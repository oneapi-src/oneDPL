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
    using submission_container_t = std::shared_ptr<util::concurrent_queue<T>>;
    using submissions_t = std::vector<T>;
    submission_container_t submission_container;
    submissions_t submissions;

    public:
    submission_group_t() : submission_container(std::make_shared<util::concurrent_queue<T>>()) {};

    //add to submission group after submit
    void add_submission(T w) {
      submission_container->push(w);
    }

    //wait on all submissions in the list
    void wait() {
      while(!submission_container->empty()){
        T w;
        if(submission_container->pop_if_present(w)){
            w->wait();
            delete w;
        } else {
            throw std::runtime_error("Submission container empty\n");
        }
      }
    }
  };
} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SUBMISSION_GROUP_H*/
