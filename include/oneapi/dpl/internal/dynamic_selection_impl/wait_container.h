// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_WAIT_CONTAINER_H
#define _ONEDPL_WAIT_CONTAINER_H

#include "oneapi/dpl/internal/dynamic_selection_impl/concurrent_queue.h"
#include <list>

namespace oneapi {
namespace dpl {
namespace experimental {
  template<typename T>
  class waiters_t {
    using waiter_container_t = util::concurrent_queue<T>;
    using waiter_list_t = std::list<T>;
    waiter_list_t waiters;
    waiter_container_t waiter_container;

    public:
    waiters_t() = default;
    waiters_t(const waiters_t&) = delete;
    waiters_t& operator=(const waiters_t&) = delete;

    //add to waiters list after submit
    void add_waiter(T w) { 
      waiter_container.push(w);	
    }	
    //get the list of all waiters
    waiter_list_t get_waiters() {
      waiter_container.pop_all(waiters);
	return waiters;
    }

    //wait on all waiters in the list
    auto wait() {
      while(!waiter_container.empty()){
        T w;  
        waiter_container.pop(w);
        w->wait();
        delete w;
      }
    }

    //get the list of native sync objects
    auto get_native() { 
      std::list<typename T::native_sync_t> native_list;
      waiter_container.pop_all(waiters);
      for (auto w : waiters) {
        native_list.push_back(w->get_native());
        return native_list;
      }
    }
  };	
} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_WAIT_CONTAINER_H*/
