// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef _ONEDPL_INTERNAL_DYNAMIC_SELECTION_H
#define _ONEDPL_INTERNAL_DYNAMIC_SELECTION_H

#include <memory>
#include <utility>
#include <list>
#include "oneapi/dpl/internal/dynamic_selection_impl/policy_traits.h"

namespace oneapi {
namespace dpl {
namespace experimental {
//ds_properties

  namespace execution_info {
    struct execution_time_t {
      static constexpr bool is_execution_info_v = true;
      using value_type = uint64_t;
    };
    inline constexpr execution_time_t execution_time;
  }

  namespace property {
    struct task_completion_t {
      static constexpr bool is_property_v = true;
      static constexpr bool can_report_v = true;
    };
    inline constexpr task_completion_t task_completion;

    template<typename T, typename Property>
    auto query(T&& t, const Property& prop) {
      return std::forward<T>(t).query(prop);
    }

    template<typename T, typename Property, typename Argument>
    auto query(T&& t, const Property& prop, const Argument& arg) {
      return std::forward<T>(t).query(prop, arg);
    }

    template<typename Handle, typename Property>
    auto report(Handle&& h, const Property& prop) {
      return std::forward<Handle>(h).report(prop);
    }

    template<typename Handle, typename Property, typename ValueType>
    auto report(Handle&& h, const Property& prop, const ValueType& v) {
      return std::forward<Handle>(h).report(prop, v);
    }
  } //namespace property

//ds_algorithms
  template<typename DSPolicy>
  auto get_universe(DSPolicy&& dp) {
    return std::forward<DSPolicy>(dp).get_universe();
  }

  template<typename DSPolicy>
  auto get_universe_size(DSPolicy&& dp) {
    return std::forward<DSPolicy>(dp).get_universe_size();
  }

  template<typename DSPolicy, typename ...Args>
  auto set_universe(DSPolicy&& dp, Args&&... args) {
    return std::forward<DSPolicy>(dp).get_universe(std::forward<Args>(args)...);
  }

  template<typename Handle>
  auto wait(Handle&& h) {
    return std::forward<Handle>(h).wait();
  }

  template<typename Handle>
  auto wait(std::list<Handle> l) {
      for(auto h : l){
        return h->wait();
      }
  }

  template<typename DSPolicy>
  auto  get_wait_list(DSPolicy&& dp){
    return std::forward<DSPolicy>(dp).get_wait_list();
  }

  template<typename DSPolicy, typename... Args>
  typename policy_traits<DSPolicy>::selection_type select(DSPolicy&& dp, Args&&... args) {
    return std::forward<DSPolicy>(dp).select(std::forward<Args>(args)...);
  }

  template<typename DSPolicy, typename Function, typename... Args>
  auto invoke_async(DSPolicy&& dp, typename policy_traits<DSPolicy>::selection_type e, Function&&f, Args&&... args) {
    return std::forward<DSPolicy>(dp).invoke_async(e, std::forward<Function>(f), std::forward<Args>(args)...);
  }

  template<typename DSPolicy, typename Function, typename... Args>
  auto has_invoke_async_impl(...) -> std::false_type;

  template<typename DSPolicy, typename Function, typename... Args>
  auto has_invoke_async_impl(int) -> decltype(std::declval<DSPolicy>().invoke_async(std::declval<Function>(), std::declval<Args>()...), std::true_type{});

  template<typename DSPolicy, typename Function, typename... Args>
  struct has_invoke_async : decltype(has_invoke_async_impl<DSPolicy, Function, Args...>(0)) {};

  template<typename DSPolicy, typename Function, typename... Args>
  auto invoke_async(DSPolicy&& dp, Function&&f, Args&&... args) {
    if constexpr(has_invoke_async<DSPolicy, Function, Args...>::value == true) {
        return std::forward<DSPolicy>(dp).invoke_async(std::forward<Function>(f), std::forward<Args>(args)...);
    }
    else {
        return invoke_async(std::forward<DSPolicy>(dp), std::forward<DSPolicy>(dp).select(f, args...), std::forward<Function>(f), std::forward<Args>(args)...);
    }
  }


  template<typename DSPolicy, typename Function, typename... Args>
  auto has_invoke_impl(...) -> std::false_type;

  template<typename DSPolicy, typename Function, typename... Args>
  auto has_invoke_impl(int) -> decltype(std::declval<DSPolicy>().invoke(std::declval<Function>(), std::declval<Args>()...), std::true_type{});

  template<typename DSPolicy, typename Function, typename... Args>
  struct has_invoke : decltype(has_invoke_impl<DSPolicy, Function, Args...>(0)) {};

  template<typename DSPolicy, typename Function, typename... Args>
  auto invoke(DSPolicy&& dp, Function&&f, Args&&... args) {
    if constexpr(has_invoke<DSPolicy, Function, Args...>::value == true) {
        return std::forward<DSPolicy>(dp).invoke(std::forward<Function>(f), std::forward<Args>(args)...);
    }
    else{
        return wait(std::forward<DSPolicy>(dp).invoke_async(std::forward<DSPolicy>(dp).select(f, args...), std::forward<Function>(f), std::forward<Args>(args)...));
    }
  }


  template<typename DSPolicy, typename SelectionHandle,  typename Function, typename... Args>
  auto has_invoke_handle_impl(...) -> std::false_type;

  template<typename DSPolicy, typename SelectionHandle,  typename Function, typename... Args>
  auto has_invoke_handle_impl(int) -> decltype(std::declval<DSPolicy>().invoke(std::declval<SelectionHandle>(), std::declval<Function>(), std::declval<Args>()...), std::true_type{});

  template<typename DSPolicy, typename SelectionHandle, typename Function, typename... Args>
  struct has_invoke_handle : decltype(has_invoke_handle_impl<DSPolicy, SelectionHandle , Function, Args...>(0)) {};

  template<typename DSPolicy, typename Function, typename... Args>
  auto invoke(DSPolicy&& dp, typename policy_traits<DSPolicy>::selection_type e, Function&&f, Args&&... args) {
    if constexpr(has_invoke_handle<DSPolicy, typename policy_traits<DSPolicy>::selection_type, Function, Args...>::value == true) {
        return std::forward<DSPolicy>(dp).invoke(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }else{
        return wait(invoke_async(std::forward<DSPolicy>(dp), e, std::forward<Function>(f), std::forward<Args>(args)...));
    }
  }

  template<typename T>
  auto has_unwrap_impl(...) -> std::false_type;

  template<typename T>
  auto has_unwrap_impl(int) -> decltype(std::declval<T>().unwrap(), std::true_type{});

  template<typename T>
  struct has_unwrap : decltype(has_unwrap_impl<T>(0)) {};

  template<typename T>
  auto unwrap(T&& v) {
    if constexpr(has_unwrap<T>::value == true) {
        return std::forward<T>(v).unwrap();
    } else {
        return v;
    }
  }

  template<typename S, typename Info>
  auto has_report_impl(...) -> std::false_type;

  template<typename S, typename Info>
  auto has_report_impl(int) -> decltype(std::declval<S>().report(std::declval<Info>()), std::true_type{});

  template<typename S, typename Info>
  struct has_report : decltype(has_report_impl<S,Info>(0)) {};

  template<typename S, typename Info>
  void report(S&& s, const Info& i) {
    if constexpr(has_report<S,Info>::value == true) {
      std::forward<S>(s).report(i);
    } 
  }

  template<typename S, typename Info>
  auto has_report_value_impl(...) -> std::false_type;

  template<typename S, typename Info>
  auto has_report_value_impl(int) -> decltype(std::declval<S>().report(std::declval<Info>(), std::declval<typename Info::value_type>()), std::true_type{});

  template<typename S, typename Info>
  struct has_report_value : decltype(has_report_value_impl<S,Info>(0)) {};

  template<typename S, typename Info, typename Value>
  void report(S&& s, const Info& i, const Value& v) {
    if constexpr(has_report_value<S,Info>::value == true) {
      std::forward<S>(s).report(i, v);
    } 
  }

  template<typename S, typename Info>
  struct report_execution_info {
    static constexpr bool value = std::disjunction_v<has_report<S,Info>, has_report_value<S,Info>>;   
  };
  template<typename S, typename Info>
  inline constexpr bool report_execution_info_v = report_execution_info<S,Info>::value; 

} // namespace experimental
} // namespace dpl
} // namespace oneapi
#if _DS_BACKEND_SYCL != 0
#include "oneapi/dpl/internal/dynamic_selection_impl/sycl_scheduler.h"
#endif
#include "oneapi/dpl/internal/dynamic_selection_impl/static_policy_impl.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/round_robin_policy_impl.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/auto_tune_policy.h"

#endif /*_ONEDPL_INTERNAL_DYNAMIC_SELECTION_H*/
