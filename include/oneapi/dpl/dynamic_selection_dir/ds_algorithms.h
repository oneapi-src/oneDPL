// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//


#ifndef _ONEDPL_DS_ALGORITHMS_DEFS_H
#define _ONEDPL_DS_ALGORITHMS_DEFS_H
#pragma once


namespace oneapi {
namespace dpl {
namespace experimental {

  template<typename Handle>
  inline auto wait_for_all(Handle&& h) {
    return std::forward<Handle>(h).wait_for_all();
  }

  template<typename DSPolicy, typename... Args>
  inline auto select(DSPolicy&& dp, Args&&... args) {
    return std::forward<DSPolicy>(dp).select(std::forward<Args>(args)...);
  }

  template<typename DSPolicy, typename Function, typename... Args>
  inline auto invoke_async(DSPolicy&& dp, Function&&f, Args&&... args) {
    return std::forward<DSPolicy>(dp).invoke_async(std::forward<Function>(f), std::forward<Args>(args)...);
  }

  template<typename DSPolicy, typename Function, typename... Args>
  inline auto invoke(DSPolicy&& dp, Function&&f, Args&&... args) {
    return wait_for_all(invoke_async(std::forward<DSPolicy>(dp), std::forward<Function>(f), std::forward<Args>(args)...));
  }

  template<typename DSPolicy, typename Function, typename... Args>
  inline auto invoke_async(DSPolicy&& dp, typename DSPolicy::selection_handle_t e, Function&&f, Args&&... args) {
    return std::forward<DSPolicy>(dp).invoke_async(e, std::forward<Function>(f), std::forward<Args>(args)...);
  }

  template<typename DSPolicy, typename Function, typename... Args>
  inline auto invoke(DSPolicy&& dp, typename DSPolicy::selection_handle_t e, Function&&f, Args&&... args) {
    return wait_for_all(invoke_async(std::forward<DSPolicy>(dp), e, std::forward<Function>(f), std::forward<Args>(args)...));
  }

} // namespace experimental
} // namespace dpl
} // namespace oneapi
#endif /*_ONEDPL_DS_ALGORITHMS_DEFS_H*/
