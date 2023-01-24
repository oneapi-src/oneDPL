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

#ifndef _ONEDPL_DS_ALGORITHMS_DEFS_H
#define _ONEDPL_DS_ALGORITHMS_DEFS_H
#pragma once

namespace oneapi {
namespace dpl{

//public API for dynamic selection algorithms:
namespace experimental{

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

} //namespace experimental

} //namespace dpl

} //namespace oneapi

#endif /*_ONEDPL_DS_ALGORITHMS_DEFS_H*/
