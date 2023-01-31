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

namespace ds {

  template<typename NativeContext>
  struct nop_execution_resource_t {
    using native_resource_t = NativeContext;
    native_resource_t native_resource_;
    nop_execution_resource_t() : native_resource_(native_resource_t{}) {}
    nop_execution_resource_t(native_resource_t nc) : native_resource_(nc) {}
    native_resource_t get_native() const { return native_resource_; }
    bool operator==(const nop_execution_resource_t& e) const {
      return native_resource_ == e.native_resource_;
    }
    bool operator==(const native_resource_t& e) const {
      return native_resource_ == e;
    }
  };

}

