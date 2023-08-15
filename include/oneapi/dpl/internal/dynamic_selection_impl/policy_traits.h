// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_POLICY_TRAITS_H
#define _ONEDPL_POLICY_TRAITS_H

namespace oneapi {
namespace dpl{
namespace experimental{

    template<typename Policy>
    struct policy_traits{
        //selection_type, resource_type, submission_type, wait_type, submission_group_type
        using selection_t = typename std::decay<Policy>::type::selection_handle_t;  //selection type
        using resource_t = typename std::decay<Policy>::type::execution_resource_t; //resource type

        using wait_t = typename std::decay<Policy>::type::native_sync_t; //wait_type
        using submission_t = typename std::decay<Policy>::type::submission_t; //submission_type
        using submission_group_t = typename std::decay<Policy>::type::submission_group_t; //submission group type
    };
}
}
}
#endif //_ONEDPL_POLICY_TRAITS_H

