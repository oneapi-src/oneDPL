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
        using selection_type = typename std::decay<Policy>::type::selection_type;  //selection type
        using resource_type = typename std::decay<Policy>::type::resource_type; //resource type

        using wait_type = typename std::decay<Policy>::type::wait_type; //wait_type
        //using submission_group_t = typename std::decay<Policy>::type::submission_group_t; //submission group type
    };
}
}
}
#endif //_ONEDPL_POLICY_TRAITS_H

