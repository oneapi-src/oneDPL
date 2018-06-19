/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef __PSTL_glue_execution_defs_H
#define __PSTL_glue_execution_defs_H

#include <type_traits>

#include "execution_defs.h"

namespace std {
    // Type trait
    using pstl::execution::is_execution_policy;
#if __PSTL_CPP14_VARIABLE_TEMPLATES_PRESENT
#if __INTEL_COMPILER
    template<class T> constexpr bool is_execution_policy_v = is_execution_policy<T>::value;
#else
    using pstl::execution::is_execution_policy_v;
#endif
#endif

    namespace execution {
        // Standard C++ policy classes
        using pstl::execution::sequenced_policy;
#if __PSTL_USE_PAR_POLICIES
        using pstl::execution::parallel_policy;
        using pstl::execution::parallel_unsequenced_policy;
#endif
        // Standard predefined policy instances
        using pstl::execution::seq;
#if __PSTL_USE_PAR_POLICIES
        using pstl::execution::par;
        using pstl::execution::par_unseq;
#endif
        // Implementation-defined names
        // Unsequenced policy is not yet standard, but for consistency
        // we include it into namespace std::execution as well
        using pstl::execution::unseq;
        using pstl::execution::unsequenced_policy;
    } // namespace execution
} // namespace std

#endif /* __PSTL_glue_execution_defs_H */
