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

#ifndef __PSTL_execution_policy_defs_H
#define __PSTL_execution_policy_defs_H

#include <type_traits>

namespace pstl {
namespace execution {
inline namespace v1 {

// 2.4, Sequential execution policy
class sequenced_policy {
public:
    // For internal use only
    static constexpr std::false_type __allow_unsequenced() {return std::false_type{};}
    static constexpr std::false_type __allow_vector() {return std::false_type{};}
    static constexpr std::false_type __allow_parallel() {return std::false_type{};}
};

#if __PSTL_USE_PAR_POLICIES
// 2.5, Parallel execution policy
class parallel_policy {
public:
    // For internal use only
    static constexpr std::false_type __allow_unsequenced() {return std::false_type{};}
    static constexpr std::false_type __allow_vector() {return std::false_type{};}
    static constexpr std::true_type __allow_parallel() {return std::true_type{};}
};

// 2.6, Parallel+Vector execution policy
class parallel_unsequenced_policy {
public:
    // For internal use only
    static constexpr std::true_type __allow_unsequenced() {return std::true_type{};}
    static constexpr std::true_type __allow_vector() {return std::true_type{};}
    static constexpr std::true_type __allow_parallel() {return std::true_type{};}
};
#endif

class unsequenced_policy {
public:
    // For internal use only
    static constexpr std::true_type __allow_unsequenced() {return std::true_type{};}
    static constexpr std::true_type __allow_vector() {return std::true_type{};}
    static constexpr std::false_type __allow_parallel() {return std::false_type{};}
};


// 2.8, Execution policy objects
constexpr sequenced_policy seq{};
#if __PSTL_USE_PAR_POLICIES
constexpr parallel_policy par{};
constexpr parallel_unsequenced_policy par_unseq{};
#endif
constexpr unsequenced_policy unseq{};

// 2.3, Execution policy type trait
template<class T> struct is_execution_policy: std::false_type {};

template<> struct is_execution_policy<sequenced_policy     >: std::true_type {};
#if __PSTL_USE_PAR_POLICIES
template<> struct is_execution_policy<parallel_policy       >: std::true_type {};
template<> struct is_execution_policy<parallel_unsequenced_policy>: std::true_type {};
#endif
template<> struct is_execution_policy<unsequenced_policy    >: std::true_type {};

#if __PSTL_CPP14_VARIABLE_TEMPLATES_PRESENT
template<class T> constexpr bool is_execution_policy_v = is_execution_policy<T>::value;
#endif

} // namespace v1
} // namespace execution

namespace internal {
    template<class ExecPolicy, class T> using enable_if_execution_policy = typename std::enable_if<
      pstl::execution::is_execution_policy<typename std::decay<ExecPolicy>::type>::value, T>::type;
} // namespace internal

} // namespace pstl

#endif /* __PSTL_execution_policy_defs_H */
