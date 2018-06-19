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

#ifndef __PSTL_execution_impl_H
#define __PSTL_execution_impl_H

#include <iterator>
#include <type_traits>

#include "execution_defs.h"

namespace pstl {
namespace internal {

using namespace pstl::execution;

/* predicate */

template<typename T>
    std::false_type lazy_and( T, std::false_type ) { return std::false_type{}; };

template<typename T>
    inline T lazy_and( T a, std::true_type ) { return a; }

template<typename T>
    std::true_type lazy_or( T, std::true_type ) { return std::true_type{}; };

template<typename T>
    inline T lazy_or( T a, std::false_type ) { return a; }

/* iterator */
template<typename iterator_type, typename... other_iterator_types>
struct is_random_access_iterator {
    static constexpr bool value =
        is_random_access_iterator<iterator_type>::value &&
        is_random_access_iterator<other_iterator_types...>::value;
    typedef std::integral_constant<bool, value> type;
};

template<typename iterator_type>
struct is_random_access_iterator<iterator_type>
    : std::is_same<typename std::iterator_traits<iterator_type>::iterator_category,
                   std::random_access_iterator_tag> {
};

/* policy */
template<typename Policy>
struct policy_traits {};

template <>
struct policy_traits<sequenced_policy> {
    typedef std::false_type allow_parallel;
    typedef std::false_type allow_unsequenced;
    typedef std::false_type allow_vector;
};

template <>
struct policy_traits<unsequenced_policy> {
    typedef std::false_type allow_parallel;
    typedef std::true_type  allow_unsequenced;
    typedef std::true_type  allow_vector;
};


#if __PSTL_USE_PAR_POLICIES
template <>
struct policy_traits<parallel_policy> {
    typedef std::true_type  allow_parallel;
    typedef std::false_type allow_unsequenced;
    typedef std::false_type allow_vector;
};

template <>
struct policy_traits<parallel_unsequenced_policy> {
    typedef std::true_type allow_parallel;
    typedef std::true_type allow_unsequenced;
    typedef std::true_type allow_vector;
};
#endif

template<typename ExecutionPolicy> using collector_t =
    typename policy_traits<typename std::decay<ExecutionPolicy>::type>::collector_type;

template<typename ExecutionPolicy> using allow_vector =
    typename internal::policy_traits<typename std::decay<ExecutionPolicy>::type>::allow_vector;

template<typename ExecutionPolicy> using allow_unsequenced =
    typename internal::policy_traits<typename std::decay<ExecutionPolicy>::type>::allow_unsequenced;

template<typename ExecutionPolicy> using allow_parallel =
    typename internal::policy_traits<typename std::decay<ExecutionPolicy>::type>::allow_parallel;


template<typename ExecutionPolicy, typename... iterator_types>
auto is_vectorization_preferred(ExecutionPolicy&& exec) ->
decltype(lazy_and( exec.__allow_vector(), typename is_random_access_iterator<iterator_types...>::type()))
{
    return lazy_and( exec.__allow_vector(), typename is_random_access_iterator<iterator_types...>::type() );
}

template<typename ExecutionPolicy, typename... iterator_types>
auto is_parallelization_preferred(ExecutionPolicy&& exec) ->
decltype(lazy_and( exec.__allow_parallel(), typename is_random_access_iterator<iterator_types...>::type()))
{
    return lazy_and( exec.__allow_parallel(), typename is_random_access_iterator<iterator_types...>::type() );
}

template<typename policy, typename... iterator_types>
struct prefer_unsequenced_tag {
    static constexpr bool value =
        allow_unsequenced<policy>::value && is_random_access_iterator<iterator_types...>::value;
    typedef std::integral_constant<bool, value> type;
};

template<typename policy, typename... iterator_types>
struct prefer_parallel_tag {
    static constexpr bool value =
        allow_parallel<policy>::value && is_random_access_iterator<iterator_types...>::value;
    typedef std::integral_constant<bool, value> type;
};

} // namespace internal
} // namespace pstl

#endif /* __PSTL_execution_impl_H */
