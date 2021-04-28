/*
 *  Copyright (c) Intel Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _ONEDPL_BINARY_FUNCTION_IMPL
#define _ONEDPL_BINARY_FUNCTION_IMPL

namespace oneapi
{
namespace dpl
{

#if __cplusplus > 201402L
template <class _Arg1, class _Arg2, class _Result>
struct binary_function
{
    typedef _Arg1 first_argument_type;
    typedef _Arg2 second_argument_type;
    typedef _Result result_type;
};

template <class _Arg, class _Result>
struct unary_function
{
    typedef _Arg argument_type;
    typedef _Result result_type;
};
#endif

} // end namespace dpl
} // end namespace oneapi

#endif
