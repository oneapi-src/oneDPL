/* Copyright (c) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *  
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _ONEDPL_SYCL_SUBMITTER_BASE_IMPL_H
#define _ONEDPL_SYCL_SUBMITTER_BASE_IMPL_H

#if _ONEDPL_BACKEND_SYCL

#include <type_traits>      // std::is_same_v, std::decay_t
#include <utility>          // std::forward

namespace oneapi
{
namespace dpl
{
namespace internal
{

////////////////////////////////////////////////////////////////////////////////
// struct __sycl_submitter_base - base class for all sycl submitters
template <typename _ExecutionPolicy>
struct __sycl_submitter_base
{
  protected:

    // We should instantiate this submitter only for cleared _ExecutionPolicy type
    static_assert(std::is_same_v<_ExecutionPolicy, std::decay_t<_ExecutionPolicy>>);

    // Execution policy for using inside submitter implementation
    _ExecutionPolicy __exec;

    template <typename _ExecutionPolicyCtor>
    __sycl_submitter_base(_ExecutionPolicyCtor&& __exec)
        : __exec(std::forward<_ExecutionPolicyCtor>(__exec))
    {
    }

public:

   inline const _ExecutionPolicy&
   get_execution_policy() const
   {
       return __exec;
   }
};

} // namespace internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_SYCL_SUBMITTER_BASE_IMPL_H
