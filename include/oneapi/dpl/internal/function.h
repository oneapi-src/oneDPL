// -*- C++ -*-
//===-- function.h ---------------------------------------------------------===//
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

#ifndef _ONEDPL_INTERNAL_FUNCTION_H
#define _ONEDPL_INTERNAL_FUNCTION_H

#include <utility>
#if _ONEDPL_BACKEND_SYCL
#    include "../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#endif
#include "../functional"
#include <tuple>

namespace oneapi
{
namespace dpl
{
namespace internal
{
using ::std::get;

// Helper used to eliminate compile errors when an algorithm needs to pass a policy instance to
// multiple Parallel STL functions.
template <typename Policy, typename NewName>
struct rebind_policy
{
    using type = Policy;
};

#if _ONEDPL_BACKEND_SYCL
template <typename KernelName, typename NewName>
struct rebind_policy<oneapi::dpl::execution::device_policy<KernelName>, NewName>
{
    using type = oneapi::dpl::execution::device_policy<NewName>;
};

#    if _ONEDPL_FPGA_DEVICE
template <unsigned int factor, typename KernelName, typename NewName>
struct rebind_policy<oneapi::dpl::execution::fpga_policy<factor, KernelName>, NewName>
{
    using type = oneapi::dpl::execution::fpga_policy<factor, NewName>;
};
#    endif

using oneapi::dpl::__internal::is_hetero_iterator;
using oneapi::dpl::__par_backend_hetero::__internal::__buffer;
#endif

#if _ONEDPL_BACKEND_SYCL
// Helpers used to get indexable access to the data passed to the SYCL implementation of an
// algorithm from either a SYCL iterator or a USM pointer.
template <sycl::access::mode Mode, typename Policy, typename Iterator>
auto
get_access(Policy, Iterator i, typename ::std::enable_if<is_hetero_iterator<Iterator>::value, void>::type* = nullptr)
    -> decltype(i.get_buffer().template get_access<Mode>())
{
    return i.get_buffer().template get_access<Mode>();
}

template <sycl::access::mode Mode, typename Policy, typename Iterator>
Iterator
get_access(Policy, Iterator i, typename ::std::enable_if<!is_hetero_iterator<Iterator>::value, void>::type* = nullptr)
{
    return i;
}

template <sycl::access::mode Mode, typename Policy, typename T>
counting_iterator<T>
get_access(Policy, counting_iterator<T> i)
{
    return i;
}

template <sycl::access::mode Mode, typename Policy, typename T>
T*
get_access(const Policy& policy, T* ptr)
{
    assert(sycl::get_pointer_type(ptr, policy.queue().get_context()) == sycl::usm::alloc::shared);
    return ptr;
}
#endif

// struct for checking if iterator is a discard_iterator or not
template <typename Iter, typename Void = void> // for non-discard iterators
struct is_discard_iterator : ::std::false_type
{
};

template <typename Iter> // for discard iterators
struct is_discard_iterator<Iter, typename ::std::enable_if<Iter::is_discard::value, void>::type> : ::std::true_type
{
};

// Used by: exclusive_scan_by_key
// Lambda: [pred, &new_value](Ref1 a, Ref2 s) {return pred(s) ? new_value : a; });
template <typename T, typename Predicate>
struct replace_if_fun
{
    using result_of = T;

    replace_if_fun(Predicate _pred, T _new_value) : pred(_pred), new_value(_new_value) {}

    template <typename _T1, typename _T2>
    T
    operator()(_T1&& a, _T2&& s) const
    {
        return pred(s) ? new_value : a;
    }

  private:
    Predicate pred;
    const T new_value;
};

// Used by: exclusive_scan_by_key
template <typename ValueType, typename FlagType, typename BinaryOp>
struct scan_by_key_fun
{
    using result_of = ::std::tuple<ValueType, FlagType>;

    scan_by_key_fun(BinaryOp input) : binary_op(input) {}

    template <typename _T1, typename _T2>
    result_of
    operator()(_T1&& x, _T2&& y) const
    {
        using ::std::get;
        return ::std::make_tuple(get<1>(y) ? get<0>(y) : binary_op(get<0>(x), get<0>(y)), get<1>(x) | get<1>(y));
    }

  private:
    BinaryOp binary_op;
};

// Used by: reduce_by_key
template <typename ValueType, typename FlagType, typename BinaryOp>
struct segmented_scan_fun
{
    segmented_scan_fun(BinaryOp input) : binary_op(input) {}

    template <typename _T1, typename _T2>
    _T1
    operator()(const _T1& x, const _T2& y) const
    {
        using ::std::get;
        auto new_x = get<1>(y) ? get<0>(y) : binary_op(get<0>(x), get<0>(y));
        auto new_y = get<1>(x) | get<1>(y);
        return _T1(new_x, new_y);
    }

  private:
    BinaryOp binary_op;
};

// Used by: reduce_by_key on host
template <typename Output1, typename Output2>
class scatter_and_accumulate_fun
{
  public:
    scatter_and_accumulate_fun(Output1 _result1, Output2 _result2) : result1(_result1), result2(_result2) {}

    template <typename _T>
    void
    operator()(_T&& x) const
    {
        using ::std::get;
        if (::std::get<2>(x))
        {
            result1[::std::get<1>(x)] = ::std::get<0>(x);
        }
        if (::std::get<4>(x))
        {
            result2[::std::get<1>(x)] = ::std::get<3>(x);
        }
    }

  private:
    Output1 result1;
    Output2 result2;
};

// Used by: reduce_by_key, mapping rules for scatter_if and gather_if
template <typename T, typename Predicate, typename UnaryOperation = identity>
class transform_if_stencil_fun
{
  public:
    using result_of = T;

    transform_if_stencil_fun(Predicate _pred, UnaryOperation _op = identity()) : pred(_pred), op(_op) {}

    template <typename _T>
    void
    operator()(_T&& t) const
    {
        using ::std::get;
        if (pred(get<1>(t)))
            get<2>(t) = op(get<0>(t));
    }

  private:
    Predicate pred;
    UnaryOperation op;
};
} // namespace internal
} // namespace dpl
} // namespace oneapi
#endif
