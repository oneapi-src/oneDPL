// -*- C++ -*-
//===-- function.h ---------------------------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#ifndef DPCPP_INTERNAL_FUNCTION_H_
#define DPCPP_INTERNAL_FUNCTION_H_

#include <utility>
#if _PSTL_BACKEND_SYCL
#include <dpstd/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>
#endif
#include <dpstd/functional>
#include <tuple>

namespace dpstd {

namespace internal {

    using std::get;

    // Helper used to eliminate compile errors when an algorithm needs to pass a policy instance to
    // multiple parallel STL functions.
    template <typename Policy, typename NewName>
    struct rebind_policy { using type = Policy; };

#if _PSTL_BACKEND_SYCL
    template <typename DevicePolicy, typename KernelName, typename NewName>
    struct rebind_policy<dpstd::execution::device_policy<DevicePolicy, KernelName>, NewName>
    { using type = dpstd::execution::device_policy<DevicePolicy, NewName>; };

    using dpstd::__par_backend_hetero::__internal::get;
    using dpstd::__par_backend_hetero::__internal::__buffer;
#else
    using dpstd::__par_backend::__buffer;
#endif

#if _PSTL_BACKEND_SYCL
    // Helpers used to get indexable access to the data passed to the SYCL implementation of an
    // algorithm from either a SYCL iterator or a USM pointer.
    template<cl::sycl::access::mode Mode, typename Iterator>
    auto get_access(Iterator i)
        -> decltype(i.get_buffer().template get_access<Mode>())
    { return i.get_buffer().template get_access<Mode>(); }

    template<cl::sycl::access::mode Mode, typename T>
    counting_iterator<T> get_access(counting_iterator<T> i)
    { return i; }

    template<cl::sycl::access::mode Mode, typename T>
    T* get_access(T* ptr)
    { return ptr; }
#endif

    // Used by: exclusive_scan_by_key
    // Lambda: [pred, &new_value](Ref1 a, Ref2 s) {return pred(s) ? new_value : a; });
    template<typename T, typename Predicate>
    struct replace_if_fun {
        using result_of = T;

        replace_if_fun(Predicate _pred, T _new_value)
            : pred(_pred), new_value(_new_value)
        {}

        template<typename _T1, typename _T2>
        T operator() (_T1&& a, _T2&& s) const
        {
            return pred(s) ? new_value : a;
        }
    private:
        Predicate pred;
        const T new_value;
    };

    // Used by: exclusive_scan_by_key
    template<typename ValueType, typename FlagType, typename BinaryOp>
    struct scan_by_key_fun {
        using result_of = std::tuple<ValueType,FlagType>;

        scan_by_key_fun(BinaryOp input)
          : binary_op(input)
        {}

        template<typename _T1, typename _T2>
        result_of
        operator() (_T1&& x, _T2&& y) const
        {
            return std::make_tuple(
                get<1>(y) ? get<0>(y) :
                            binary_op(get<0>(x), get<0>(y)),
                get<1>(x) | get<1>(y));
        }
    private:
        BinaryOp binary_op;
    };

    // Used by: reduce_by_key
    template<typename ValueType, typename FlagType, typename BinaryOp>
    struct segmented_scan_fun {
        segmented_scan_fun(BinaryOp input)
            : binary_op(input)
        {}

        template<typename _T1, typename _T2>
        auto operator() (_T1&& x, _T2&& y) const
            -> decltype(std::make_tuple(get<0>(y), get<1>(y)))
        {
            return std::make_tuple(get<1>(y) ? get<0>(y) :
                                  binary_op(get<0>(x), get<0>(y)),
                                  get<1>(x) | get<1>(y) );
        }
    private:
        BinaryOp binary_op;
    };

    // Used by: reduce_by_key on host
    template<typename Output1, typename Output2>
    class scatter_and_accumulate_fun {
    public:
        scatter_and_accumulate_fun(Output1 _result1, Output2 _result2)
            : result1(_result1), result2(_result2)
        {}

        template<typename _T>
        void operator() (_T&& x) const
        {
            if (std::get<2>(x)) {
                result1[std::get<1>(x)] = std::get<0>(x);
            }
            if (std::get<4>(x)) {
                result2[std::get<1>(x)] = std::get<3>(x);
            }
        }

    private:
        Output1 result1;
        Output2 result2;
    };

    // Used by: reduce_by_key, mapping rules for scatter_if and gather_if
    template<typename T, typename Predicate, typename UnaryOperation = identity>
    class transform_if_stencil_fun {
    public:
        using result_of = T;

        transform_if_stencil_fun(Predicate _pred, UnaryOperation _op = identity())
            : pred(_pred), op(_op)
        {}

        template<typename _T>
        void operator() (_T&& t)
        {
            if (pred(get<1>(t)))
              get<2>(t) = op(get<0>(t));
        }

    private:
        Predicate pred;
        UnaryOperation op;
    };
} // namespace internal

} // namespace dpstd
#endif
