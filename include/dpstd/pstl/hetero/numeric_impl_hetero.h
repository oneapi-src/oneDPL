// -*- C++ -*-
//===-- numeric_impl_hetero.h ---------------------------------------------===//
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

#ifndef _PSTL_numeric_impl_hetero_H
#define _PSTL_numeric_impl_hetero_H

#include <iterator>
#include "../parallel_backend.h"
#if _PSTL_BACKEND_SYCL
#    include "dpcpp/execution_sycl_defs.h"
#    include "algorithm_impl_hetero.h" // to use __pattern_walk2_brick
#    include "dpcpp/parallel_backend_sycl_utils.h"
#    include "dpcpp/unseq_backend_sycl.h"
#endif

namespace dpstd
{
namespace __internal
{

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Tp,
          typename _BinaryOperation1, typename _BinaryOperation2>
dpstd::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                           _RandomAccessIterator2 __first2, _Tp __init, _BinaryOperation1 __binary_op1,
                           _BinaryOperation2 __binary_op2, /*vector=*/std::true_type, /*parallel=*/std::true_type)
{
    if (__first1 == __last1)
        return __init;

    using namespace __par_backend_hetero;
    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk2<_Policy, _BinaryOperation2>;
    using _RepackedTp = __repacked_tuple_t<_Tp>;

    _RepackedTp __res = __parallel_transform_reduce<_RepackedTp>(
        std::forward<_ExecutionPolicy>(__exec), zip(make_iter_mode<read>(__first1), make_iter_mode<read>(__first2)),
        zip(make_iter_mode<read>(__last1), make_iter_mode<read>(/*last2=*/__first2 + (__last1 - __first1))),
        unseq_backend::transform_init<_Policy, _BinaryOperation1, _Functor>{__binary_op1,
                                                                            _Functor{__binary_op2}}, // transform
        __binary_op1,                                                                                // combine
        unseq_backend::reduce<_Policy, _BinaryOperation1, _RepackedTp>{__binary_op1}                 // reduce
    );

    return __binary_op1(__init, _Tp{__res});
}

//------------------------------------------------------------------------
// transform_reduce (with unary and binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Tp, typename _BinaryOperation,
          typename _UnaryOperation>
dpstd::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
                           _BinaryOperation __binary_op, _UnaryOperation __unary_op, /*vector=*/std::true_type,
                           /*parallel=*/std::true_type)
{
    if (__first == __last)
        return __init;

    using namespace __par_backend_hetero;
    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk1<_Policy, _UnaryOperation>;
    using _RepackedTp = __repacked_tuple_t<_Tp>;

    _RepackedTp __res = __parallel_transform_reduce<_RepackedTp>(
        std::forward<_ExecutionPolicy>(__exec), make_iter_mode<read>(__first), make_iter_mode<read>(__last),
        unseq_backend::transform_init<_Policy, _BinaryOperation, _Functor>{__binary_op,
                                                                           _Functor{__unary_op}}, // transform
        __binary_op,                                                                              // combine
        unseq_backend::reduce<_Policy, _BinaryOperation, _RepackedTp>{__binary_op}                // reduce
    );

    return __binary_op(__init, _Tp{__res});
}

//------------------------------------------------------------------------
// transform_scan
//------------------------------------------------------------------------

struct copy_functor
{
    template <typename _Value, typename _Idx, typename _InAcc, typename _OutAcc>
    void
    operator()(const _Value& __value, const _Value&, const _Idx __global_idx, const _InAcc&, const _OutAcc& __output)
    {
        __output[__global_idx] = __value;
    }
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation, typename _Tp,
          typename _BinaryOperation, typename _Inclusive>
dpstd::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator2>
__pattern_transform_scan(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                         _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op, _Inclusive,
                         /*vector=*/std::true_type, /*parallel=*/std::true_type)
{
    using namespace __par_backend_hetero;
    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk1<_Policy, _UnaryOperation>;
    using _RepackedTp = __repacked_tuple_t<_Tp>;

    return __parallel_transform_scan(
               std::forward<_ExecutionPolicy>(__exec), make_iter_mode<read>(__first), make_iter_mode<read>(__last),
               make_iter_mode<write>(__result), __binary_op, _RepackedTp{__init},
               unseq_backend::transform_init<_Policy, _BinaryOperation, _Functor>{__binary_op,
                                                                                  _Functor{__unary_op}}, // transform
               unseq_backend::reduce<_Policy, _BinaryOperation, _RepackedTp>{__binary_op},               // reduce
               unseq_backend::scan<_Inclusive, _Policy, _BinaryOperation, _Functor, copy_functor, _RepackedTp>{
                   __binary_op, _Functor{__unary_op}, copy_functor{}} // scan
               )
        .first;
}

template <typename _KernelName>
struct __reduce_for_scan
{
};
template <typename _KernelName>
struct __fill_for_scan
{
};
template <typename _KernelName>
struct __scan_without_init
{
};
// transform_scan without initial element
// TODO: replace the implementation below with the __parallel_transform_scan function without initial element
template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation,
          typename _BinaryOperation, typename _Inclusive>
dpstd::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator2>
__pattern_transform_scan(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                         _UnaryOperation __unary_op, _BinaryOperation __binary_op, _Inclusive,
                         /*vector=*/std::true_type, /*parallel=*/std::true_type)
{
    typedef typename std::iterator_traits<_Iterator1>::value_type _Tp;
    if (__first != __last)
    {
        using _Functor = unseq_backend::walk1<_ExecutionPolicy, _UnaryOperation>;
        using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;

        // The code below is the analogue of the __tmp = __unary_op(*__first) operation
        // It's needed to workaround the issue with extra copying data from host to device
        // for the version of transform_scan without initial element
        _RepackedTp __tmp = __par_backend_hetero::__parallel_transform_reduce<_RepackedTp>(
            __par_backend_hetero::make_wrapped_policy<__reduce_for_scan>(std::forward<_ExecutionPolicy>(__exec)),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::read>(__first),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::read>(__first + 1),
            unseq_backend::transform_init<_ExecutionPolicy, _BinaryOperation, _Functor>{
                __binary_op, _Functor{__unary_op}},                                             // transform
            __binary_op,                                                                        // combine
            unseq_backend::reduce<_ExecutionPolicy, _BinaryOperation, _RepackedTp>{__binary_op} // reduce
        );
        // The code below is the analogue of the *__result = __tmp operation
        // It's needed to workaround the issue with extra copying data from host to device
        // for the version of transform_scan without initial element
        __pattern_fill(
            __par_backend_hetero::make_wrapped_policy<__fill_for_scan>(std::forward<_ExecutionPolicy>(__exec)),
            __result, __result + 1, __tmp, std::true_type(), std::true_type());
        // transform scan with __tmp as an initial element

        // TODO:
        //   Since we have multiple kernels, we have to explicitly syncronize
        //   if we use USM pointers which do not have implicit sync mechanism
        //   Normally, it ought to be done with event dependency, but our interface
        //   at that level repeats CPU interface and, hence, is not that extensible
        __par_backend_hetero::explicit_wait_if<std::is_pointer<_Iterator2>::value>{}(__exec);

        return __par_backend_hetero::__parallel_transform_scan(
                   __par_backend_hetero::make_wrapped_policy<__scan_without_init>(
                       std::forward<_ExecutionPolicy>(__exec)),
                   __par_backend_hetero::make_iter_mode<__par_backend_hetero::read>(__first + 1),
                   __par_backend_hetero::make_iter_mode<__par_backend_hetero::read>(__last),
                   __par_backend_hetero::make_iter_mode<__par_backend_hetero::write>(__result + 1), __binary_op, __tmp,
                   unseq_backend::transform_init<_ExecutionPolicy, _BinaryOperation, _Functor>{
                       __binary_op, _Functor{__unary_op}},                                              // transform
                   unseq_backend::reduce<_ExecutionPolicy, _BinaryOperation, _RepackedTp>{__binary_op}, // reduce
                   unseq_backend::scan<_Inclusive, _ExecutionPolicy, _BinaryOperation, _Functor, copy_functor,
                                       _RepackedTp>{__binary_op, _Functor{__unary_op}, copy_functor{}} // scan
                   )
            .first;
    }
    else
    {
        return __result;
    }
}

//------------------------------------------------------------------------
// adjacent_difference
//------------------------------------------------------------------------

// a wrapper for the policy is required to avoid the kernel naming issue
template <typename Name>
struct adjacent_difference_wrapper
{
};

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _BinaryOperation>
dpstd::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator2>
__pattern_adjacent_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                              _ForwardIterator2 __d_first, _BinaryOperation __op, /*vector*/ std::true_type,
                              /*parallel*/ std::true_type)
{
    using _It1ValueT = typename std::iterator_traits<_ForwardIterator1>::value_type;
    using _It2ValueTRef = typename std::iterator_traits<_ForwardIterator2>::reference;

    _ForwardIterator2 __d_last = __d_first + (__last - __first);

    if (__first == __last)
    {
        return __d_first;
    }
#if !__SYCL_UNNAMED_LAMBDA__
    // if we have the only element, just copy it according to the specification
    else if (__last - __first == 1)
    {
        auto __wrapped_policy = __par_backend_hetero::make_wrapped_policy<adjacent_difference_wrapper>(
            std::forward<_ExecutionPolicy>(__exec));

        __internal::__pattern_walk2_brick(__wrapped_policy, __first, __last, __d_first,
                                          __internal::__brick_copy<decltype(__wrapped_policy)>{}, std::true_type{});

        return __d_last;
    }
#endif
    else
    {
        using namespace __par_backend_hetero;

        auto __fn = [__op](_It1ValueT __in1, _It1ValueT __in2, _It2ValueTRef __out1) mutable {
            __out1 = __op(__in2, __in1); // This move assignment is allowed by the C++ standard draft N4810
        };

        __parallel_for(std::forward<_ExecutionPolicy>(__exec),
                       zip(make_iter_mode<read>(__first), make_iter_mode<write>(__d_first)),
                       zip(make_iter_mode<read>(__last), make_iter_mode<write>(__d_last)),
                       unseq_backend::walk_adjacent_difference<_ExecutionPolicy, decltype(__fn)>{__fn});

        return __d_last;
    }
}

} // namespace __internal
} // namespace dpstd

#endif /* _PSTL_numeric_impl_hetero_H */
