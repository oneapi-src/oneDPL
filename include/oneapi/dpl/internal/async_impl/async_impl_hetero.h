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

#ifndef _ONEDPL_ASYNC_IMPL_HETERO_H
#define _ONEDPL_ASYNC_IMPL_HETERO_H

#include <cassert>

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Function,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_walk1_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f)
{
    auto __n = __last - __first;
    assert(__n > 0);

    auto __keep =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf.all_view());
    return __future_obj;
}

template <typename _IsSync = ::std::false_type,
          __par_backend_hetero::access_mode __acc_mode1 = __par_backend_hetero::access_mode::read,
          __par_backend_hetero::access_mode __acc_mode2 = __par_backend_hetero::access_mode::write,
          typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Function,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_walk2_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                      _ForwardIterator2 __first2, _Function __f)
{
    auto __n = __last1 - __first1;
    assert(__n > 0);

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode1, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);

    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode2, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __future = oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf1.all_view(), __buf2.all_view());

    if constexpr (_IsSync::value)
        __future.wait();

    return __future.__make_future(__first2 + __n);
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _ForwardIterator3,
          typename _Function, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_walk3_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                      _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f)
{
    auto __n = __last1 - __first1;
    assert(__n > 0);

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);
    auto __keep3 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator3>();
    auto __buf3 = __keep3(__first3, __first3 + __n);

    auto __future = oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf1.all_view(), __buf2.all_view(), __buf3.all_view());

    return __future.__make_future(__first3 + __n);
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Brick,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_walk2_brick_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                            _ForwardIterator2 __first2, _Brick __brick)
{
    return __pattern_walk2_async(
        __par_backend_hetero::make_wrapped_policy<__walk2_brick_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __last1, __first2, __brick);
}

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Tp,
          typename _BinaryOperation1, typename _BinaryOperation2,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_transform_reduce_async(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                                 _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Tp __init,
                                 _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    assert(__first1 < __last1);

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _BinaryOperation2>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    auto __n = __last1 - __first1;
    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_Policy, _BinaryOperation1, _Functor>{__binary_op1,
                                                                            _Functor{__binary_op2}}, // transform
        unseq_backend::transform_init<_Policy, _BinaryOperation1, _NoOpFunctor>{__binary_op1, _NoOpFunctor{}},
        unseq_backend::reduce<_Policy, _BinaryOperation1, _RepackedTp>{__binary_op1}, // reduce
        unseq_backend::__init_value<_Tp>{__init},                                     //initial value
        __buf1.all_view(), __buf2.all_view());
    return __res;
}

//------------------------------------------------------------------------
// transform_reduce (with unary and binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Tp, typename _BinaryOperation,
          typename _UnaryOperation,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                                 _Tp __init, _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    assert(__first < __last);

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _UnaryOperation>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_Policy, _BinaryOperation, _Functor>{__binary_op,
                                                                           _Functor{__unary_op}}, // transform
        unseq_backend::transform_init<_Policy, _BinaryOperation, _NoOpFunctor>{__binary_op, _NoOpFunctor{}},
        unseq_backend::reduce<_Policy, _BinaryOperation, _RepackedTp>{__binary_op}, // reduce
        unseq_backend::__init_value<_Tp>{__init},                                   //initial value
        __buf.all_view());
    return __res;
}

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _T,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_fill_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _T& __value)
{
    return __pattern_walk1_async(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__last),
        fill_functor<_T>{__value});
}

//------------------------------------------------------------------------
// transform_scan
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation,
          typename _InitType, typename _BinaryOperation, typename _Inclusive,
          oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_transform_scan_base_async(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last,
                                    _Iterator2 __result, _UnaryOperation __unary_op, _InitType __init,
                                    _BinaryOperation __binary_op, _Inclusive)
{
    assert(__first < __last);

    using _Type = typename _InitType::__value_type;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _UnaryFunctor = unseq_backend::walk_n<_ExecutionPolicy, _UnaryOperation>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _Assigner __assign_op;
    _NoAssign __no_assign_op;
    _NoOpFunctor __get_data_op;

    auto __n = __last - __first;
    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator2>();
    auto __buf2 = __keep2(__result, __result + __n);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_transform_scan(
        ::std::forward<_ExecutionPolicy>(__exec), __buf1.all_view(), __buf2.all_view(), __binary_op, __init,
        // local scan
        unseq_backend::__scan<_Inclusive, _ExecutionPolicy, _BinaryOperation, _UnaryFunctor, _Assigner, _Assigner,
                              _NoOpFunctor, _InitType>{__binary_op, _UnaryFunctor{__unary_op}, __assign_op, __assign_op,
                                                       __get_data_op},
        // scan between groups
        unseq_backend::__scan</*inclusive=*/::std::true_type, _ExecutionPolicy, _BinaryOperation, _NoOpFunctor,
                              _NoAssign, _Assigner, _NoOpFunctor, unseq_backend::__no_init_value<_Type>>{
            __binary_op, _NoOpFunctor{}, __no_assign_op, __assign_op, __get_data_op},
        // global scan
        unseq_backend::__global_scan_functor<_Inclusive, _BinaryOperation, _InitType>{__binary_op, __init});
    return __res.__make_future(__result + __n);
}

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation, typename _Type,
          typename _BinaryOperation, typename _Inclusive,
          oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_transform_scan_async(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                               _UnaryOperation __unary_op, _Type __init, _BinaryOperation __binary_op, _Inclusive)
{
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__init_value<_RepackedType>;

    return __pattern_transform_scan_base_async(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                                               __unary_op, _InitType{__init}, __binary_op, _Inclusive{});
}

// scan without initial element
template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation,
          typename _BinaryOperation, typename _Inclusive,
          oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_transform_scan_async(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                               _UnaryOperation __unary_op, _BinaryOperation __binary_op, _Inclusive)
{
    using _Type = typename ::std::iterator_traits<_Iterator1>::value_type;
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__no_init_value<_RepackedType>;

    return __pattern_transform_scan_base_async(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                                               __unary_op, _InitType{}, __binary_op, _Inclusive{});
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_ASYNC_IMPL_HETERO_H */
