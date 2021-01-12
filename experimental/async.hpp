/*
 *  Copyright (c) 2020 Intel Corporation
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

#ifndef async_hpp
#define async_hpp

#include <CL/sycl.hpp>

#include <stdio.h>

//#include "parallel_backend_sycl_utils.h"

namespace oneapi
{

namespace dpl
{

// namespace async { // Is extra namespace necessary, or control through execution policy attributes sufficient?

namespace __internal
{

template <typename _Tp>
struct __async_value
{
    virtual _Tp
    data() = 0;
    // workaround for transform reduce pattern
    virtual sycl::buffer<_Tp>
    raw_data() const = 0;
    virtual ~__async_value() = default;
};

template <typename _Tp>
struct __async_direct : public __async_value<_Tp>
{
    _Tp __data;
    __async_direct(_Tp _d) : __data(_d) {}
    _Tp
    data()
    {
        return __data;
    }
    sycl::buffer<_Tp>
    raw_data() const
    {
        return sycl::buffer<_Tp>{__data};
    }
};

template <typename _Tp, typename _Op = ::std::plus<_Tp>, typename _Buf = sycl::buffer<_Tp>>
struct __async_indirect : public __async_value<_Tp>
{
    _Buf __buf;
    _Op __op;
    _Tp __init;
    __async_indirect(_Buf _b, _Tp _v = 0, _Op _o = std::plus<_Tp>()) : __buf(_b), __op(_o), __init(_v) {}
    _Tp
    data()
    {
        auto ret_val = __buf.template get_access<access_mode::read>()[0];
        return __op(ret_val, __init);
    }
    sycl::buffer<_Tp>
    raw_data() const
    {
        return __buf;
    }
};

#if 0
        // TODO: Introduce base clase and remove dummy for direct access.
        template< typename _Tp > struct _async_value : public _async_base {
            const bool indirect_obj;
            //const bool 
            _Tp __data;
            sycl::buffer<_Tp> __buf;
            _async_value(_Tp __d) : __data(__d), __buf(sycl::buffer<_Tp>{1}), indirect_obj(false) {}
            _async_value(sycl::buffer<_Tp> __b) : __buf(__b), indirect_obj(true) {}
            _async_value(_Tp __init, sycl::buffer<_Tp> __b) : __data(__init), __buf(__b), indirect_obj(true) {}
            
            _Tp raw_data() {
                if(indirect_obj) return __buf.template get_access<access_mode::read>()[0];
                return __data;
            }
        };
#endif
//

struct _tmp_base
{
};

template <typename... Ts>
struct _TempObjs : public _tmp_base
{
    std::tuple<Ts&...> __my_tmps;
    _TempObjs(Ts&... __t) : __my_tmps(::std::forward_as_tuple(__t...)) {}
};

#if 1

using event = sycl::event;

#else

class event {
    sycl::event __event;
    event( sycl::event __e ) : __event(__e) {}
}

#endif

#if 1

class __future_base {
    event __my_event;
public:
    __future_base(event __e) : __my_event(__e) {}
    event get_event() const { return __my_event; }
    void wait() { __my_event.wait(); }
    operator event() const { return event(__my_event); }
};

#else

template <typename _ExecPolicy>
using __future = __par_backend_hetero::__future<_ExecPolicy>;

#endif

/*
        template<typename _ExecPolicy, typename... _Ts> class __future_with_tmps<_ExecPolicy>
   __make_future_with_tmps(_ExecPolicy __f , _Ts... __t) { return __future_with_tmps<_ExecPolicy>{__f,__t...};}
*/

template <typename _Tp>
class __future : public __future_base
{
    ::std::unique_ptr<__async_value<_Tp>> __data; // This is a value/buffer for read access!
    _tmp_base __tmp;

  public:
    template <typename... _Ts>
    __future(event __e, _Tp __d, _Ts... __t)
        : __future_base(__e), __data(::std::unique_ptr<__async_direct<_Tp>>(new __async_direct<_Tp>(__d))),
          __tmp(_TempObjs<_Ts...>{__t...})
    {
    }
    template <typename... _Ts>
    __future(event __e, sycl::buffer<_Tp> __d, _Ts... __t)
        : __future_base(__e), __data(::std::unique_ptr<__async_indirect<_Tp>>(new __async_indirect<_Tp>(__d))),
          __tmp(_TempObjs<_Ts...>{__t...})
    {
    }
    // Constructor for reduce_transform pattern
    template <typename _Op>
    __future(const __future<_Tp>& _fp, _Tp __i, _Op __o)
        : __future_base(_fp.get_event()),
          __data(::std::unique_ptr<__async_indirect<_Tp, _Op>>(new __async_indirect<_Tp, _Op>(_fp.raw_data(), __i, __o))), __tmp(_fp.__tmp)
    {
    }
    auto
    raw_data() const
    {
        return __data->raw_data();
    }
    _Tp
    data() const
    {
        return __data->data();
    }
};

template<>
class __future<void> : public __future_base
{
    _tmp_base __tmp;

  public:
    template <typename... _Ts>
    __future(event __e, _Ts... __t)
        : __future_base(__e), __tmp(_TempObjs<_Ts...>{__t...})
    {
    }
};

/*
        template<typename _ExecutionPolicy, typename _T, typename... _Ts> __future_promise_pair<_ExecutionPolicy,_T>
        __make_future_promise_pair( __future<_ExecutionPolicy> __f, _T __v , _Ts... __t) {
            return __future_promise_pair<_ExecutionPolicy,_T>{__f,__v,__t...};
        }

        template<typename _ExecutionPolicy, typename _T, typename... _Ts> __future_promise_pair<_ExecutionPolicy,_T>
        __make_future_promise_pair_indirect( __future<_ExecutionPolicy> __f, sycl::buffer<_T> __d, _Ts... __t) {
            return __future_promise_pair<_ExecutionPolicy,_T>{__f,__d,__t...};
        }
*/
} // namespace __internal

#if 0

namespace execution {

template <typename KernelName = DefaultKernelName>
class async_policy : public device_policy<KernelName> {
using parent = device_policy<KernelName>;
public:
explicit async_policy(sycl::queue q_) : parent(q_) {}
};

// make_policy functions
template <typename KernelName = DefaultKernelName>
async_policy<KernelName>
make_async_policy(sycl::queue q)
{
    return async_policy<KernelName>(q);
}

} // namespace execution

#endif

namespace __internal
{

template <typename _T>
struct __is_async_execution_policy : ::std::false_type
{
};

#if 0

template <typename... PolicyParams>
struct __is_async_execution_policy<::oneapi::dpl::execution::async_policy<PolicyParams...>> : ::std::true_type
{
};

template <typename... PolicyParams>
struct __is_hetero_execution_policy<::oneapi::dpl::execution::async_policy<PolicyParams...>> : ::std::true_type
{
};

template <typename... PolicyParams>
struct __is_device_execution_policy<::oneapi::dpl::execution::async_policy<PolicyParams...>> : ::std::true_type
{
};

#endif
/*
template <typename _ExecPolicy, typename _T>
using __enable_if_async_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;
*/
template <typename _ExecPolicy, typename _T, typename... _Events>
using __enable_if_async_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value && ( true && ... && ::std::is_convertible_v<_Events,event> ) , _T>::type;

template <typename _ExecPolicy, typename _T, typename _Op1, typename... _Events>
using __enable_if_async_execution_policy_1 = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value && !::std::is_convertible_v<_Op1,event> && ( true && ... && ::std::is_convertible_v<_Events,event> ) , _T>::type;

} // namespace __internal

namespace async
{

template <class _ExecutionPolicy, class InputIter, class OutputIter>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<OutputIter>>
copy(_ExecutionPolicy&& __exec, InputIter __input_first, InputIter __input_last, OutputIter __output_first);

// namespace async {

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
       _BinaryOperation __binary_op);

template <class _ExecutionPolicy, class InputIter, class UnaryFunction>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<void>>
for_each(_ExecutionPolicy&& __exec, InputIter __first, InputIter __last, UnaryFunction __f);

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<void>>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _UnaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIt2>>
transform(_ExecutionPolicy&& policy, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 d_first, _UnaryOperation unary_op);

// fill();

// merge();

template <typename... _Ts>
void
wait_for_all(const _Ts&... __Events) {
    ::std::vector<__internal::event> __wait_list = {__Events...};
    for(auto _a : __wait_list) _a.wait();
}
    
} // namespace async

// Public API for asynch algorithms:

template <class _ExecutionPolicy, class InputIter, class OutputIter, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<OutputIter>>
copy_async(_ExecutionPolicy&& __exec, InputIter __input_first, InputIter __input_last, OutputIter __output_first, _Events&&... __dependencies) {
    oneapi::dpl::async::wait_for_all(__dependencies...);
    return async::copy(std::forward<_ExecutionPolicy>(__exec), __input_first, __input_last, __output_first);
}

template <class _ExecutionPolicy, class InputIter, class UnaryFunction, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<void>>
for_each_async(_ExecutionPolicy&& __exec, InputIter __first, InputIter __last, UnaryFunction __f,  _Events&&... __dependencies) {
    oneapi::dpl::async::wait_for_all(__dependencies...);
    return async::for_each(std::forward<_ExecutionPolicy>(__exec), __first, __last, __f);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>, _Events...>
reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op, _Events&&... __dependencies) {
    oneapi::dpl::async::wait_for_all(__dependencies...);
    return async::reduce(std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op);
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<void>, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Events&&... __dependencies) {
    using __T = typename ::std::iterator_traits<_RandomAccessIterator>::value_type;
    oneapi::dpl::async::wait_for_all(__dependencies...);
    return async::sort( std::forward<_ExecutionPolicy>(__exec), __first, __last, ::std::less<__T>() );
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_1<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<void>, _Compare, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, _Events&&... __dependencies) {
    oneapi::dpl::async::wait_for_all(__dependencies...);
    return async::sort( std::forward<_ExecutionPolicy>(__exec), __first, __last, __comp );
}

} // namespace dpl

} // namespace oneapi

#include "async_impl.hpp"

#endif
