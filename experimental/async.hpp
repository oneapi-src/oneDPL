//
//  async_reduce.hpp
//  
//
//  Created by Pablo Reble (Intel) on 2/3/20.
//

#ifndef async_hpp
#define async_hpp

#include <CL/sycl.hpp>

#include <stdio.h>

//#include "parallel_backend_sycl_utils.h"

namespace oneapi {

namespace dpl {
    
//namespace async { // Is extra namespace necessary, or control through execution policy attributes sufficient?

namespace __internal {

        template<typename _Tp> struct __async_value {
            virtual _Tp data() = 0;
            // workaround for transform reduce pattern
            virtual sycl::buffer<_Tp> raw_data() const = 0;
            virtual ~__async_value() = default;
        };

        template<typename _Tp> struct __async_direct : public __async_value<_Tp> {
            _Tp __data;
            __async_direct(_Tp _d) : __data(_d) {}
            _Tp data() { return __data; }
            sycl::buffer<_Tp> raw_data() const {return sycl::buffer<_Tp>{__data};}
        };

        template<typename _Tp, typename _Op = ::std::plus<_Tp>, typename _Buf = sycl::buffer<_Tp>> struct __async_indirect : public __async_value<_Tp> {
            _Buf __buf;
            _Op __op;
            _Tp __init;
            __async_indirect(_Buf _b, _Tp _v = 0, _Op _o = std::plus<_Tp>()) : __buf(_b), __op(_o), __init(_v) {}
            _Tp data() {
                auto ret_val = __buf.template get_access<access_mode::read>()[0];
                return __op(ret_val, __init);
            }
            sycl::buffer<_Tp> raw_data() const {return __buf;}
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

        struct _tmp_base {};

        template<typename... Ts> struct _TempObjs : public _tmp_base {
            std::tuple<Ts&...> __my_tmps;
            _TempObjs(Ts&... __t) : __my_tmps(::std::forward_as_tuple(__t...)) {}
        };

        // TODO: Rework class hierarchy for futures:

        template<typename _ExecPolicy> using __future = __par_backend_hetero::__future<_ExecPolicy>; 

        template<typename _ExecPolicy> class __future_with_tmps {
            __future<_ExecPolicy> __my_future;

            _tmp_base __tmp;
          public:
            template<typename... _Ts> __future_with_tmps( _ExecPolicy __f , _Ts... __t) : __my_future(__future<_ExecPolicy>{__f}), __tmp(_TempObjs<_Ts...>{__t...}) {}
            void wait() { __my_future.wait(); } 
        };
/*
        template<typename _ExecPolicy, typename... _Ts> class __future_with_tmps<_ExecPolicy> __make_future_with_tmps(_ExecPolicy __f , _Ts... __t) 
            { return __future_with_tmps<_ExecPolicy>{__f,__t...};}
*/
        //using tbb::internal::make_unique;

        template <typename _ExecutionPolicy, typename _Tp /*, typename... _Ts*/>
        class __future_promise_pair //: public __future<_ExecutionPolicy>
        {
            __future<_ExecutionPolicy> __my_future; // future to wait for completion
            ::std::unique_ptr<__async_value<_Tp>> __data; // This is a value/buffer for read access!
            _tmp_base __tmp;

          public:
            template<typename... _Ts> __future_promise_pair( __future<_ExecutionPolicy> __f, _Tp __d, _Ts... __t) : 
                __my_future(__f), __data(tbb::internal::make_unique<__async_direct<_Tp>>(__d)), __tmp(_TempObjs<_Ts...>{__t...}) {}
            template<typename... _Ts> __future_promise_pair( __future<_ExecutionPolicy> __f, sycl::buffer<_Tp> __d, _Ts... __t) : 
                __my_future(__f), __data(tbb::internal::make_unique<__async_indirect<_Tp>>(__d)), __tmp(_TempObjs<_Ts...>{__t...}) {}
            // Constructor for reduce_transform pattern
            template<typename _Op> __future_promise_pair( const __future_promise_pair<_ExecutionPolicy,_Tp>& _fp, _Tp __i, _Op __o) : 
                __my_future(_fp.__my_future), __data(tbb::internal::make_unique<__async_indirect<_Tp,_Op>>(_fp.raw_data(),__i,__o)), __tmp(_fp.__tmp) {

            }
            void wait() // TODO: Func over permissive, not waiting on actual event.
            {
                __my_future.wait();
            }
            auto raw_data() const {
                return __data->raw_data(); 
            }
            _Tp data() const {
                return __data->data();
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

namespace __internal {

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

template <typename _ExecPolicy, typename _T>
using __enable_if_async_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

} // namespace internal

namespace async {
        
template<class Policy, class InputIter, class OutputIter>
oneapi::dpl::__internal::__enable_if_async_execution_policy<Policy,
oneapi::dpl::__internal::__future_promise_pair<Policy, OutputIter>>
copy(Policy&& policy, InputIter input_first, InputIter input_last, OutputIter output_first);

//namespace async {

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
    oneapi::dpl::__internal::__future_promise_pair<_ExecutionPolicy, _Tp>>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
       _BinaryOperation __binary_op);

//template<class _ExecutionPolicy, class _ForwardIterator, class T = typename std::iterator_traits<_ForwardIterator>::value_type>
//oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
//oneapi::dpl::__internal::__future_promise_pair<_ExecutionPolicy, typename ::std::iterator_traits<_ForwardIterator>::value_type>>
//reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template<class Policy, class InputIter, class UnaryFunction>
oneapi::dpl::__internal::__enable_if_async_execution_policy<Policy,
oneapi::dpl::__par_backend_hetero::__future<Policy>>
for_each(Policy&& __exec, InputIter __first, InputIter __last, UnaryFunction __f);

//template<class Policy, class InputIter>
//oneapi::dpl::__internal::__enable_if_async_execution_policy<Policy,
//oneapi::dpl::__internal::__future_with_tmps<Policy>>
//sort(Policy&& __exec, InputIter __first, InputIter __last);

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
    oneapi::dpl::__internal::__future_with_tmps<_ExecutionPolicy>>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp);

template< class Policy, class ForwardIt1, class ForwardIt2, class UnaryOperation >
oneapi::dpl::__internal::__enable_if_async_execution_policy<Policy,
oneapi::dpl::__internal::__future_promise_pair<Policy,ForwardIt2>>
transform( Policy&& policy, ForwardIt1 first1, ForwardIt1 last1,
                    ForwardIt2 d_first, UnaryOperation unary_op );

//fill();

//merge();
        
        
} // namespace async

} // namespace dpl

} // namespace oneapi

#include "async_impl.hpp"

#endif
