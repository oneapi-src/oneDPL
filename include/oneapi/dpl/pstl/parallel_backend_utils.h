// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_UTILS_H
#define _ONEDPL_PARALLEL_BACKEND_UTILS_H

#include <iterator>
#include <utility>
#include <cassert>
#include "utils.h"
#include "memory_fwd.h"

namespace oneapi
{
namespace dpl
{
namespace __utils
{
//------------------------------------------------------------------------
// Uninitialized / initialized copy / move for trivial / non-trivial types
//------------------------------------------------------------------------

// struct __op_smart_ctor - helper struct for uninitialized/initialized copy/move operations.
// It can be used to perform copy/move operations for scalar and non-scalar types.
template <typename _ValueType>
struct __op_smart_ctor
{
    using _ValueTypeDecayed = std::decay_t<_ValueType>;

    template <typename... Args>
    inline void
    operator()(_ValueTypeDecayed* __p_value_to, Args&&... args) const
    {
        if constexpr (std::is_scalar_v<_ValueTypeDecayed> && sizeof...(Args) <= 1)
        {
            if constexpr (sizeof...(Args) == 1)
            {
                __assign_impl(__p_value_to, std::forward<Args>(args)...);
            }
        }
        else
        {
            ::new (__p_value_to) _ValueTypeDecayed(std::forward<Args>(args)...);
        }
    }

  private:
    template <typename Arg>
    inline void
    __assign_impl(_ValueTypeDecayed* __p_value_to, Arg&& arg) const
    {
        *__p_value_to = std::forward<Arg>(arg);
    }
};

// struct __op_smart_dtor - helper struct for destructor calls.
// It can be used to perform destructor calls for types,
// which are not trivially-destructible (we call destructor)
// and for all other types (we doesn't call destructor).
template <typename _ValueType>
struct __op_smart_dtor
{
    static_assert(std::is_same_v<_ValueType, std::decay_t<_ValueType>>);

    static constexpr bool required = !std::is_trivially_destructible_v<_ValueType>;

    inline void
    operator()(_ValueType& __value) const
    {
        if constexpr (required)
            __value.~_ValueType();
    }
};

//------------------------------------------------------------------------
// raw buffer (with specified _TAllocator)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Tp, template <typename _T> typename _TAllocator>
class __buffer_impl
{
    _TAllocator<_Tp> _M_allocator;
    _Tp* _M_ptr = nullptr;
    const ::std::size_t _M_buf_size = 0;

    __buffer_impl(const __buffer_impl&) = delete;
    void
    operator=(const __buffer_impl&) = delete;

  public:
    //! Try to obtain buffer of given size to store objects of _Tp type
    __buffer_impl(_ExecutionPolicy /*__exec*/, const ::std::size_t __n)
        : _M_allocator(), _M_ptr(_M_allocator.allocate(__n)), _M_buf_size(__n)
    {
    }
    //! True if buffer was successfully obtained, zero otherwise.
    operator bool() const { return _M_ptr != nullptr; }
    //! Return pointer to buffer, or nullptr if buffer could not be obtained.
    _Tp*
    get() const
    {
        return _M_ptr;
    }
    //! Destroy buffer
    ~__buffer_impl() { _M_allocator.deallocate(_M_ptr, _M_buf_size); }
};

//! Destroy sequence [xs,xe)
struct __serial_destroy
{
    template <typename _RandomAccessIterator>
    void
    operator()(_RandomAccessIterator __zs, _RandomAccessIterator __ze)
    {
        typedef typename ::std::iterator_traits<_RandomAccessIterator>::value_type _ValueType;

        if constexpr (oneapi::dpl::__utils::__op_smart_dtor<_ValueType>::required)
        {
            while (__zs != __ze)
            {
                --__ze;
                oneapi::dpl::__utils::__op_smart_dtor<_ValueType>{}(*__ze);
            }
        }
    }
};

//! Merge sequences [__xs,__xe) and [__ys,__ye) to output sequence [__zs,(__xe-__xs)+(__ye-__ys)), using ::std::move
struct __serial_move_merge
{
    const ::std::size_t _M_nmerge;

    explicit __serial_move_merge(::std::size_t __nmerge) : _M_nmerge(__nmerge) {}
    template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Compare,
              class _MoveValueX, class _MoveValueY, class _MoveSequenceX, class _MoveSequenceY>
    void
    operator()(_RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe, _RandomAccessIterator2 __ys,
               _RandomAccessIterator2 __ye, _RandomAccessIterator3 __zs, _Compare __comp, _MoveValueX __move_value_x,
               _MoveValueY __move_value_y, _MoveSequenceX __move_sequence_x, _MoveSequenceY __move_sequence_y)
    {
        constexpr bool __same_move_val = ::std::is_same_v<_MoveValueX, _MoveValueY>;
        constexpr bool __same_move_seq = ::std::is_same_v<_MoveSequenceX, _MoveSequenceY>;

        auto __n = _M_nmerge;
        assert(__n > 0);

        auto __nx = __xe - __xs;
        //auto __ny = __ye - __ys;
        _RandomAccessIterator3 __zs_beg = __zs;

        if (__xs != __xe)
        {
            if (__ys != __ye)
            {
                for (;;)
                {
                    if (__comp(*__ys, *__xs))
                    {
                        const auto __i = __zs - __zs_beg;
                        if (__i < __nx)
                            __move_value_x(__ys, __zs);
                        else
                            __move_value_y(__ys, __zs);
                        ++__zs, --__n;
                        if (++__ys == __ye)
                        {
                            break;
                        }
                        else if (__n == 0)
                        {
                            const auto __j = __zs - __zs_beg;
                            if (__same_move_seq || __j < __nx)
                                __zs = __move_sequence_x(__ys, __ye, __zs);
                            else
                                __zs = __move_sequence_y(__ys, __ye, __zs);
                            break;
                        }
                    }
                    else
                    {
                        const auto __i = __zs - __zs_beg;
                        if (__same_move_val || __i < __nx)
                            __move_value_x(__xs, __zs);
                        else
                            __move_value_y(__xs, __zs);
                        ++__zs, --__n;
                        if (++__xs == __xe)
                        {
                            const auto __j = __zs - __zs_beg;
                            if (__same_move_seq || __j < __nx)
                                __move_sequence_x(__ys, __ye, __zs);
                            else
                                __move_sequence_y(__ys, __ye, __zs);
                            return;
                        }
                        else if (__n == 0)
                        {
                            const auto __j = __zs - __zs_beg;
                            if (__same_move_seq || __j < __nx)
                            {
                                __zs = __move_sequence_x(__xs, __xe, __zs);
                                __move_sequence_x(__ys, __ye, __zs);
                            }
                            else
                            {
                                __zs = __move_sequence_y(__xs, __xe, __zs);
                                __move_sequence_y(__ys, __ye, __zs);
                            }
                            return;
                        }
                    }
                }
            }
            __ys = __xs;
            __ye = __xe;
        }
        const auto __i = __zs - __zs_beg;
        if (__same_move_seq || __i < __nx)
            __move_sequence_x(__ys, __ye, __zs);
        else
            __move_sequence_y(__ys, __ye, __zs);
    }
};

template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _Compare,
          typename _CopyConstructRange>
_OutputIterator
__set_union_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                      _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                      _CopyConstructRange __cc_range)
{
    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    for (; __first1 != __last1; ++__result)
    {
        if (__first2 == __last2)
            return __cc_range(__first1, __last1, __result);
        if (__comp(*__first2, *__first1))
        {
            __op_smart_ctor<_Tp>{}(std::addressof(*__result), *__first2);
            ++__first2;
        }
        else
        {
            __op_smart_ctor<_Tp>{}(std::addressof(*__result), *__first1);
            if (!__comp(*__first1, *__first2))
                ++__first2;
            ++__first1;
        }
    }
    return __cc_range(__first2, __last2, __result);
}

template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _Compare,
          typename _CopyFunc, typename _CopyFromFirstSet>
_OutputIterator
__set_intersection_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                             _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _CopyFunc _copy,
                             _CopyFromFirstSet)
{
    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;
    for (; __first1 != __last1 && __first2 != __last2;)
    {
        if (__comp(*__first1, *__first2))
            ++__first1;
        else
        {
            if (!__comp(*__first2, *__first1))
            {

                if constexpr (_CopyFromFirstSet::value)
                {
                    _copy(*__first1, *__result);
                }
                else
                {
                    _copy(*__first2, *__result);
                }
                ++__result;
                ++__first1;
            }
            ++__first2;
        }
    }
    return __result;
}

template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _Compare,
          typename _CopyConstructRange>
_OutputIterator
__set_difference_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                           _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                           _CopyConstructRange __cc_range)
{
    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    for (; __first1 != __last1;)
    {
        if (__first2 == __last2)
            return __cc_range(__first1, __last1, __result);

        if (__comp(*__first1, *__first2))
        {
            __op_smart_ctor<_Tp>{}(std::addressof(*__result), *__first1);
            ++__result;
            ++__first1;
        }
        else
        {
            if (!__comp(*__first2, *__first1))
                ++__first1;
            ++__first2;
        }
    }
    return __result;
}
template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _Compare,
          typename _CopyConstructRange>
_OutputIterator
__set_symmetric_difference_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                                     _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                                     _CopyConstructRange __cc_range)
{
    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    for (; __first1 != __last1;)
    {
        if (__first2 == __last2)
            return __cc_range(__first1, __last1, __result);

        if (__comp(*__first1, *__first2))
        {
            __op_smart_ctor<_Tp>{}(std::addressof(*__result), *__first1);
            ++__result;
            ++__first1;
        }
        else
        {
            if (__comp(*__first2, *__first1))
            {
                __op_smart_ctor<_Tp>{}(std::addressof(*__result), *__first2);
                ++__result;
            }
            else
                ++__first1;
            ++__first2;
        }
    }
    return __cc_range(__first2, __last2, __result);
}

} // namespace __utils
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_UTILS_H
