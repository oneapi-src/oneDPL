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

#ifndef _ONEDPL_UTILS_RANGES_SYCL_H
#define _ONEDPL_UTILS_RANGES_SYCL_H

#include <iterator>
#include <type_traits>

#include "../../utils_ranges.h"
#include "../../iterator_impl.h"
#include "../../glue_numeric_defs.h"
#include "sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __ranges
{

//A SYCL range over SYCL buffer
template <typename _T, sycl::access::mode _AccMode = sycl::access::mode::read,
          __dpl_sycl::__target _Target = __dpl_sycl::__target_device,
          sycl::access::placeholder _Placeholder = sycl::access::placeholder::true_t>
class all_view
{
    using __return_t = typename ::std::conditional<_AccMode == sycl::access::mode::read, const _T, _T>::type;
    using __diff_type = typename ::std::iterator_traits<_T*>::difference_type;
    using __accessor_t = sycl::accessor<_T, 1, _AccMode, _Target, _Placeholder>;

  public:
    using value_type = _T;

    all_view(sycl::buffer<_T, 1> __buf = sycl::buffer<_T, 1>(0), __diff_type __offset = 0, __diff_type __n = 0)
        : __m_acc(__create_accessor(__buf, __offset, __n))
    {
    }

    all_view(__accessor_t __acc) : __m_acc(__acc) {}

    __return_t*
    begin() const
    {
        return &__m_acc[0];
    } //or “honest” iterator over an accessor and a sentinel

    __return_t*
    end() const
    {
        return begin() + size();
    }
    __return_t& operator[](__diff_type i) const { return begin()[i]; }

    __diff_type
    size() const
    {
        return __dpl_sycl::__get_accessor_size(__m_acc);
    }
    bool
    empty() const
    {
        return size() == 0;
    }

    void
    require_access(sycl::handler& __cgh)
    {
        __cgh.require(__m_acc);
    }

  private:
    static __accessor_t __create_accessor(sycl::buffer<_T, 1>& __buf, __diff_type __offset, __diff_type __n)
    {
        auto __n_buf = __dpl_sycl::__get_buffer_size(__buf);
        auto __n_acc = (__n > 0 ? __n : __n_buf);

        assert(__offset + __n_acc <= __n_buf &&
               "The sum of accessRange and accessOffset should not exceed the range of buffer");

        return {__buf, sycl::range<1>(__n_acc), __offset};
    }

  private:
    __accessor_t __m_acc;
};

template <sycl::access::mode AccMode = sycl::access::mode::read_write,
          __dpl_sycl::__target _Target = __dpl_sycl::__target_device,
          sycl::access::placeholder _Placeholder = sycl::access::placeholder::true_t>
struct all_view_fn
{
    template <typename _T>
    constexpr oneapi::dpl::__ranges::all_view<_T, AccMode, _Target, _Placeholder>
    operator()(sycl::buffer<_T, 1> __buf, typename ::std::iterator_traits<_T*>::difference_type __offset = 0,
               typename ::std::iterator_traits<_T*>::difference_type __n = 0) const
    {
        return oneapi::dpl::__ranges::all_view<_T, AccMode, _Target, _Placeholder>(__buf, __offset, __n);
    }

    template <typename _R>
    auto
    operator()(_R&& __r) const -> decltype(::std::forward<_R>(__r))
    {
        return ::std::forward<_R>(__r);
    }
};

namespace views
{
inline constexpr all_view_fn<sycl::access::mode::read_write, __dpl_sycl::__target_device,
                             sycl::access::placeholder::true_t>
    all;

inline constexpr all_view_fn<sycl::access::mode::read, __dpl_sycl::__target_device, sycl::access::placeholder::true_t>
    all_read;

inline constexpr all_view_fn<sycl::access::mode::write, __dpl_sycl::__target_device, sycl::access::placeholder::true_t>
    all_write;

inline constexpr all_view_fn<sycl::access::mode::read_write, __dpl_sycl::__host_target,
                             sycl::access::placeholder::false_t>
    host_all;
} // namespace views

//all_view traits

template <typename Iter, typename Void = void> // for iterators that should not be passed directly
struct is_zip : ::std::false_type
{
};

template <typename Iter> // for iterators defined as direct pass
struct is_zip<Iter, typename ::std::enable_if<Iter::is_zip::value, void>::type> : ::std::true_type
{
};

template <typename Iter, typename Void = void>
struct is_permutation : ::std::false_type
{
};

template <typename Iter> // for permutation_iterators
struct is_permutation<Iter, typename ::std::enable_if<Iter::is_permutation::value, void>::type> : ::std::true_type
{
};

template <typename _Iter>
using is_hetero_it = oneapi::dpl::__internal::is_hetero_iterator<_Iter>;

template <typename _Iter>
using is_passed_directly_it = oneapi::dpl::__internal::is_passed_directly<_Iter>;

//struct for checking if it needs to create a temporary SYCL buffer or not

template <typename _Iter, typename Void = void>
struct is_temp_buff : ::std::false_type
{
};

template <typename _Iter>
struct is_temp_buff<_Iter, typename ::std::enable_if<!is_hetero_it<_Iter>::value && !::std::is_pointer<_Iter>::value &&
                                                         !is_passed_directly_it<_Iter>::value,
                                                     void>::type> : ::std::true_type
{
};

template <typename _Iter>
using val_t = typename ::std::iterator_traits<_Iter>::value_type;

//range/zip_view/all_view/ variadic utilities

template <typename _Range, typename... _Ranges>
struct __get_first_range_type
{
    using type = _Range;
};

template <typename _Range, typename... _Ranges>
constexpr auto
__get_first_range_size(const _Range& __rng, const _Ranges&...) -> decltype(__rng.size())
{
    return __rng.size();
}

template <typename _Cgh>
struct _require_access_args
{
    _Cgh __cgh;
    template <typename... Args>
    void
    operator()(Args&&... args)
    {
        __require_access(__cgh, ::std::forward<Args>(args)...);
    }
};

template <typename... _Ranges>
void
__require_access_zip(sycl::handler& __cgh, oneapi::dpl::__ranges::zip_view<_Ranges...>& __zip)
{
    const ::std::size_t __num_ranges = sizeof...(_Ranges);
    oneapi::dpl::__ranges::invoke(__zip.tuple(), _require_access_args<decltype(__cgh)>{__cgh},
                                  ::std::make_index_sequence<__num_ranges>());
}

//__require_access utility

inline void
__require_access(sycl::handler&)
{
}

template <typename T, sycl::access::mode M>
void
__require_access_range(sycl::handler& __cgh, oneapi::dpl::__ranges::all_view<T, M>& sycl_view)
{
    sycl_view.require_access(__cgh);
}

template <typename... _Ranges>
void
__require_access_range(sycl::handler& __cgh, zip_view<_Ranges...>& zip_rng)
{
    __require_access_zip(__cgh, zip_rng);
}

template <typename _BaseRange>
void
__require_access_range(sycl::handler&, _BaseRange&)
{
}

template <typename _Range, typename... _Ranges>
void
__require_access(sycl::handler& __cgh, _Range&& __rng, _Ranges&&... __rest)
{
    assert(!__rng.empty());

    //getting an access for the all_view based range
    auto base_rng = oneapi::dpl::__ranges::pipeline_base_range<_Range>(::std::forward<_Range>(__rng)).base_range();

    __require_access_range(__cgh, base_rng);

    //getting an access for the rest ranges
    __require_access(__cgh, ::std::forward<_Ranges>(__rest)...);
}

template <typename _R>
struct __range_holder
{
    _R __r;
    constexpr _R
    all_view() const
    {
        return __r;
    }

    //TODO: The dummy conversion operators.
    //In case when a temporary buffer doesn't need we have to use a dummy type - "oneapi::dpl::internal::ignore_copyable, for example
    operator oneapi::dpl::internal::ignore_copyable() const { return oneapi::dpl::internal::ignore; }
    operator oneapi::dpl::__internal::tuple<oneapi::dpl::internal::ignore_copyable,
                                            oneapi::dpl::internal::ignore_copyable>() const
    {
        return oneapi::dpl::__internal::tuple<oneapi::dpl::internal::ignore_copyable,
                                              oneapi::dpl::internal::ignore_copyable>(oneapi::dpl::internal::ignore,
                                                                                      oneapi::dpl::internal::ignore);
    }
    operator oneapi::dpl::__internal::tuple<oneapi::dpl::internal::ignore_copyable,
                                            oneapi::dpl::internal::ignore_copyable,
                                            oneapi::dpl::internal::ignore_copyable>() const
    {
        return oneapi::dpl::__internal::tuple<oneapi::dpl::internal::ignore_copyable,
                                              oneapi::dpl::internal::ignore_copyable,
                                              oneapi::dpl::internal::ignore_copyable>(
            oneapi::dpl::internal::ignore, oneapi::dpl::internal::ignore, oneapi::dpl::internal::ignore);
    }
};

// We have to keep sycl buffer instance here by sync reasons, at least in case of host iterators. SYCL runtime has sync
// in buffer destruction and a sycl view instance keeps just placeholder accessor, not a buffer.
template <typename _T>
using buf_type = sycl::buffer<_T, 1>;

template <typename _T, sycl::access::mode AccMode>
struct __buffer_holder
{
    buf_type<_T> __buf;

    constexpr oneapi::dpl::__ranges::all_view<_T, AccMode>
    all_view() const
    {
        return oneapi::dpl::__ranges::all_view<_T, AccMode>(__buf);
    }
};

template <sycl::access::mode AccMode, typename _Iterator>
struct __get_sycl_range
{
    __get_sycl_range()
    {
        m_buffers.reserve(4); //4 - due to a number of arguments(host iterators) cannot be too big.
    }

  private:
    //We have to keep sycl buffer(s) instance here by sync reasons;
    ::std::vector<::std::shared_ptr<oneapi::dpl::__internal::__lifetime_keeper_base>> m_buffers;

    template <typename _Iter>
    buf_type<val_t<_Iter>>
    copy_back(_Iter __first, buf_type<val_t<_Iter>> buf, /*_copy_back*/ ::std::false_type)
    {
        return buf;
    }

    template <typename _Iter>
    buf_type<val_t<_Iter>>
    copy_back(_Iter __first, buf_type<val_t<_Iter>> buf, /*_copy_back*/ ::std::true_type)
    {
        buf.set_final_data(__first);
        buf.set_write_back(true);
        return buf;
    }

    template <typename _Iter, typename _CopyDirectTag>
    buf_type<val_t<_Iter>>
    copy_direct(_Iter __first, _Iter __last, _CopyDirectTag)
    {
        //create a SYCL buffer and copy data [first, last) or create a empty SYCL buffer with size = (last - first)
        if constexpr (_CopyDirectTag::value)
            return sycl::buffer<val_t<_Iter>, 1>(__first, __last);
        else
            return sycl::buffer<val_t<_Iter>, 1>(__last - __first);
    }

    template <typename _F, typename _It, typename _DiffType>
    static auto
    gen_view(_F& __f, _It __it, _DiffType __n) -> decltype(__f(__it, __it + __n))
    {
        return __f(__it, __it + __n);
    }

    template <typename _TupleType, typename _DiffType, ::std::size_t... _Ip>
    auto
    gen_zip_view(_TupleType __t, _DiffType __n, ::std::index_sequence<_Ip...>)
        -> decltype(oneapi::dpl::__ranges::make_zip_view(gen_view(*this, ::std::get<_Ip>(__t), __n).all_view()...))
    {
        auto tmp = oneapi::dpl::__internal::make_tuple(gen_view(*this, ::std::get<_Ip>(__t), __n)...);
        return oneapi::dpl::__ranges::make_zip_view(::std::get<_Ip>(tmp).all_view()...);
    }

  public:
    //zip iterators

    template <typename... Iters>
    auto
    operator()(oneapi::dpl::zip_iterator<Iters...> __first, oneapi::dpl::zip_iterator<Iters...> __last)
        -> decltype(__range_holder<decltype(gen_zip_view(__first.base(), __last - __first,
                                                         ::std::make_index_sequence<sizeof...(Iters)>()))>{
            gen_zip_view(__first.base(), __last - __first, ::std::make_index_sequence<sizeof...(Iters)>())})
    {
        assert(__first < __last);

        const ::std::size_t __num_it = sizeof...(Iters);
        auto rng = gen_zip_view(__first.base(), __last - __first, ::std::make_index_sequence<__num_it>());
        return __range_holder<decltype(rng)>{rng};
    }

    //specialization for transform_iterator
    template <typename _Iter, typename _UnaryFunction>
    auto
    operator()(oneapi::dpl::transform_iterator<_Iter, _UnaryFunction> __first,
               oneapi::dpl::transform_iterator<_Iter, _UnaryFunction> __last)
        -> __range_holder<oneapi::dpl::__ranges::transform_view_simple<
            decltype(::std::declval<__get_sycl_range<AccMode, _Iterator>>()(__first.base(), __last.base()).all_view()),
            _UnaryFunction>>
    {
        assert(__first < __last);

        auto res = this->operator()(__first.base(), __last.base());
        auto rng = oneapi::dpl::__ranges::transform_view_simple<decltype(res.all_view()), decltype(__first.functor())>{
            res.all_view(), __first.functor()};

        return __range_holder<decltype(rng)>{rng};
    }

  private:
    template <typename _R, typename _Map, typename _Size,
              typename ::std::enable_if<oneapi::dpl::__internal::__is_functor<_Map>, int>::type = 0>
    static auto
    __get_permutation_view(_R __r, _Map __m, _Size __s)
    {
        return oneapi::dpl::__ranges::permutation_view_simple<_R, _Map>{__r, __m, __s};
    }

    template <
        typename _R, typename _Map, typename _Size,
        typename ::std::enable_if<oneapi::dpl::__internal::__is_random_access_iterator<_Map>::value, int>::type = 0>
    auto
    __get_permutation_view(_R __r, _Map __m, _Size __s)
    {
        auto view_map = this->operator()(__m, __m + __s).all_view();
        return oneapi::dpl::__ranges::permutation_view_simple<_R, decltype(view_map)>{__r, view_map};
    }

  public:
    //specialization for permutation_iterator using sycl_iterator as source
    template <typename _It, typename _Map, typename ::std::enable_if<is_hetero_it<_It>::value, int>::type = 0>
    auto
    operator()(oneapi::dpl::permutation_iterator<_It, _Map> __first,
               oneapi::dpl::permutation_iterator<_It, _Map> __last)
    {
        auto __n = __last - __first;
        assert(__n > 0);

        auto res_src = this->operator()(__first.base(),
                                        __first.base() + __dpl_sycl::__get_buffer_size(__first.base().get_buffer()));

        auto rng = __get_permutation_view(res_src.all_view(), __first.map(), __n);

        return __range_holder<decltype(rng)>{rng};
    }

    // TODO Add specialization for general case, e.g., permutation_iterator using host
    // or another fancy iterator.
    //specialization for permutation_iterator using USM pointer as source
    template <typename _It, typename _Map, typename ::std::enable_if<!is_hetero_it<_It>::value, int>::type = 0>
    auto
    operator()(oneapi::dpl::permutation_iterator<_It, _Map> __first,
               oneapi::dpl::permutation_iterator<_It, _Map> __last)
    {
        auto __n = __last - __first;
        assert(__n > 0);

        // The size of the source range is unknown. Use non-zero size to create the view.
        // permutation_view_simple access is controlled by the map range view.
        auto res_src = this->operator()(__first.base(), __first.base() + 1 /*source size*/);

        auto rng = __get_permutation_view(res_src.all_view(), __first.map(), __n);

        return __range_holder<decltype(rng)>{rng};
    }

    //specialization for permutation discard iterator
    template <typename _Map>
    auto
    operator()(oneapi::dpl::permutation_iterator<oneapi::dpl::discard_iterator, _Map> __first,
               oneapi::dpl::permutation_iterator<oneapi::dpl::discard_iterator, _Map> __last)
        -> __range_holder<oneapi::dpl::__ranges::permutation_discard_view>
    {
        auto __n = __last - __first;
        assert(__n > 0);

        auto rng = oneapi::dpl::__ranges::permutation_discard_view(__n);

        return __range_holder<decltype(rng)>{rng};
    }

    // for raw pointers and direct pass objects (for example, counting_iterator, iterator of USM-containers)
    template <typename _Iter>
    typename ::std::enable_if<is_passed_directly_it<_Iter>::value,
                              __range_holder<oneapi::dpl::__ranges::guard_view<_Iter>>>::type
    operator()(_Iter __first, _Iter __last)
    {
        assert(__first < __last);
        return __range_holder<oneapi::dpl::__ranges::guard_view<_Iter>>{
            oneapi::dpl::__ranges::guard_view<_Iter>{__first, __last - __first}};
    }

    //specialization for hetero iterator
    template <typename _Iter>
    auto
    operator()(_Iter __first, _Iter __last) ->
        typename ::std::enable_if<is_hetero_it<_Iter>::value,
                                  __range_holder<oneapi::dpl::__ranges::all_view<val_t<_Iter>, AccMode>>>::type
    {
        assert(__first < __last);
        using value_type = val_t<_Iter>;

        const auto __offset = __first - oneapi::dpl::begin(__first.get_buffer());
        const auto __size = __dpl_sycl::__get_buffer_size(__first.get_buffer());
        const auto __n = ::std::min(decltype(__size)(__last - __first), __size);
        assert(__offset + __n <= __size);

        return __range_holder<oneapi::dpl::__ranges::all_view<value_type, AccMode>>{
            oneapi::dpl::__ranges::all_view<value_type, AccMode>(__first.get_buffer() /* buffer */,
                                                                 __offset /* offset*/, __n /* size*/)};
    }

    //specialization for a host iterator
    template <typename _Iter>
    auto
    operator()(_Iter __first, _Iter __last) ->
        typename ::std::enable_if<is_temp_buff<_Iter>::value && !is_zip<_Iter>::value && !is_permutation<_Iter>::value,
                                  __buffer_holder<val_t<_Iter>, AccMode>>::type
    {
        static_assert(!oneapi::dpl::__internal::is_const_iterator<_Iter>::value || AccMode == sycl::access::mode::read,
                      "Should be non-const iterator for a modifying algorithm.");

        assert(__first < __last);

        using copy_direct_tag = ::std::integral_constant<bool, AccMode == sycl::access::mode::read_write ||
                                                                   AccMode == sycl::access::mode::read>;
        using copy_back_tag = ::std::integral_constant<bool, AccMode == sycl::access::mode::read_write ||
                                                                 AccMode == sycl::access::mode::write>;

        auto buf = copy_direct(__first, __last, copy_direct_tag());
        buf = copy_back(__first, buf, copy_back_tag());

        using BufferType = oneapi::dpl::__internal::__lifetime_keeper<decltype(buf)>;
        m_buffers.push_back(::std::make_shared<BufferType>(buf));

        return __buffer_holder<val_t<_Iter>, AccMode>{buf};
    }
};

} // namespace __ranges
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UTILS_RANGES_SYCL_H
