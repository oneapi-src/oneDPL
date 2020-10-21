// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

namespace oneapi
{
namespace dpl
{
namespace __ranges
{

//A SYCL range over SYCL buffer
template <typename _T, cl::sycl::access::mode AccMode = cl::sycl::access::mode::read>
class all_view
{
    using return_t = typename ::std::conditional<AccMode == cl::sycl::access::mode::read, const _T, _T>::type;
    using accessor_t = cl::sycl::accessor<_T, 1, AccMode, cl::sycl::access::target::global_buffer,
                                          cl::sycl::access::placeholder::true_t>;
    using diff_type = typename ::std::iterator_traits<_T*>::difference_type;

  public:
    all_view(cl::sycl::buffer<_T, 1> __buf = cl::sycl::buffer<_T, 1>(0), diff_type __offset = 0, diff_type __n = 0)
        : m_acc(__buf, cl::sycl::range<1>(__n > 0 ? __n : __buf.get_count()), __offset)
    {
    }

    all_view(accessor_t acc) : m_acc(acc) {}

    return_t*
    begin() const
    {
        return m_acc.get_pointer();
    } //or “honest” iterator over an accessor and a sentinel

    return_t*
    end() const
    {
        return begin() + size();
    }
    return_t& operator[](diff_type i) const { return begin()[i]; }

    diff_type
    size() const
    {
        return m_acc.get_count();
    }
    bool
    empty() const
    {
        return size() == 0;
    }

    void
    require_access(cl::sycl::handler& cgh)
    {
        cgh.require(m_acc);
    } //non-standard method

  private:
    accessor_t m_acc;
};

struct all_view_fn
{
    template <typename _T, typename cl::sycl::access::mode AccMode = cl::sycl::access::mode::read>
    _PSTL_CONSTEXPR_FUN oneapi::dpl::__ranges::all_view<_T, AccMode>
    operator()(cl::sycl::buffer<_T, 1> __buf, typename ::std::iterator_traits<_T*>::difference_type __offset = 0,
               typename ::std::iterator_traits<_T*>::difference_type __n = 0) const
    {
        return oneapi::dpl::__ranges::all_view<_T, AccMode>(__buf, __offset, __n);
    }

    template <typename _T, typename cl::sycl::access::mode AccMode>
    _PSTL_CONSTEXPR_FUN oneapi::dpl::__ranges::all_view<_T, AccMode>
    operator()(oneapi::dpl::__ranges::all_view<_T, AccMode> __view) const
    {
        return __view;
    }
};

namespace views
{

_PSTL_CONSTEXPR_VAR all_view_fn all;
}

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
using is_hetero_it = oneapi::dpl::__par_backend_hetero::__internal::is_hetero_iterator<_Iter>;

template <typename _Iter>
using is_passed_directly_it = oneapi::dpl::__par_backend_hetero::__internal::is_passed_directly<_Iter>;

//struct for checking if it needs to create a temporay SYCL buffer or not

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
constexpr _Range
__get_first_range(_Range&& __rng, _Ranges&&...)
{
    return __rng;
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
__require_access_zip(cl::sycl::handler& __cgh, oneapi::dpl::__ranges::zip_view<_Ranges...>& __zip)
{
    const ::std::size_t __num_ranges = sizeof...(_Ranges);
    oneapi::dpl::__ranges::invoke(__zip.tuple(), _require_access_args<decltype(__cgh)>{__cgh},
                                  oneapi::dpl::__internal::__make_index_sequence<__num_ranges>());
}

//__require_access utility

inline void
__require_access(cl::sycl::handler& __cgh)
{
}

template <typename T, cl::sycl::access::mode M>
void
__require_access_range(cl::sycl::handler& __cgh, oneapi::dpl::__ranges::all_view<T, M>& sycl_view)
{
    sycl_view.require_access(__cgh);
}

template <typename... _Ranges>
void
__require_access_range(cl::sycl::handler& __cgh, zip_view<_Ranges...>& zip_rng)
{
    __require_access_zip(__cgh, zip_rng);
}

template <typename _BaseRange>
void
__require_access_range(cl::sycl::handler& __cgh, _BaseRange&)
{
}

template <typename _Range, typename... _Ranges>
void
__require_access(cl::sycl::handler& __cgh, _Range&& __rng, _Ranges&&... __rest)
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
    _PSTL_CONSTEXPR_FUN _R
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

// We have to keep sycl buffer intance here by sync reasons, at least in case of host iterators. SYCL runtime has sync
// in bufer desctruction and a sycl view instance keeps just placeholder accessor, not a buffer.
template <typename _T>
using buf_type = cl::sycl::buffer<_T, 1>;

template <typename _T, cl::sycl::access::mode AccMode>
struct __buffer_holder
{
    buf_type<_T> __buf;

    _PSTL_CONSTEXPR_FUN oneapi::dpl::__ranges::all_view<_T, AccMode>
    all_view() const
    {
        return oneapi::dpl::__ranges::all_view<_T, AccMode>(__buf);
    }
};

template <typename _T>
struct __buffer_wrap : public buf_type<_T>
{
    __buffer_wrap() : buf_type<_T>(0) {}

    template <cl::sycl::access::mode AccMode>
    __buffer_wrap&
    operator=(__buffer_holder<_T, AccMode> buf)
    {
        buf_type<_T>::operator=(buf.__buf);
        return *this;
    }
};

template <typename Iter>
struct __iter_types
{
    //TODO: we cannot use "base()" because the same method is used for iterators of the standard STL containers.
    using base_iter = Iter; //typename pipeline_base<Iter>::type;
    using value_type = val_t<base_iter>;

    //TODO: In case when a temporary buffer doesn't need we have to use a dummy type - "oneapi::dpl::internal::ignore_copyable, for example
    //For zip_iterator it makes sense to "collapse" a tuple keeper order to have exact number of temporary buffers for lifetime extending.
    using type = typename ::std::conditional<is_temp_buff<base_iter>::value, __buffer_wrap<value_type>,
                                             oneapi::dpl::internal::ignore_copyable>::type;
};

template <typename... Iters>
struct __iter_types<oneapi::dpl::zip_iterator<Iters...>>
{
    using type = oneapi::dpl::__internal::tuple<typename __iter_types<Iters>::type...>;
};

template <typename _Map, typename = void>
struct __map_type
{
    using type = oneapi::dpl::internal::ignore_copyable;
};

template <typename _Map>
struct __map_type<_Map, typename ::std::enable_if<is_map_iterator<_Map>::value>::type>
{
    using type = typename ::std::conditional<is_temp_buff<_Map>::value, __buffer_wrap<val_t<_Map>>,
                                             oneapi::dpl::internal::ignore_copyable>::type;
};

template <typename _It, typename _Map>
struct __iter_types<oneapi::dpl::permutation_iterator<_It, _Map>>
{
    using type1 = typename ::std::conditional<is_temp_buff<_It>::value, __buffer_wrap<val_t<_It>>,
                                              oneapi::dpl::internal::ignore_copyable>::type;
    using type2 = typename __map_type<_Map>::type;

    using type = oneapi::dpl::__internal::tuple<type1, type2>;
};

template <cl::sycl::access::mode AccMode, typename _Iterator>
struct __get_sycl_range
{
  private:
    //We have to keep sycl buffer(s) intance here by sync reasons; see __iter_types definition above
    typename __iter_types<_Iterator>::type m_keep;

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

    template <typename _Iter, typename _copy_direct_tag>
    buf_type<val_t<_Iter>>
    copy_direct(_Iter __first, _Iter __last, _copy_direct_tag)
    {
        //create a SYCL buffer and copy data [first, last) or create a empty SYCL buffer with size = (last - first)
        return oneapi::dpl::__internal::__invoke_if_else(
            _copy_direct_tag{}, [&]() { return cl::sycl::buffer<val_t<_Iter>, 1>(__first, __last); },
            [&]() { return cl::sycl::buffer<val_t<_Iter>, 1>(__last - __first); });
    }

    template <typename _F, typename _It, typename _DiffType>
    static auto
    gen_view(_F __f, _It __it, _DiffType __n) -> decltype(__f(__it, __it + __n))
    {
        return __f(__it, __it + __n);
    }

    template <typename _TupleType, typename _DiffType, ::std::size_t... _Ip>
    auto
    gen_zip_view(_TupleType __t, _DiffType __n, oneapi::dpl::__internal::__index_sequence<_Ip...>)
        -> decltype(oneapi::dpl::__ranges::make_zip_view(gen_view(*this, ::std::get<_Ip>(__t), __n).all_view()...))
    {
        auto tmp = oneapi::dpl::__internal::make_tuple(gen_view(*this, ::std::get<_Ip>(__t), __n)...);
        m_keep = tmp;
        return oneapi::dpl::__ranges::make_zip_view(::std::get<_Ip>(tmp).all_view()...);
    }

  public:
    //zip iterators

    template <typename... Iters>
    auto
    operator()(oneapi::dpl::zip_iterator<Iters...> __first, oneapi::dpl::zip_iterator<Iters...> __last) -> decltype(
        __range_holder<decltype(gen_zip_view(__first.base(), __last - __first,
                                             oneapi::dpl::__internal::__make_index_sequence<sizeof...(Iters)>()))>{
            gen_zip_view(__first.base(), __last - __first,
                         oneapi::dpl::__internal::__make_index_sequence<sizeof...(Iters)>())})
    {
        assert(__first < __last);

        const ::std::size_t __num_it = sizeof...(Iters);
        auto rng =
            gen_zip_view(__first.base(), __last - __first, oneapi::dpl::__internal::__make_index_sequence<__num_it>());
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
    //helper SFINAE utilities for a permutation_iterator support
    template <typename _Map, typename _Size>
    auto
    __get_it_map_view(_Map __m, _Size __n) ->
        typename ::std::enable_if<is_map_iterator<_Map>::value,
                                  decltype(::std::declval<__get_sycl_range<AccMode, _Iterator>>()(__m,
                                                                                                  __m + __n))>::type
    {
        return this->operator()(__m, __m + __n);
    }
    template <typename _Map, typename _Size>
    auto
    __get_it_map_view(_Map __m, _Size __n) -> typename ::std::enable_if<is_map_functor<_Map>::value, _Size>::type
    {
        return _Size(0);
    }
    template <typename _Map, typename _T>
    static auto
    __get_all_view(_Map __m, _T __t) ->
        typename ::std::enable_if<is_map_iterator<_Map>::value, decltype(__t.all_view())>::type
    {
        return __t.all_view();
    }
    template <typename _Map, typename _T>
    static auto
    __get_all_view(_Map __m, _T __t) -> typename ::std::enable_if<is_map_functor<_Map>::value, _Map>::type
    {
        return __m;
    }

  public:
    //specialization for permutation_iterator using sycl_iterator as source
    template <typename _It, typename _Map>
    auto
    operator()(oneapi::dpl::permutation_iterator<_It, _Map> __first,
               oneapi::dpl::permutation_iterator<_It, _Map> __last) ->
        typename ::std::enable_if<
            is_hetero_it<_It>::value,
            __range_holder<oneapi::dpl::__ranges::permutation_view_simple<
                decltype(::std::declval<__get_sycl_range<AccMode, _Iterator>>()(
                             __first.base(), __first.base() + __first.base().get_buffer().get_count())
                             .all_view()),
                decltype(__get_all_view(__first.map(), ::std::declval<__get_sycl_range<AccMode, _Iterator>>()
                                                           .__get_it_map_view(__first.map(), __last - __first)))>>>::
            type
    {
        auto __n = __last - __first;
        assert(__n > 0);

        auto res_src = this->operator()(__first.base(), __first.base() + __first.base().get_buffer().get_count());
        auto res_idx = __get_it_map_view(__first.map(), __n);

        auto rng = oneapi::dpl::__ranges::permutation_view_simple<decltype(res_src.all_view()),
                                                                  decltype(__get_all_view(__first.map(), res_idx))>{
            res_src.all_view(), __get_all_view(__first.map(), res_idx)};

        return __range_holder<decltype(rng)>{rng};
    }

    // TODO Add specialization for general case, e.g., permutation_iterator using host
    // or another fancy iterator.
    //specialization for permutation_iterator using USM pointer as source
    template <typename _It, typename _Map>
    auto
    operator()(oneapi::dpl::permutation_iterator<_It, _Map> __first,
               oneapi::dpl::permutation_iterator<_It, _Map> __last) ->
        typename ::std::enable_if<
            !is_hetero_it<_It>::value,
            __range_holder<oneapi::dpl::__ranges::permutation_view_simple<
                decltype(
                    ::std::declval<__get_sycl_range<AccMode, _Iterator>>()(__first.base(), __first.base()).all_view()),
                decltype(__get_all_view(__first.map(), ::std::declval<__get_sycl_range<AccMode, _Iterator>>()
                                                           .__get_it_map_view(__first.map(), __last - __first)))>>>::
            type
    {
        auto __n = __last - __first;
        assert(__n > 0);

        // The size of the source range is unknown. Use non-zero size to create the view.
        // permutation_view_simple access is controlled by the map range view.
        auto res_src = this->operator()(__first.base(), __first.base() + 1 /*source size*/);
        auto res_idx = __get_it_map_view(__first.map(), __n);

        auto rng = oneapi::dpl::__ranges::permutation_view_simple<decltype(res_src.all_view()),
                                                                  decltype(__get_all_view(__first.map(), res_idx))>{
            res_src.all_view(), __get_all_view(__first.map(), res_idx)};

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
        return __range_holder<oneapi::dpl::__ranges::all_view<value_type, AccMode>>{
            oneapi::dpl::__ranges::all_view<value_type, AccMode>(__first.get_buffer(),
                                                                 __first - oneapi::dpl::begin(__first.get_buffer()),
                                                                 __last - __first)}; //{buffer, offset, size}
    }

    //specialization for a host iterator
    template <typename _Iter>
    auto
    operator()(_Iter __first, _Iter __last) ->
        typename ::std::enable_if<is_temp_buff<_Iter>::value && !is_zip<_Iter>::value && !is_permutation<_Iter>::value,
                                  __buffer_holder<val_t<_Iter>, AccMode>>::type
    {
        static_assert(!oneapi::dpl::__internal::is_const_iterator<_Iter>::value ||
                          AccMode == cl::sycl::access::mode::read,
                      "Should be non-const iterator for a modifying algorithm.");

        assert(__first < __last);

        using copy_direct_tag = ::std::integral_constant<bool, AccMode == cl::sycl::access::mode::read_write ||
                                                                   AccMode == cl::sycl::access::mode::read>;
        using copy_back_tag = ::std::integral_constant<bool, AccMode == cl::sycl::access::mode::read_write ||
                                                                 AccMode == cl::sycl::access::mode::write>;

        auto buf = copy_direct(__first, __last, copy_direct_tag());
        buf = copy_back(__first, buf, copy_back_tag());

        return __buffer_holder<val_t<_Iter>, AccMode>{buf};
    }
};

} // namespace __ranges
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_UTILS_RANGES_SYCL_H */
