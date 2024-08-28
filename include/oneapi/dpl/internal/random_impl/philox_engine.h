// -*- C++ -*-
//===-- philox-engine.h ------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Public header file provides implementation for Philox Engine

#ifndef _ONEDPL_PHILOX_ENGINE_H
#define _ONEDPL_PHILOX_ENGINE_H

#include "random_common.h"
#include "counter_based_engines_stuff.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{

template <typename _UIntType, ::std::size_t _w, ::std::size_t _n, ::std::size_t _r,
          internal::element_type_t<_UIntType>... _consts>
class philox_engine;

template <class _CharT, class _Traits, typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r,
          __UIntType... __consts>
::std::basic_ostream<_CharT, _Traits>&
operator<<(::std::basic_ostream<_CharT, _Traits>&, const philox_engine<__UIntType, __w, __n, __r, __consts...>&);

template <typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r, __UIntType... __consts>
const sycl::stream&
operator<<(const sycl::stream&, const philox_engine<__UIntType, __w, __n, __r, __consts...>&);

template <class _CharT, class _Traits, typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r,
          __UIntType... __consts>
::std::basic_istream<_CharT, _Traits>&
operator>>(::std::basic_istream<_CharT, _Traits>&, philox_engine<__UIntType, __w, __n, __r, __consts...>&);

template <typename _UIntType, ::std::size_t _w, ::std::size_t _n, ::std::size_t _r,
          internal::element_type_t<_UIntType>... _consts>
class philox_engine
{
    /* The size of the consts arrays */
    static constexpr ::std::size_t array_size = _n / 2;

  public:
    /* Types */
    using result_type = _UIntType;
    using scalar_type = internal::element_type_t<result_type>;

    /* Engine characteristics */
    static constexpr ::std::size_t word_size = _w;
    static constexpr ::std::size_t word_count = _n;
    static constexpr ::std::size_t round_count = _r;

    static_assert(_n == 2 || _n == 4, "_n must be 2 or 4");
    static_assert(sizeof...(_consts) == _n, "the amount of _consts must be equal to _n");
    static_assert(_r > 0, "_r must be more than 0");
    static_assert(_w > 0 && _w <= ::std::numeric_limits<scalar_type>::digits,
                  "_w must satisfy 0 < _w < ::std::numeric_limits<_UIntType>::digits");
    static_assert(::std::numeric_limits<scalar_type>::digits <= 64,
                  "size of the scalar _UIntType (in case of sycl::vec<T, N> the size of T) must be less than 64 bits");
    static_assert(::std::is_unsigned_v<scalar_type>, "_UIntType must be unsigned type or vector of unsigned types");

    static constexpr ::std::array<scalar_type, array_size> multipliers =
        internal::experimental::get_even_array_from_tuple<scalar_type>(::std::make_tuple(_consts...),
                                                                       ::std::make_index_sequence<array_size>{});
    static constexpr ::std::array<scalar_type, array_size> round_consts =
        internal::experimental::get_odd_array_from_tuple<scalar_type>(::std::make_tuple(_consts...),
                                                                      ::std::make_index_sequence<array_size>{});
    static constexpr scalar_type
    min()
    {
        return 0;
    }
    static constexpr scalar_type
    max()
    {
        return ::std::numeric_limits<scalar_type>::max() & in_mask;
    }
    static constexpr scalar_type default_seed = 20111115u;

    /* Constructors and seeding functions */
    philox_engine() : philox_engine(default_seed) {}
    explicit philox_engine(scalar_type __seed) { seed(__seed); }
    void
    seed(scalar_type __seed = default_seed)
    {
        seed_internal({__seed & in_mask});
    }

    /* Set the state to arbitrary position */
    void
    set_counter(const ::std::array<scalar_type, word_count>& __counter)
    {
        auto __end = __counter.end();
        for (::std::size_t __i = 0; __i < word_count; __i++)
        {
            // all counters are set in reverse order
            state_.X[__i] = (*--__end) & in_mask;
        }
    }

    /* Generating functions */
    result_type
    operator()()
    {
        result_type __ret = generate_internal<internal::type_traits_t<result_type>::num_elems>();
        return __ret;
    }
    /* operator () overload for result portion generation */
    result_type
    operator()(unsigned int __random_nums)
    {
        result_type __ret = generate_internal<internal::type_traits_t<result_type>::num_elems>(__random_nums);
        return __ret;
    }

    /* Shift the counter only forward relative to its current position */
    void
    discard(unsigned long long __z)
    {
        discard_internal(__z);
    }

    /* Equality operators */
    friend bool
    operator==(const philox_engine& __x, const philox_engine& __y)
    {
        if (!::std::equal(__x.state_.X.begin(), __x.state_.X.end(), __y.state_.X.begin()) ||
            !::std::equal(__x.state_.K.begin(), __x.state_.K.end(), __y.state_.K.begin()) ||
            !::std::equal(__x.state_.Y.begin(), __x.state_.Y.end(), __y.state_.Y.begin()) ||
            __x.state_.idx != __y.state_.idx)
        {
            return false;
        }
        return true;
    }
    friend bool
    operator!=(const philox_engine& __x, const philox_engine& __y)
    {
        return !(__x == __y);
    }

    /* Inserters and extractors */
    template <class _CharT, class _Traits, typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r,
              __UIntType... __consts>
    friend ::std::basic_ostream<_CharT, _Traits>&
    operator<<(::std::basic_ostream<_CharT, _Traits>&, const philox_engine<__UIntType, __w, __n, __r, __consts...>&);

    template <typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r, __UIntType... __consts>
    friend const sycl::stream&
    operator<<(const sycl::stream&, const philox_engine<__UIntType, __w, __n, __r, __consts...>&);

    template <class _CharT, class _Traits, typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r,
              __UIntType... __consts>
    friend ::std::basic_istream<_CharT, _Traits>&
    operator>>(::std::basic_istream<_CharT, _Traits>&, philox_engine<__UIntType, __w, __n, __r, __consts...>&);

  private:
    /* Internal generator state */
    struct state
    {
        ::std::array<scalar_type, word_count> X;     // counters
        ::std::array<scalar_type, word_count / 2> K; // keys
        ::std::array<scalar_type, word_count> Y;     // results
        scalar_type idx;                             // index
    } state_;

    /* Processing mask */
    static constexpr auto in_mask = internal::experimental::word_mask<scalar_type, word_size>;

    void
    seed_internal(::std::initializer_list<scalar_type> __seed)
    {
        auto __start = __seed.begin();
        auto __end = __seed.end();
        // all counters are set to zero
        for (::std::size_t __i = 0; __i < word_count; __i++)
        {
            state_.X[__i] = 0;
        }
        // keys are set as seed
        for (::std::size_t __i = 0; __i < (word_count / 2); __i++)
        {
            state_.K[__i] = (__start == __end) ? 0 : (*__start++) & in_mask;
        }
        // results are set to zero
        for (::std::size_t __i = 0; __i < word_count; __i++)
        {
            state_.Y[__i] = 0;
        }

        state_.idx = word_count;
    }

    /* Increment counter by 1 */
    void
    increase_counter_internal()
    {
        state_.X[0] = (state_.X[0] + 1) & in_mask;
        for (::std::size_t __i = 1; __i < word_count; ++__i)
        {
            if (state_.X[__i - 1])
            {
                return;
            }
            state_.X[__i] = (state_.X[__i] + 1) & in_mask;
        }
    }

    /* Increment counter by an arbitrary __z */
    void
    increase_counter_internal(unsigned long long __z)
    {
        unsigned long long __carry = 0;
        unsigned long long __ctr_inc = __z;

        for (::std::size_t __i = 0; __i < word_count; ++__i)
        {
            scalar_type __initial_x = state_.X[__i];
            state_.X[__i] = (__initial_x + __ctr_inc + __carry) & in_mask;

            __carry = 0;
            // check overflow of the current chunk
            if (state_.X[__i] < __initial_x)
            {
                __carry = 1;
            }

            //          select high chunk             shift for addition with the next counter chunk
            __ctr_inc = (__ctr_inc & (~in_mask)) >> (std::numeric_limits<unsigned long long>::digits - word_size);
        }
    }

    /* generate_internal() specified for sycl_vec output 
       and overload for result portion generation */
    template <unsigned int _N = 0>
    ::std::enable_if_t<(_N > 0), result_type>
    generate_internal(unsigned int __random_nums)
    {
        if (__random_nums >= _N)
            return operator()();

        result_type __loc_result;
        scalar_type __curr_idx = state_.idx;

        for (int __elm_count = 0; __elm_count < __random_nums; __elm_count++)
        {
            if (__curr_idx == word_count)
            { // empty buffer
                philox_kernel();
                increase_counter_internal();
                __curr_idx = 0;
            }
            __loc_result[__elm_count] = state_.Y[__curr_idx];
            __curr_idx++;
        }

        state_.idx = __curr_idx;

        return __loc_result;
    }

    /* generate_internal() specified for sycl_vec output */
    template <unsigned int _N = 0>
    ::std::enable_if_t<(_N > 0), result_type>
    generate_internal()
    {
        result_type __loc_result;
        scalar_type __curr_idx = state_.idx;

        for (int __elm_count = 0; __elm_count < _N; __elm_count++)
        {
            if (__curr_idx == word_count)
            { // empty buffer
                philox_kernel();
                increase_counter_internal();
                __curr_idx = 0;
            }
            __loc_result[__elm_count] = state_.Y[__curr_idx];
            __curr_idx++;
        }

        state_.idx = __curr_idx;

        return __loc_result;
    }

    /* generate_internal() specified for a scalar output */
    template <unsigned int _N = 0>
    ::std::enable_if_t<(_N == 0), result_type>
    generate_internal()
    {
        result_type __loc_result;
        scalar_type __curr_idx = state_.idx;

        if (__curr_idx == word_count)
        { // empty buffer
            philox_kernel();
            increase_counter_internal();
            __curr_idx = 0;
        }

        // There are already generated numbers in the buffer
        __loc_result = state_.Y[__curr_idx];
        state_.idx = ++__curr_idx;

        return __loc_result;
    }

    void
    discard_internal(unsigned long long __z)
    {
        scalar_type __curr_idx = state_.idx % word_count;
        unsigned long long __newridx = (__curr_idx + __z) % word_count;
        if (__newridx == 0)
        {
            __newridx = word_count;
        }

        // check if we can't simply iterate the index in the buffer
        if (__z >= word_count - state_.idx)
        {
            unsigned long long __counters_increment = __z / word_count;
            __counters_increment += ((__z % word_count) + __curr_idx) / word_count;

            if (state_.idx < word_count)
            {
                __counters_increment--;
            }

            increase_counter_internal(__counters_increment);

            if (__newridx < word_count)
            {
                philox_kernel();
                increase_counter_internal();
            }
        }

        state_.idx = __newridx;
    }

    /* Internal generation Philox kernel */
    void
    philox_kernel()
    {
        if constexpr (word_count == 2)
        {
            scalar_type __R0 = (state_.X[0]) & in_mask;
            scalar_type __L0 = (state_.X[1]) & in_mask;
            scalar_type __K0 = (state_.K[0]) & in_mask;
            for (::std::size_t __i = 0; __i < round_count; ++__i)
            {
                auto [__hi0, __lo0] = internal::experimental::mulhilo<scalar_type, word_size>(__R0, multipliers[0]);
                __R0 = __hi0 ^ __K0 ^ __L0;
                __L0 = __lo0;
                __K0 = (__K0 + round_consts[0]) & in_mask;
            }
            state_.Y[0] = __R0;
            state_.Y[1] = __L0;
        }
        else if constexpr (word_count == 4)
        {
            scalar_type __R0 = (state_.X[0]) & in_mask;
            scalar_type __L0 = (state_.X[1]) & in_mask;
            scalar_type __R1 = (state_.X[2]) & in_mask;
            scalar_type __L1 = (state_.X[3]) & in_mask;
            scalar_type __K0 = (state_.K[0]) & in_mask;
            scalar_type __K1 = (state_.K[1]) & in_mask;
            for (::std::size_t __i = 0; __i < round_count; ++__i)
            {
                auto [__hi0, __lo0] = internal::experimental::mulhilo<scalar_type, word_size>(__R0, multipliers[0]);
                auto [__hi1, __lo1] = internal::experimental::mulhilo<scalar_type, word_size>(__R1, multipliers[1]);
                __R0 = __hi1 ^ __L0 ^ __K0;
                __L0 = __lo1;
                __R1 = __hi0 ^ __L1 ^ __K1;
                __L1 = __lo0;
                __K0 = (__K0 + round_consts[0]) & in_mask;
                __K1 = (__K1 + round_consts[1]) & in_mask;
            }
            state_.Y[0] = __R0;
            state_.Y[1] = __L0;
            state_.Y[2] = __R1;
            state_.Y[3] = __L1;
        }
    }
};

template <class _CharT, class _Traits, typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r,
          __UIntType... __consts>
::std::basic_ostream<_CharT, _Traits>&
operator<<(::std::basic_ostream<_CharT, _Traits>& __os,
           const philox_engine<__UIntType, __w, __n, __r, __consts...>& __engine)
{
    internal::save_stream_flags<_CharT, _Traits> __flags(__os);

    __os.setf(::std::ios_base::dec | ::std::ios_base::left);
    _CharT __sp = __os.widen(' ');
    __os.fill(__sp);

    for (auto x_elm : __engine.state_.X)
    {
        __os << x_elm << __sp;
    }
    for (auto k_elm : __engine.state_.K)
    {
        __os << k_elm << __sp;
    }
    for (auto y_elm : __engine.state_.Y)
    {
        __os << y_elm << __sp;
    }
    __os << __engine.state_.idx;

    return __os;
}

template <typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r, __UIntType... __consts>
const sycl::stream&
operator<<(const sycl::stream& __os, const philox_engine<__UIntType, __w, __n, __r, __consts...>& __engine)
{
    for (auto __x_elm : __engine.state_.X)
    {
        __os << __x_elm << ' ';
    }
    for (auto __k_elm : __engine.state_.K)
    {
        __os << __k_elm << ' ';
    }
    for (auto __y_elm : __engine.state_.Y)
    {
        __os << __y_elm << ' ';
    }
    __os << __engine.state_.idx;

    return __os;
}

template <class _CharT, class _Traits, typename __UIntType, ::std::size_t __w, ::std::size_t __n, ::std::size_t __r,
          __UIntType... __consts>
::std::basic_istream<_CharT, _Traits>&
operator>>(::std::basic_istream<_CharT, _Traits>& __is, philox_engine<__UIntType, __w, __n, __r, __consts...>& __engine)
{
    internal::save_stream_flags<_CharT, _Traits> __flags(__is);

    __is.setf(::std::ios_base::dec);

    const ::std::size_t __state_size = 2 * __n + __n / 2 + 1;

    ::std::vector<__UIntType> __tmp_inp(__state_size);
    for (::std::size_t __i = 0; __i < __state_size; ++__i)
    {
        __is >> __tmp_inp[__i];
    }

    if (!__is.fail())
    {
        int __inp_itr = 0;
        for (::std::size_t __i = 0; __i < __n; ++__i, ++__inp_itr)
            __engine.state_.X[__i] = __tmp_inp[__inp_itr];
        for (::std::size_t __i = 0; __i < __n / 2; ++__i, ++__inp_itr)
            __engine.state_.K[__i] = __tmp_inp[__inp_itr];
        for (::std::size_t __i = 0; __i < __n; ++__i, ++__inp_itr)
            __engine.state_.Y[__i] = __tmp_inp[__inp_itr];
        __engine.state_.idx = __tmp_inp[__inp_itr];
    }

    return __is;
}

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PHILOX_ENGINE_H
