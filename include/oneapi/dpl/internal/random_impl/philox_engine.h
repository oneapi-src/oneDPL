// -*- C++ -*-
//===-- philox-engine.h ------------------------------------------===//
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

template<typename UIntType, ::std::size_t w, ::std::size_t n, ::std::size_t r,  internal::element_type_t<UIntType> ...consts>
class philox_engine;

template<class CharT, class Traits, typename UIntType_, ::std::size_t w_, ::std::size_t n_, ::std::size_t r_, UIntType_... consts_>
::std::basic_ostream<CharT, Traits>& 
operator<<(::std::basic_ostream<CharT, Traits>&, const philox_engine<UIntType_, w_, n_, r_, consts_...>&);

template<typename UIntType_, ::std::size_t w_, ::std::size_t n_, ::std::size_t r_, UIntType_... consts_>
const sycl::stream&
operator<<(const sycl::stream&, const philox_engine<UIntType_, w_, n_, r_, consts_...>&);

template<class CharT, class Traits, typename UIntType_, ::std::size_t w_, ::std::size_t n_, ::std::size_t r_, UIntType_... consts_>
::std::basic_istream<CharT, Traits>& 
operator>>(::std::basic_istream<CharT, Traits>&, philox_engine<UIntType_, w_, n_, r_, consts_...>&);

template<typename UIntType, ::std::size_t w, ::std::size_t n, ::std::size_t r,
         internal::element_type_t<UIntType> ...consts>
class philox_engine
{
public:
    /* types */
    using result_type = UIntType;
    using scalar_type = internal::element_type_t<result_type>;

    /* engine characteristics */
    static constexpr ::std::size_t word_size = w;
    static constexpr ::std::size_t word_count = n;
    static constexpr ::std::size_t round_count = r;

private:
    static_assert(n == 2 || n == 4, "n must be 2 or 4");
    static_assert(sizeof...(consts) == n, "the amount of consts must be equal to n");
    static_assert(r > 0, "r must be more than 0");
    static_assert(w > 0 && w <= ::std::numeric_limits<scalar_type>::digits, "w must 0 < w < ::std::numeric_limits<UIntType>::digits");
    static_assert(::std::numeric_limits<scalar_type>::digits <= 64, "UIntType size must be less than 64 bits");
    static_assert(::std::is_unsigned_v<scalar_type>, "UIntType must be unsigned type or vector of unsigned types");

    /* Internal generator state */
    struct state {
        ::std::array<scalar_type, word_count> X;   // counters
        ::std::array<scalar_type, word_count/2> K; // keys
        ::std::array<scalar_type, word_count> Y;   // results
        scalar_type idx;                           // index
    } state_; 
  
    /* Processing mask */
    static constexpr auto in_mask = detail::word_mask<scalar_type, word_size>;
    static constexpr ::std::size_t array_size = word_count / 2;
    
    void seed_internal(::std::initializer_list<scalar_type> seed) {
        auto start = seed.begin();
        auto end = seed.end();
        // all counters are set to zero
        for(::std::size_t i = 0; i < word_count; i++) {
            state_.X[i] = 0;
        }
        // keys are set as seed
        for (::std::size_t i = 0; i < (word_count/2); i++) {
            state_.K[i] = (start == end) ? 0 : (*start++) & in_mask;
        }
        // results are set to zero
        for (::std::size_t i = 0; i < word_count; i++) {
            state_.Y[i] = 0;
        }

        state_.idx = word_count;
    }
    
    /* Increment counter by 1 */
    void increase_counter_internal() {
        state_.X[0] = (state_.X[0] + 1) & in_mask;
        for (::std::size_t i = 1; i < word_count; ++i) {
            if (state_.X[i - 1]) {
                [[likely]] return;
            }
            state_.X[i] = (state_.X[i] + 1) & in_mask;
        }
    }

    /* Increment counter by an arbitrary z */
    void increase_counter_internal(unsigned long long z) {
        unsigned long long carry = 0;
        unsigned long long ctr_inc = z;

        for (::std::size_t i = 0; i < word_count; ++i) {
            scalar_type initial_x = state_.X[i];
            state_.X[i] = (initial_x + ctr_inc + carry) & in_mask;

            carry = 0;
            // check overflow of the current chunk
            if(state_.X[i] < initial_x) {
                carry = 1;
            }

            //          select high chunk            shift for addition with the next counter chunk
            ctr_inc = (ctr_inc & (~in_mask)) >> (std::numeric_limits<unsigned long long>::digits - word_size);
        }
    }

    /* generate_internal() specified for sycl_vec output and overload for result portion generation */
    template <unsigned int _N = 0>
    ::std::enable_if_t<(_N > 0), result_type>
    generate_internal(unsigned int __random_nums) {

        if (__random_nums >= _N)
            return operator()();

        result_type tmp;
        scalar_type curr_idx = state_.idx;

        for (int elm_count = 0; elm_count < __random_nums; elm_count++) {
            if(curr_idx  == word_count) { // empty buffer
                philox_kernel();
                increase_counter_internal();
                curr_idx = 0;
            }
            tmp[elm_count] = state_.Y[curr_idx];
            curr_idx++;
        }
        
        state_.idx = curr_idx;

        return tmp;
    }
    
    /* generate_internal() specified for sycl_vec output */
    template <unsigned int _N = 0>
    ::std::enable_if_t<(_N > 0), result_type>
    generate_internal() {
        result_type tmp;
        scalar_type curr_idx = state_.idx;

        for (int elm_count = 0; elm_count < _N; elm_count++) {
            if(curr_idx  == word_count) { // empty buffer
                philox_kernel();
                increase_counter_internal();
                curr_idx = 0;
            }
            tmp[elm_count] = state_.Y[curr_idx];
            curr_idx++;
        }
        
        state_.idx = curr_idx;

        return tmp;
    }
    
    /* generate_internal() specified for a scalar output */
    template <unsigned int _N = 0>
    ::std::enable_if_t<(_N == 0), result_type>
    generate_internal() {
        result_type tmp;

        scalar_type curr_idx = state_.idx;
        if(curr_idx  == word_count) { // empty buffer
            philox_kernel();
            increase_counter_internal();
            curr_idx = 0;
        }
        
        // There are already generated numbers in the buffer
        tmp = state_.Y[curr_idx];
        state_.idx = ++curr_idx;

        return tmp;
    }

public:
    static constexpr ::std::array<scalar_type, array_size> multipliers =
        detail::get_even_array_from_tuple<scalar_type>(::std::make_tuple(consts...),
                                                    ::std::make_index_sequence<array_size>{});
    static constexpr ::std::array<scalar_type, array_size> round_consts =
        detail::get_odd_array_from_tuple<scalar_type>(::std::make_tuple(consts...),
                                                    ::std::make_index_sequence<array_size>{});
    static constexpr scalar_type min() { return 0; }
    static constexpr scalar_type max() { return ::std::numeric_limits<scalar_type>::max() & in_mask; }
    static constexpr scalar_type default_seed = 20111115u;

    /* Constructors and seeding functions */
    philox_engine() : philox_engine(default_seed) {}
    explicit philox_engine(scalar_type value) { seed(value); }
    void seed(scalar_type value = default_seed) { seed_internal({ value & in_mask }); }

    /* Set the state to arbitrary position */
    void set_counter(const ::std::array<scalar_type, word_count>& counter) {
        auto start = counter.begin();
        auto end = counter.end();
        for (::std::size_t i = 0; i < word_count; i++) {
            // all counters are set in everse order
            state_.X[i] = (start == end) ? 0 : (*--end) & in_mask;
        }
    }

    /* Generating functions */
    result_type operator()() {
        result_type ret = generate_internal<internal::type_traits_t<result_type>::num_elems>();
        return ret;
    }
    
    /* operator () overload for result portion generation */
    result_type operator()(unsigned int __random_nums) {
        result_type ret = generate_internal<internal::type_traits_t<result_type>::num_elems>(__random_nums);
        return ret;
    }

    /* Shift the counter only forward relative to its current position */
    void discard(unsigned long long z) {
        scalar_type curr_idx = state_.idx % word_count;
        unsigned long long newridx = (curr_idx + z) % word_count;
        if(newridx == 0) {
            newridx = word_count;
        }
        
        // otherwise, simply iterate the index in the buffer
        if(z >= word_count - state_.idx) {
            unsigned long long counters_increment = z / word_count;
            counters_increment += ((z % word_count) + curr_idx)/word_count;

            if(state_.idx < word_count) {
                counters_increment--;
            }

            increase_counter_internal(counters_increment);

            if(newridx < word_count) {
                philox_kernel();
                increase_counter_internal();
            }
        }

        state_.idx = newridx;
    }

    /* Equality operators */
    friend bool operator==(const philox_engine& x, const philox_engine& y) {
        if(!::std::equal(x.state_.X.begin(), x.state_.X.end(), y.state_.X.begin()) ||
           !::std::equal(x.state_.K.begin(), x.state_.K.end(), y.state_.K.begin()) ||
           !::std::equal(x.state_.Y.begin(), x.state_.Y.end(), y.state_.Y.begin()) ||
           x.state_.idx != y.state_.idx) {
                return false;
        }
        return true;
    }
    friend bool
    operator!=(const philox_engine& __x, const philox_engine& __y)
    {
        return !(__x == __y);
    }

    /* inserters and extractors */
    template<class CharT, class Traits, typename UIntType_, ::std::size_t w_, ::std::size_t n_, ::std::size_t r_, UIntType_... consts_>
    friend ::std::basic_ostream<CharT, Traits>& 
    operator<<(::std::basic_ostream<CharT, Traits>&, const philox_engine<UIntType_, w_, n_, r_, consts_...>&);
    
    template<typename UIntType_, ::std::size_t w_, ::std::size_t n_, ::std::size_t r_, UIntType_... consts_>
    friend const sycl::stream&
    operator<<(const sycl::stream&, const philox_engine<UIntType_, w_, n_, r_, consts_...>&);

    template<class CharT, class Traits, typename UIntType_, ::std::size_t w_, ::std::size_t n_, ::std::size_t r_, UIntType_... consts_>
    friend ::std::basic_istream<CharT, Traits>& 
    operator>>(::std::basic_istream<CharT, Traits>&, philox_engine<UIntType_, w_, n_, r_, consts_...>&);

private:
    /* Internal generation Philox kernel */
    void philox_kernel() {
        if constexpr (word_count == 2) {
                scalar_type R0 = (state_.X[0]) & in_mask;
                scalar_type L0 = (state_.X[1]) & in_mask;
                scalar_type K0 = (state_.K[0]) & in_mask;
                for (::std::size_t i = 0; i < round_count; ++i) {
                    auto [hi0, lo0] = detail::mulhilo<scalar_type, word_size>(R0, multipliers[0]);
                    R0 = hi0 ^ K0 ^ L0;
                    L0 = lo0;
                    K0 = (K0 + round_consts[0]) & in_mask;
                }
                state_.Y[0] = R0;
                state_.Y[1] = L0;
        }
        else if constexpr (word_count == 4) {
                scalar_type R0 = (state_.X[0]) & in_mask;
                scalar_type L0 = (state_.X[1]) & in_mask;
                scalar_type R1 = (state_.X[2]) & in_mask;
                scalar_type L1 = (state_.X[3]) & in_mask;
                scalar_type K0 = (state_.K[0]) & in_mask;
                scalar_type K1 = (state_.K[1]) & in_mask;
                for (::std::size_t i = 0; i < round_count; ++i) {
                    auto [hi0, lo0] = detail::mulhilo<scalar_type, word_size>(R0, multipliers[0]);
                    auto [hi1, lo1] = detail::mulhilo<scalar_type, word_size>(R1, multipliers[1]);
                    R0 = hi1 ^ L0 ^ K0;
                    L0 = lo1;
                    R1 = hi0 ^ L1 ^ K1; 
                    L1 = lo0; 
                    K0 = (K0 + round_consts[0]) & in_mask;
                    K1 = (K1 + round_consts[1]) & in_mask;
                }
                state_.Y[0] = R0;
                state_.Y[1] = L0;
                state_.Y[2] = R1;
                state_.Y[3] = L1;
        }
    }
};

template<class CharT, class Traits, typename UIntType, ::std::size_t w, ::std::size_t n, ::std::size_t r, UIntType... consts>
::std::basic_ostream<CharT, Traits>& 
operator<<(::std::basic_ostream<CharT, Traits>& os, const philox_engine<UIntType, w, n, r, consts...>& engine) {
    internal::save_stream_flags<CharT, Traits> __flags(os);

    os.setf(::std::ios_base::dec | ::std::ios_base::left);
    CharT sp = os.widen(' ');
    os.fill(sp);

    for(auto x_elm: engine.state_.X) {
        os << x_elm << sp;
    }
    for(auto k_elm: engine.state_.K) {
        os << k_elm << sp;
    }
    for(auto y_elm: engine.state_.Y) {
        os << y_elm << sp;
    }
    os << engine.state_.idx;
    
    return os;
}

template<typename UIntType, ::std::size_t w, ::std::size_t n, ::std::size_t r, UIntType... consts>
const sycl::stream&
operator<<(const sycl::stream& os, const philox_engine<UIntType, w, n, r, consts...>& engine) {
    for(auto x_elm: engine.state_.X) {
        os << x_elm << ' ';
    }
    for(auto k_elm: engine.state_.K) {
        os << k_elm << ' ';
    }
    for(auto y_elm: engine.state_.Y) {
        os << y_elm << ' ';
    }
    os << engine.state_.idx;
    
    return os;
}

template<class CharT, class Traits, typename UIntType, ::std::size_t w, ::std::size_t n, ::std::size_t r, UIntType... consts>
::std::basic_istream<CharT, Traits>& 
operator>>(::std::basic_istream<CharT, Traits>& is, philox_engine<UIntType, w, n, r, consts...>& engine) {
    internal::save_stream_flags<CharT, Traits> __flags(is);

    is.setf(::std::ios_base::dec);

    const ::std::size_t state_size = 2*n + n/2 + 1;

    ::std::vector<UIntType> tmp_inp(state_size);
    for (::std::size_t i = 0; i < state_size; ++i) {
        is >> tmp_inp[i];
    }

    if (!is.fail())
    {
        int inp_itr = 0;
        for (::std::size_t i = 0; i < n; ++i, ++inp_itr)
            engine.state_.X[i] = tmp_inp[inp_itr];
        for (::std::size_t i = 0; i < n/2; ++i, ++inp_itr)
            engine.state_.K[i] = tmp_inp[inp_itr];
        for (::std::size_t i = 0; i < n; ++i, ++inp_itr)
            engine.state_.Y[i] = tmp_inp[inp_itr];
        engine.state_.idx = tmp_inp[inp_itr];
    }
    
    return is;
}

} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PHILOX_ENGINE_H
