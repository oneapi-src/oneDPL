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

#include "counter_based_engines_stuff.h"

namespace oneapi
{
namespace dpl
{

template<typename UIntType, ::std::size_t w, ::std::size_t n, ::std::size_t r,
         UIntType ...consts>
class philox_engine
{
    static_assert(n == 2 || n == 4); // [ToDO] Extend to n = 8 and 16 
    static_assert(sizeof...(consts) == n);
    static_assert(r > 0);
    static_assert(w > 0 && w <= ::std::numeric_limits<UIntType>::digits);

public:
    //types
    using result_type = UIntType;
    using scalar_type = internal::element_type_t<result_type>;
    
    // engine characteristics
    static constexpr ::std::size_t word_size = w;
    static constexpr ::std::size_t word_count = n;
    static constexpr ::std::size_t round_count = r;
    static constexpr size_t period_counter_count = n;

private:
    /* 
    * Internal state details
    * [counter_0,..., counter_n, key_0, ..., key_(n/2-1), result_0, .. result_n, idx];
    */
    // struct state {
    //     result_type X[n];
    //     result_type K[n/2];
    //     result_type Y[n];
    //     result_type idx;
    // } state_; 
    static constexpr size_t state_size = (n + n/2 + n + 1); // X +  K  + Y + idx
    using state = ::std::array<result_type, state_size>;
    state state_;
  
    /* Processing mask */
    static constexpr auto in_mask = detail::fffmask<result_type, word_size>;
    static constexpr ::std::size_t array_size = n / 2;

    const auto& ridxref() const {
        return state_[state_size-1];
    }
    auto& ridxref() {
        return state_[state_size-1];
    }
    
    void seed_internal(::std::initializer_list<result_type> seed) {
        auto start = seed.begin();
        auto end = seed.end();
        size_t i = 0;
        for (i = 0; i < word_count; i++) {
            state_[i] = 0; // all counters are set to zero
        }
        for (; i < word_count+(word_count/2); i++) {
            state_[i] = (start == end) ? 0 : (*start++) & in_mask; // keys are set as seed
        }
        for (; i < word_count+(word_count/2)+(word_count); i++) {
            state_[i] = 0; // results are set to zero
        }
        ridxref() = word_count;
    }
    
    void increase_counter_internal() {
        state_[0] = (state_[0] + 1) & in_mask;
        for (size_t i = 1; i < n; ++i) {
            if (state_[i - 1]) {
                [[likely]] return;
            }
            state_[i] = (state_[i] + 1) & in_mask;
        }
    }

public:
    static constexpr ::std::array<result_type, array_size> multipliers =
        detail::get_even_array_from_tuple<UIntType>(::std::make_tuple(consts...),
                                                    ::std::make_index_sequence<array_size>{});
    static constexpr ::std::array<result_type, array_size> round_consts =
        detail::get_odd_array_from_tuple<UIntType>(::std::make_tuple(consts...),
                                                   ::std::make_index_sequence<array_size>{});
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return (2 << w) - 1; }
    static constexpr result_type default_seed = 20111115u;

    // constructors and seeding functions
    philox_engine() : philox_engine(default_seed) {}
    explicit philox_engine(result_type value) { seed(value); }
    template <class Sseq> explicit philox_engine(Sseq& q) { seed(q); }
    void seed(result_type value = default_seed) { seed_internal({ value & in_mask }); }
    template <class Sseq> void seed(Sseq& q) { ; /*q.generate(state_.begin(), state_.end());*/ }

    // Set the state to arbitrary position
    void set_counter(const ::std::array<result_type, n>& counter) {
        auto start = counter.begin();
        auto end = counter.end();
        for (size_t i = 0; i < word_count; i++) {
            state_[i] = (start == end) ? 0 : (*start++) & in_mask; // all counters are set
        }
    }

    // generating functions
    result_type operator()() {
        result_type ret;
        (*this)(&ret);
        return ret;
    }

    /* shift the counter only forward relative to its current position*/
    void discard(unsigned long long z) {
        result_type curr_idx = ridxref() % word_count;
        result_type newridx;

        newridx = (curr_idx + z) % (word_count);
        int counters_increment = z / word_count;

        for(int i = 0; i < counters_increment; i++)
            increase_counter_internal(); // rewrite with z

        if(newridx < word_count) {
            generate();
            increase_counter_internal();
        }

        ridxref() = newridx;
    }
    
    // [ToDO] inserters and extractors

private:

    result_type* operator()(result_type* out) {
        result_type curr_idx = ridxref();
        if(curr_idx  == word_count) { // empty buffer
            generate();
            increase_counter_internal();
            curr_idx = 0;
        }
        
        // There are already generated numebrs in the buffer
        *out = state_[n + n/2 + curr_idx];
        ridxref()=++curr_idx;

        return out;
    }
    // Implement w-bit mulhilo with an 2w-wide integer - returns
    // the w hi and w low bits of the 2w-bit product of a and b.
    static std::pair<result_type, result_type> mulhilo(result_type a, result_type b)
    {
        result_type res_hi, res_lo;
        /* multiplication fits standard types */
        if(word_size <= 32) {
            uint_fast64_t mult_result = (uint_fast64_t)a * (uint_fast64_t)b;
            res_hi = mult_result >> word_size;
            res_lo =  mult_result & in_mask;
        }
        /* pen-pencil multiplication by 32-bit chunks */
        else if(word_size == 64) {
            res_lo = a * b;

            result_type x0 = a & detail::fffmask<result_type, 32>;
            result_type x1 = a >> 32;
            result_type y0 = b & detail::fffmask<result_type, 32>;
            result_type y1 = b >> 32;

            result_type p11 = x1 * y1;
            result_type p01 = x0 * y1;
            result_type p10 = x1 * y0;
            result_type p00 = x0 * y0;

            // 64-bit product + two 32-bit values
            result_type middle = p10 + (p00 >> 32) + (p01 & detail::fffmask<result_type, 32>);

            // 64-bit product + two 32-bit values
            res_hi = p11 + (middle >> 32) + (p01 >> 32);
        }
        /* Other types are proceeds school multiplication - not supported yet */
        else {
            ; 
        }
        
        return { res_hi, res_lo };
    }

    void generate() {
        if constexpr (n == 2) {
            ;
        }
        else if constexpr (n == 4) {
                result_type R0 = (state_[0]) & in_mask;
                result_type L0 = (state_[1]) & in_mask;
                result_type R1 = (state_[2]) & in_mask;
                result_type L1 = (state_[3]) & in_mask;
                result_type K0 = (state_[4]) & in_mask;
                result_type K1 = (state_[5]) & in_mask;
                for (size_t i = 0; i < round_count; ++i) {
                    auto [hi0, lo0] = mulhilo(R0, multipliers[0]);
                    auto [hi1, lo1] = mulhilo(R1, multipliers[1]);
                    R0 = hi1 ^ L0 ^ K0;
                    L0 = lo1;
                    R1 = hi0 ^ L1 ^ K1; 
                    L1 = lo0; 
                    K0 = (K0 + round_consts[0]) & in_mask;
                    K1 = (K1 + round_consts[1]) & in_mask;
                }
                state_[6] = R0;
                state_[7] = L0;
                state_[8] = R1;
                state_[9] = L1;
        }
    }

};

} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PHILOX_ENGINE_H