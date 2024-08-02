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

private:
    /* Internal generator state */
    struct state {
        result_type X[word_count];   //counters
        result_type K[word_count/2]; //keys
        result_type Y[word_count];   //results
        result_type idx;             //index
    } state_; 
  
    /* Processing mask */
    static constexpr auto in_mask = detail::fffmask<result_type, word_size>;
    static constexpr ::std::size_t array_size = word_count / 2;
    
    void seed_internal(::std::initializer_list<result_type> seed) {
        auto start = seed.begin();
        auto end = seed.end();
        // all counters are set to zero
        for(size_t i = 0; i < word_count; i++) {
            state_.X[i] = 0;
        }
        // keys are set as seed
        for (size_t i = 0; i < (word_count/2); i++) {
            state_.K[i] = (start == end) ? 0 : (*start++) & in_mask;
        }
        // results are set to zero
        for (size_t i = 0; i < (word_count); i++) {
            state_.Y[i] = 0;
        }

        state_.idx = word_count;
    }
    
    void increase_counter_internal() {
        state_.X[0] = (state_.X[0] + 1) & in_mask;
        for (size_t i = 1; i < word_count; ++i) {
            if (state_.X[i - 1]) {
                [[likely]] return;
            }
            state_.X[i] = (state_.X[i] + 1) & in_mask;
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
    static constexpr result_type max() { return (2 << word_size) - 1; }
    static constexpr result_type default_seed = 20111115u;

    // constructors and seeding functions
    philox_engine() : philox_engine(default_seed) {}
    explicit philox_engine(result_type value) { seed(value); }
    template <class Sseq> explicit philox_engine(Sseq& q) { seed(q); }
    void seed(result_type value = default_seed) { seed_internal({ value & in_mask }); }
    template <class Sseq> void seed(Sseq& q) { ; /*q.generate(state_.begin(), state_.end());*/ }

    // Set the state to arbitrary position
    void set_counter(const ::std::array<result_type, word_count>& counter) {
        auto start = counter.begin();
        auto end = counter.end();
        for (size_t i = 0; i < word_count; i++) {
            state_.X[i] = (start == end) ? 0 : (*start++) & in_mask; // all counters are set
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
        result_type curr_idx = state_.idx % word_count;
        result_type newridx;

        newridx = (curr_idx + z) % (word_count);
        int counters_increment = z / word_count;

        for(int i = 0; i < counters_increment; i++)
            increase_counter_internal(); // rewrite with z

        if(newridx < word_count) {
            generate();
            increase_counter_internal();
        }

        state_.idx = newridx;
    }
    
    // [ToDO] inserters and extractors

private:

    result_type* operator()(result_type* out) {
        result_type curr_idx = state_.idx;
        if(curr_idx  == word_count) { // empty buffer
            generate();
            increase_counter_internal();
            curr_idx = 0;
        }
        
        // There are already generated numebrs in the buffer
        *out = state_.Y[curr_idx];
        state_.idx=++curr_idx;

        return out;
    }

    void generate() {
        if constexpr (word_count == 2) {
                result_type R0 = (state_.X[0]) & in_mask;
                result_type L0 = (state_.X[1]) & in_mask;
                result_type K0 = (state_.K[0]) & in_mask;
                for (size_t i = 0; i < round_count; ++i) {
                    auto [hi0, lo0] = detail::mulhilo<result_type, word_size>(R0, multipliers[0]);
                    R0 = hi0 ^ K0 ^ L0;
                    L0 = lo0;
                    K0 = (K0 + round_consts[0]) & in_mask;
                }
                state_.Y[0] = R0;
                state_.Y[1] = L0;
        }
        else if constexpr (word_count == 4) {
                result_type R0 = (state_.X[0]) & in_mask;
                result_type L0 = (state_.X[1]) & in_mask;
                result_type R1 = (state_.X[2]) & in_mask;
                result_type L1 = (state_.X[3]) & in_mask;
                result_type K0 = (state_.K[0]) & in_mask;
                result_type K1 = (state_.K[1]) & in_mask;
                for (size_t i = 0; i < round_count; ++i) {
                    auto [hi0, lo0] = detail::mulhilo<result_type, word_size>(R0, multipliers[0]);
                    auto [hi1, lo1] = detail::mulhilo<result_type, word_size>(R1, multipliers[1]);
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

} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PHILOX_ENGINE_H