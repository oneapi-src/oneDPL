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
    /* shifts the counter only forward relative to its current position*/
    void discard(unsigned long long z) {
    ;        // auto oldridx = ridxref();
        // unsigned newridx = (z + oldridx) % word_count;

        // increase_counter_internal();

        // unsigned long long zll = z + oldridx - (!oldridx && newridx);
        // zll /= word_count;
        // zll += !oldridx;
        // result_type zctr = zll & in_mask;
        // result_type oldctr = get_counter_internal();
        // result_type newctr = (zctr - 1 + oldctr) & in_mask;
        // //set_counter_internal(state_, newctr);
        // if (newridx) {
        //     if (zctr)
        //         ;//(*this)(::std::begin(state_), ::std::begin(results_));
            
        // }
        // else if (newctr == 0) {
        //     newridx = word_count;
        // }

        // ridxref() = newridx;
    }
    
    // [ToDO] inserters and extractors

private:
    void increase_counter_internal() {
        state_[0] = (state_[0] + 1) & in_mask;
        for (size_t i = 1; i < n; ++i) {
            if (state_[i - 1]) {
                [[likely]] return;
            }
            state_[i] = (state_[i] + 1) & in_mask;
        }
    }

    result_type* operator()(result_type* out) {
        result_type curr_idx = ridxref();
        if(curr_idx % word_count == 0) { // empty buffer
            generate();
            increase_counter_internal();
            curr_idx = 0;
        }
        
        // There are already generated numebrs in the buffer
        *out = state_[n + n/2 + curr_idx];
        ridxref()=++curr_idx;

        return out;
    }
    using uint_types = std::tuple<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;
    using promotion_types = std::tuple<std::uint16_t, std::uint32_t, std::uint64_t, __uint128_t>;

    static constexpr std::size_t log2(std::size_t val)
    {
        return ((val <= 2) ? 1 : 1 + log2(val / 2));
    }

    static constexpr std::size_t ceil_log2(std::size_t val)
    {
        std::size_t additive = static_cast<std::size_t>(!std::__has_single_bit(val));

        return log2(val) + additive;
    }
    using counter_type = std::tuple_element_t<ceil_log2(w / CHAR_BIT), uint_types>;
    using promotion_type = std::tuple_element_t<ceil_log2(w / CHAR_BIT), promotion_types>;

    static constexpr counter_type counter_mask = ~counter_type(0) >> (sizeof(counter_type) * CHAR_BIT - w);
    static constexpr result_type result_mask = ~result_type(0) >> (sizeof(result_type) * CHAR_BIT - w);
        
    // Implement w-bit mulhilo with an 2w-wide integer.
    static std::pair<counter_type, counter_type> mulhilo(result_type a, result_type b)
    {
        constexpr std::size_t shift = std::numeric_limits<promotion_type>::digits - w;
        promotion_type promoted_a = a;
        promotion_type promoted_b = b;
        promotion_type result = promoted_a * promoted_b;
        counter_type mulhi = result >> shift;
        counter_type mullo = (result << shift) >> shift;
        return {mulhi, mullo};
    }


    void generate() {
        if constexpr (n == 4) {
                result_type R0 = (state_[0]) & in_mask; // X
                result_type L0 = (state_[1]) & in_mask; // X
                result_type R1 = (state_[2]) & in_mask; // X
                result_type L1 = (state_[3]) & in_mask; // X
                result_type K0 = (state_[4]) & in_mask; // K
                result_type K1 = (state_[5]) & in_mask; // K
                for (size_t i = 0; i < round_count; ++i) {
                    auto [hi0, lo0] = mulhilo(R0, multipliers[0]); // M0
                    auto [hi1, lo1] = mulhilo(R1, multipliers[1]); // M1
                    R0 = hi1 ^ L0 ^ K0; //X1, 
                    L0 = lo1;           //X2
                    R1 = hi0 ^ L1 ^ K1; //X3 
                    L1 = lo0;           //X0 
                    K0 = (K0 + round_consts[0]) & in_mask; // C0
                    K1 = (K1 + round_consts[1]) & in_mask; // C1
                }
                state_[6] = R0;  // Y0
                state_[7] = L0;  // Y1
                state_[8] = R1;  // Y2
                state_[9] = L1;  // Y3
        }
    }

};

} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PHILOX_ENGINE_H