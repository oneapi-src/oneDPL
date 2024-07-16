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

/* 
    using philox4x32 = philox_engine<uint_fast32_t, 32, 4, 10, 0xD2511F53,         0x9E3779B9,         0xCD9E8D57,         0xBB67AE85>;
    using philox4x64 = philox_engine<uint_fast64_t, 64, 4, 10, 0xD2E7470EE14C6C93, 0x9E3779B97F4A7C15, 0xCA5A826395121157, 0xBB67AE8584CAA73B>;
*/

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
    /* Internal state details */
    static constexpr size_t state_size = (n + n/2 + n + sizeof(UIntType)); // X +  K  + Y + idx
    using state = ::std::array<result_type, state_size>;
    state state_; // state: [counter_0,..., counter_n, key_0, ..., key_n/2-1];

    /* Processing mask */
    static constexpr auto in_mask = detail::fffmask<result_type, word_size>;
    static constexpr ::std::size_t array_size = n / 2;

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
    void seed(result_type value = default_seed) { /*seed_internal({ value & in_mask });*/ }
    template <class Sseq> void seed(Sseq& q) { ; /*q.generate(state_.begin(), state_.end());*/ }

    // Set the state to arbitrary position
    void set_counter(const ::std::array<result_type, n>& counter) {
        auto start = counter.begin();
        auto end = counter.end();
        for (size_t i = 0; i < word_count; i++) {
            state_[i] = (start == end) ? 0 : (*start++) & in_mask; // all counters are set
        }
    }

    // [ToDO] equality operator

    // generating functions
    result_type operator()() {
        result_type ret;
        // (*this)(&ret);
        return ret;
    }
    void discard(unsigned long long z) {
        ;
        // auto oldridx = ridxref();
        // unsigned newridx = (z + oldridx) % word_count;
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
        //     increase_counter_internal();
        // }
        // else if (newctr == 0) {
        //     newridx = word_count;
        // }

        // ridxref() = newridx;
    }
    
    // [ToDO] inserters and extractors

private:


    /* Return values type */
    using results = ::std::array<result_type, word_count>;
    results results_;

};

} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PHILOX_ENGINE_H