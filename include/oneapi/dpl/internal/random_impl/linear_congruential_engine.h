// -*- C++ -*-
//===-- linear_congruential_engine.h --------------------------------------===//
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
//
// Abstract:
//
// Public header file provides implementation for Linear Congruential Engine

#ifndef DPSTD_LINEAR_CONGRUENTIAL_ENGINE
#define DPSTD_LINEAR_CONGRUENTIAL_ENGINE

namespace oneapi
{
namespace dpl
{

template <class _UIntType, internal::element_type_t<_UIntType> _A, internal::element_type_t<_UIntType> _C,
          internal::element_type_t<_UIntType> _M>
class linear_congruential_engine
{
  public:
    // Engine types
    using result_type = _UIntType;
    using scalar_type = internal::element_type_t<result_type>;

    // Engine characteristics
    static constexpr scalar_type multiplier = _A;
    static constexpr scalar_type increment = _C;
    static constexpr scalar_type modulus = _M;
    static constexpr scalar_type
    min()
    {
        return (increment == static_cast<scalar_type>(0u)) ? static_cast<scalar_type>(1u) : static_cast<scalar_type>(0u);
    }
    static constexpr scalar_type
    max()
    {
        return modulus - static_cast<scalar_type>(1u);
    }
    static constexpr scalar_type default_seed = static_cast<scalar_type>(1u);

    // Constructors
    linear_congruential_engine() : linear_congruential_engine(default_seed){};

    explicit linear_congruential_engine(scalar_type __seed, unsigned long long __offset = 0)
    {
        init<internal::type_traits_t<result_type>::num_elems, increment>(__seed);
        discard(__offset);
    }

    // Seeding function
    void
    seed(scalar_type __seed = default_seed)
    {
        // Engine initalization
        init<internal::type_traits_t<result_type>::num_elems, increment>(__seed);
    }

    // Discard procedure
    void
    discard(unsigned long long __num_to_skip)
    {
        // Skipping sequence
        skip_seq<internal::type_traits_t<result_type>::num_elems, increment>(__num_to_skip);
    }

    // operator () returns bits of engine recurrence
    result_type
    operator()()
    {
        result_type __state_tmp = state_;
        unsigned long long __discard_num =
            (internal::type_traits_t<result_type>::num_elems == 0) ? 1 : internal::type_traits_t<result_type>::num_elems;
        discard(__discard_num);
        return __state_tmp;
    }

    // operator () overload for result portion generation
    result_type
    operator()(unsigned int __randoms_num)
    {
        return result_portion_internal<internal::type_traits_t<result_type>::num_elems>(__randoms_num);
    }

  private:
    // Static asserts
    static_assert(((_M == 0) || (_A < _M) && ( _C < _M)),
        "oneapi::std::linear_congruential_engine. Error: unsupported parameters");

    // Function for state adjustment
    scalar_type
    mod_scalar(scalar_type __state_input)
    {
        ::std::uint64_t __mult = multiplier, __mod = modulus, __inc = increment;
        return static_cast<scalar_type>((__mult * ::std::uint64_t(__state_input) + __inc) % __mod);
    }

    // Initialization function
    template <int _N = 0, scalar_type _INC = 0>
    typename ::std::enable_if<_N == 0>::type
    init(scalar_type __seed)
    {
        if ((increment % modulus == 0) && (__seed % modulus == 0))
        {
            state_ = default_seed;
        }
        else
        {
            state_ = __seed;
        }
        state_ = mod_scalar(state_);
    }

    template <int _N = 0, scalar_type _INC = 0>
    typename ::std::enable_if<(_N != 0)>::type
    init(scalar_type __seed)
    {
        if ((increment % modulus == 0) && (__seed % modulus == 0))
        {
            state_[0] = default_seed;
        }
        else
        {
            state_[0] = __seed;
        }

        state_[0] = mod_scalar(state_[0]);

        for (int __i = 1u; __i < _N; __i++)
            state_[__i] = mod_scalar(state_[__i - 1u]);
    }

    // Internal function which is used in discard procedure
    template <int _N = 0, scalar_type _INC = 0>
    typename ::std::enable_if<_N == 0>::type
    skip_seq(unsigned long long __num_to_skip)
    {
        for (unsigned long long __i = 0; __i < __num_to_skip; ++__i)
            state_ = mod_scalar(state_);
    }

    template <int _N = 0, scalar_type _INC = 0>
    typename ::std::enable_if<(_N == 1)>::type
    skip_seq(unsigned long long __num_to_skip)
    {
        for (unsigned long long __i = 0; __i < __num_to_skip; ++__i)
            state_[0] = mod_scalar(state_[0]);
    }

    template <int _N = 0, scalar_type _INC = 0>
    typename ::std::enable_if<(_N > 1)>::type
    skip_seq(unsigned long long __num_to_skip)
    {
        for (unsigned long long __i = 0; __i < __num_to_skip; ++__i)
        {
            for (int __j = 0; __j < (_N - 1); ++__j)
            {
                state_[__j] = state_[__j + 1];
            }
            state_[_N - 1] = mod_scalar(state_[_N - 2]);
        }
    }

    // result_portion implementation
    template <int _N>
    typename ::std::enable_if<(_N > 0), result_type>::type
    result_portion_internal(unsigned int __randoms_num)
    {
        result_type __part_vec;
        if (__randoms_num < 1)
            return __part_vec;

        unsigned int __num_to_gen = (__randoms_num > _N) ? _N : __randoms_num;
        for (unsigned int __i = 0; __i < __num_to_gen; ++__i)
            __part_vec[__i] = state_[__i];

        discard(__num_to_gen);
        return __part_vec;
    }

    result_type state_;
};

} // namespace dpl
} // namespace oneapi

#endif // ifndef DPSTD_LINEAR_CONGRUENTIAL_ENGINE