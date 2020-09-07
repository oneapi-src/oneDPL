// -*- C++ -*-
//===-- subtract_with_carry_engine.h --------------------------------------===//
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
// Public header file provides implementation for Subtract with Carry Engine

#ifndef DPSTD_SUBTRACT_WITH_CARRY_ENGINE
#define DPSTD_SUBTRACT_WITH_CARRY_ENGINE

namespace oneapi
{
namespace dpl
{

template <class _UIntType, size_t _W, size_t _S, size_t _R>
class subtract_with_carry_engine
{
  public:
    // Engine types
    using result_type = _UIntType;
    using scalar_type = internal::element_type_t<result_type>;

    // Engine characteristics
    static constexpr size_t word_size = _W;
    static constexpr size_t short_lag = _S;
    static constexpr size_t long_lag = _R;
    static constexpr scalar_type
    min()
    {
        return 0u;
    }
    static constexpr scalar_type
    max()
    {
        return (static_cast<scalar_type>(1u) << word_size) - static_cast<scalar_type>(1u);
    }
    static constexpr scalar_type default_seed = 19780503u;

    // Constructors
    subtract_with_carry_engine() : subtract_with_carry_engine(default_seed){};

    explicit subtract_with_carry_engine(scalar_type __seed, unsigned long long __offset = 0)
    {
        // Engine initalization
        init(__seed);

        // Discard offset
        discard(__offset);
    }

    // Seeding function
    void
    seed(scalar_type __seed = default_seed)
    {
        // Engine initalization
        init(__seed);
    }

    // Discard procedure
    void
    discard(unsigned long long __num_to_skip)
    {
        if (!__num_to_skip)
            return;

        for (; __num_to_skip > 0; --__num_to_skip)
            generate_internal_scalar();
    }

    // operator () returns bits of engine recurrence
    result_type
    operator()()
    {
        result_type res = generate_internal<internal::type_traits_t<result_type>::num_elems>();

        return res;
    }

    // operator () overload for result portion generation
    result_type
    operator()(unsigned int __randoms_num)
    {
        result_type __res = generate_internal<internal::type_traits_t<result_type>::num_elems>(__randoms_num);

        return __res;
    }

  private:
    // Static asserts
    static_assert((0 < _S) && (_S < _R),
        "oneapi::std::subtract_with_carry_engine. Error: unsupported parameters");

    static_assert((0 < _W) && (_W <= std::numeric_limits<scalar_type>::digits),
        "oneapi::std::subtract_with_carry_engine. Error: unsupported parameters");
    // Initialization function
    void
    init(scalar_type __seed)
    {
        linear_congruential_engine<scalar_type, 40014u, 0u, 2147483563u> __engine(__seed == 0 ? default_seed : __seed);

        for (size_t __i = 0; __i < long_lag; ++__i)
        {
            x_[__i] = static_cast<scalar_type>(__engine() & max());

            if (word_size > 32)
            {
                ::std::uint64_t __tmp = (::std::uint64_t(__engine()) << 32);
                x_[__i] = x_[__i] + __tmp;
            }

            x_[__i] &= max();
        }
        c_ = x_[long_lag - 1] == 0;
    }

    // Function for state adjustment
    scalar_type
    generate_internal_scalar()
    {
        int64_t __id = static_cast<int64_t>(i_);
        int64_t __id_1 = (__id - short_lag);
        if (__id_1 < 0)
            __id_1 += long_lag;
        ::std::int64_t __id_2 = __id;

        if (x_[__id_1] >= (x_[__id_2] + c_))
        {
            x_[__id] = x_[__id_1] - x_[__id_2] - c_;
            c_ = 0;
        }
        else
        {
            x_[i_] = max() - x_[__id_2] - c_ + x_[__id_1] + 1u;
            c_ = 1;
        }
        i_++;
        if (i_ >= long_lag)
            i_ = 0;
        return x_[__id];
    };

    // Generate implementation
    template <int _N>
    typename ::std::enable_if<(_N == 0), result_type>::type
    generate_internal()
    {
        return generate_internal_scalar();
    }


    template <int _N>
    typename ::std::enable_if<(_N > 0), result_type>::type
    generate_internal()
    {
        result_type __res;
        for (int __i = 0; __i < _N; ++__i)
        {
            __res[__i] = generate_internal_scalar();
        }
        return __res;
    }

    template <int _N>
    typename ::std::enable_if<(_N > 0), result_type>::type
    generate_internal(unsigned int __randoms_num)
    {
        result_type __res;
        for (unsigned int __i = 0; __i < __randoms_num; ++__i)
        {
            __res[__i] = generate_internal_scalar();
        }

        return __res;
    }

    scalar_type x_[long_lag];
    scalar_type c_;
    size_t i_ = 0;
};

} // namespace dpl
} // namespace oneapi

#endif // ifndef DPSTD_SUBTRACT_WITH_CARRY_ENGINE
