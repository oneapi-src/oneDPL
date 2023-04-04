// -*- C++ -*-
//===-- discard_block_engine.h ---------------------------------------------===//
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
// Public header file provides implementation for Discard Block Engine

#ifndef _ONEDPL_DISCARD_BLOCK_ENGINE_H
#define _ONEDPL_DISCARD_BLOCK_ENGINE_H

#include <cstddef>
#include <utility>
#include <type_traits>

#include "random_common.h"

namespace oneapi
{
namespace dpl
{

template <class _Engine, ::std::size_t _P, ::std::size_t _R>
class discard_block_engine
{
  public:
    // Engine types
    using result_type = typename _Engine::result_type;
    using scalar_type = internal::element_type_t<result_type>;

    // Engine characteristics
    static constexpr ::std::size_t block_size = _P;
    static constexpr ::std::size_t used_block = _R;
    static constexpr scalar_type
    min()
    {
        return _Engine::min();
    }
    static constexpr scalar_type
    max()
    {
        return _Engine::max();
    }

    // Constructors
    discard_block_engine() : n_(0) {}

    explicit discard_block_engine(const _Engine& __e) : engine_(__e), n_(0) {}
    explicit discard_block_engine(_Engine&& __e) : engine_(::std::move(__e)), n_(0) {}
    explicit discard_block_engine(scalar_type __seed, unsigned long long __offset = 0)
    {
        engine_.seed(__seed);

        // Discard offset
        discard(__offset);
    }

    // Seeding function
    void
    seed()
    {
        n_ = 0;
        engine_.seed();
    }

    void
    seed(scalar_type __seed)
    {
        n_ = 0;
        engine_.seed(__seed);
    }

    // Discard procedure
    void
    discard(unsigned long long __num_to_skip)
    {
        if (!__num_to_skip)
            return;

        if (__num_to_skip < (used_block - n_))
        {
            n_ += __num_to_skip;
            engine_.discard(__num_to_skip);
        }
        else
        {
            unsigned long long __n_skip =
                __num_to_skip + static_cast<unsigned long long>((__num_to_skip + n_) / used_block) *
                                    static_cast<unsigned long long>(block_size - used_block);
            // Check the overflow case
            if (__n_skip >= __num_to_skip)
            {
                n_ = (__num_to_skip - (used_block - n_)) % used_block;
                engine_.discard(__n_skip);
            }
            else
            {
                for (; __num_to_skip > 0; --__num_to_skip)
                    operator()();
            }
        }
    }

    // operator () returns bits of engine recurrence
    result_type
    operator()()
    {
        return generate_internal<internal::type_traits_t<result_type>::num_elems>();
    }

    // operator () overload for result portion generation
    result_type
    operator()(unsigned int __random_nums)
    {
        return generate_internal<internal::type_traits_t<result_type>::num_elems>(__random_nums);
    }

    // Property function
    const _Engine&
    base() const noexcept
    {
        return engine_;
    }

  private:
    // Static asserts
    static_assert((0 < _R) && (_R <= _P), "oneapi::dpl::discard_block_engine. Error: unsupported parameters");

    // Function for state adjustment
    template <int _N>
    typename ::std::enable_if<(_N == 0), scalar_type>::type
    generate_internal_scalar()
    {
        if (n_ >= used_block)
        {
            engine_.discard(static_cast<unsigned long long>(block_size - used_block));
            n_ = 0;
        }
        ++n_;
        return engine_();
    };

    template <int N>
    typename ::std::enable_if<(N > 0), scalar_type>::type
    generate_internal_scalar()
    {
        if (n_ >= used_block)
        {
            engine_.discard(static_cast<unsigned long long>(block_size - used_block));
            n_ = 0;
        }
        ++n_;
        return static_cast<scalar_type>(engine_(1u)[0]);
    };

    // Generate implementation
    template <int _N>
    typename ::std::enable_if<(_N == 0), result_type>::type
    generate_internal()
    {
        return generate_internal_scalar<internal::type_traits_t<result_type>::num_elems>();
    }

    template <int _N>
    typename ::std::enable_if<(_N > 0), result_type>::type
    generate_internal()
    {
        result_type __res;
        if (static_cast<::std::size_t>(_N) < (used_block - n_))
        {
            __res = engine_();
            n_ += static_cast<::std::size_t>(_N);
        }
        else
        {
            for (int __i = 0; __i < _N; ++__i)
            {
                __res[__i] = generate_internal_scalar<internal::type_traits_t<result_type>::num_elems>();
            }
        }
        return __res;
    }

    template <int _N>
    typename ::std::enable_if<(_N > 0), result_type>::type
    generate_internal(unsigned int __random_nums)
    {
        if (__random_nums >= _N)
            return operator()();

        result_type __part_vec;

        for (unsigned int __i = 0; __i < __random_nums; ++__i)
        {
            __part_vec[__i] = generate_internal_scalar<internal::type_traits_t<result_type>::num_elems>();
        }

        return __part_vec;
    }

    _Engine engine_;
    ::std::size_t n_ = 0;
};

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_DISCARD_BLOCK_ENGINE_H
