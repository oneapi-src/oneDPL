// -*- C++ -*-
//===-- uniform_int_distribution.h ----------------------------------------===//
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
// Public header file provides implementation for Uniform Int Distribution

#ifndef _ONEDPL_UNIFORM_INT_DISTRIBUTION
#define _ONEDPL_UNIFORM_INT_DISTRIBUTION

namespace oneapi
{
namespace dpl
{

template <class _IntType = int>
class uniform_int_distribution
{
  public:
    // Distribution types
    using result_type = _IntType;
    using scalar_type = internal::element_type_t<result_type>;
    class param_type
    {
      public:
        using distribution_type = uniform_int_distribution<result_type>;
        param_type() : param_type(scalar_type{0}) {}
        explicit param_type(scalar_type a, scalar_type b = ::std::numeric_limits<scalar_type>::max()) : a_(a), b_(b) {}
        scalar_type
        a() const
        {
            return a_;
        }
        scalar_type
        b() const
        {
            return b_;
        }
        friend bool
        operator==(const param_type& p1, const param_type& p2)
        {
            return p1.a_ == p2.a_ && p1.b_ == p2.b_;
        }
        friend bool
        operator!=(const param_type& p1, const param_type& p2)
        {
            return !(p1 == p2);
        }

      private:
        scalar_type a_;
        scalar_type b_;
    };

    // Constructors
    uniform_int_distribution() : uniform_int_distribution(scalar_type{0}) {}
    explicit uniform_int_distribution(scalar_type __a, scalar_type __b = ::std::numeric_limits<scalar_type>::max())
        : a_(__a), b_(__b)
    {
    }
    explicit uniform_int_distribution(const param_type& __params) : a_(__params.a()), b_(__params.b()) {}

    // Reset function
    void
    reset()
    {
    }

    // Property functions
    scalar_type
    a() const
    {
        return a_;
    }

    scalar_type
    b() const
    {
        return b_;
    }

    param_type
    param() const
    {
        return param_type(a_, b_);
    }

    void
    param(const param_type& __params)
    {
        a_ = __params.a();
        b_ = __params.b();
    }

    scalar_type
    min() const
    {
        return a();
    }

    scalar_type
    max() const
    {
        return b();
    }

    // Generate functions
    template <class _Engine>
    result_type
    operator()(_Engine& __engine)
    {
        return operator()<_Engine>(__engine, param_type(a_, b_));
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params)
    {
        return generate<size_of_type_, _Engine>(__engine, __params);
    }

    // Generation by portion
    template <class _Engine>
    result_type
    operator()(_Engine& __engine, unsigned int __random_nums)
    {
        return operator()<_Engine>(__engine, param_type(a_, b_), __random_nums);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params, unsigned int __random_nums)
    {
        return result_portion_internal<size_of_type_, _Engine>(__engine, __params, __random_nums);
    }

  private:
    // Size of type
    static constexpr int size_of_type_ = internal::type_traits_t<result_type>::num_elems;

    // Type of real distribution
    using RealType = typename ::std::conditional<size_of_type_ == 0, double, sycl::vec<double, size_of_type_>>::type;

    // Static asserts
    static_assert(::std::is_integral<scalar_type>::value,
                  "oneapi::dpl::uniform_int_distribution. Error: unsupported data type");

    // Distribution parameters
    scalar_type a_;
    scalar_type b_;

    // Real distribution for the conversion
    uniform_real_distribution<RealType> uniform_real_distribution_;

    // Implementation for generate function
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr != 0), result_type>::type
    generate(_Engine& __engine, const param_type& __params)
    {
        RealType __res = uniform_real_distribution_(
            __engine, typename uniform_real_distribution<RealType>::param_type(
                          static_cast<double>(__params.a()), static_cast<double>(__params.b()) + 1.0));

        result_type __res_ret;
        for (int __i = 0; __i < _Ndistr; ++__i)
            __res_ret[__i] = static_cast<scalar_type>(__res[__i]);

        return __res_ret;
    }

    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr == 0), result_type>::type
    generate(_Engine& __engine, const param_type& __params)
    {
        RealType __res = uniform_real_distribution_(
            __engine, typename uniform_real_distribution<RealType>::param_type(
                          static_cast<double>(__params.a()), static_cast<double>(__params.b()) + 1.0));

        return static_cast<scalar_type>(__res);
    }

    // Implementation for result_portion function
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr != 0), result_type>::type
    result_portion_internal(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __part_vec;
        if (__N == 0)
            return __part_vec;
        else if (__N >= _Ndistr)
            return operator()(__engine, __params);

        RealType __res =
            uniform_real_distribution_(__engine,
                                       typename uniform_real_distribution<RealType>::param_type(
                                           static_cast<double>(__params.a()), static_cast<double>(__params.b()) + 1.0),
                                       __N);

        for (unsigned int __i = 0; __i < __N; ++__i)
            __part_vec[__i] = static_cast<scalar_type>(__res[__i]);

        return __part_vec;
    }
};

} // namespace dpl
} // namespace oneapi

#endif // #ifndf _ONEDPL_UNIFORM_INT_DISTRIBUTION