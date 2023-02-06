// -*- C++ -*-
//===-- geometric_distribution.h ------------------------------------------===//
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
// Public header file provides implementation for Geometric Distribution

#ifndef _ONEDPL_GEOMETRIC_DISTRIBUTION_H
#define _ONEDPL_GEOMETRIC_DISTRIBUTION_H

namespace oneapi
{
namespace dpl
{
template <class _IntType = int>
class geometric_distribution
{
  public:
    // Distribution types
    using result_type = _IntType;
    using scalar_type = internal::element_type_t<_IntType>;
    class param_type
    {
      public:
        using distribution_type = geometric_distribution<result_type>;
        param_type() : param_type(0.5) {}
        explicit param_type(double p) : p_(p) {}
        double
        p() const
        {
            return p_;
        }
        friend bool
        operator==(const param_type& p1, const param_type& p2)
        {
            return p1.p_ == p2.p_;
        }
        friend bool
        operator!=(const param_type& p1, const param_type& p2)
        {
            return !(p1 == p2);
        }

      private:
        double p_;
    };

    // Constructors
    geometric_distribution() : geometric_distribution(0.5) {}
    explicit geometric_distribution(double __p) : p_(__p) {}
    explicit geometric_distribution(const param_type& __params) : p_(__params.p()) {}

    // Reset function
    void
    reset()
    {
    }

    // Property functions
    double
    p() const
    {
        return p_;
    }

    param_type
    param() const
    {
        return param_type(p_);
    }

    void
    param(const param_type& __params)
    {
        p_ = __params.p();
    }

    scalar_type
    min() const
    {
        return scalar_type{};
    }

    scalar_type
    max() const
    {
        return std::numeric_limits<scalar_type>::max();
    }

    // Generate functions
    template <class _Engine>
    result_type
    operator()(_Engine& __engine)
    {
        return operator()<_Engine>(__engine, param_type(p_));
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params)
    {
        return generate<size_of_type_, _Engine>(__engine, __params);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, unsigned int __random_nums)
    {
        return operator()<_Engine>(__engine, param_type(p_), __random_nums);
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

    // Static asserts
    static_assert(::std::is_integral<scalar_type>::value,
                  "oneapi::dpl::geometric_distribution. Error: unsupported data type");

    // Distribution parameters
    double p_;

    // Implementation for generate function
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr != 0), result_type>::type
    generate(_Engine& __engine, const param_type& __params)
    {
        return generate_vec<_Ndistr, _Engine>(__engine, __params);
    }

    // Specialization of the scalar generation
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr == 0), result_type>::type
    generate(_Engine& __engine, const param_type& __params)
    {
        oneapi::dpl::uniform_real_distribution<double> __u;
        return sycl::floor(sycl::log(1.0 - __u(__engine)) / sycl::log(1.0 - __params.p()));
    }

    // Specialization of the vector generation with size = [1; 2; 3]
    template <int __N, class _Engine>
    typename ::std::enable_if<(__N <= 3), result_type>::type
    generate_vec(_Engine& __engine, const param_type& __params)
    {
        return generate_n_elems<_Engine>(__engine, __params, __N);
    }

    // Specialization of the vector generation with size = [4; 8; 16]
    template <int __N, class _Engine>
    typename ::std::enable_if<(__N > 3), result_type>::type
    generate_vec(_Engine& __engine, const param_type& __params)
    {
        oneapi::dpl::uniform_real_distribution<sycl::vec<double, __N>> __distr;
        sycl::vec<double, __N> __u = __distr(__engine);
        sycl::vec<double, __N> __res_double = sycl::floor(sycl::log(1.0 - __u) / sycl::log(1.0 - __params.p()));
        result_type __res = __res_double.template convert<scalar_type, sycl::rounding_mode::rtz>();
        return __res;
    }

    // Implementation for the N vector's elements generation
    template <class _Engine>
    result_type
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res;
        oneapi::dpl::uniform_real_distribution<double> __u;
        double __tmp = sycl::log(1.0 - __params.p());
        for (int i = 0; i < __N; i++)
            __res[i] = sycl::floor(sycl::log(1.0 - __u(__engine)) / __tmp);
        return __res;
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

        __part_vec = generate_n_elems(__engine, __params, __N);
        return __part_vec;
    }
};
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_GEOMETRIC_DISTRIBUTION_H
