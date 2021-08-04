// -*- C++ -*-
//===-- lognormal_distribution.h ------------------------------------------===//
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
// Public header file provides implementation for Weibull Distribution

#ifndef _ONEDPL_lognormal_DISTRIBUTION
#define _ONEDPL_lognormal_DISTRIBUTION

namespace oneapi
{
namespace dpl
{
template <class _RealType = double>
class lognormal_distribution
{
  public:
    // Distribution types
    using result_type = _RealType;
    using scalar_type = internal::element_type_t<_RealType>;

    struct param_type
    {
        param_type() : param_type(scalar_type{0.0}) {}
        param_type(scalar_type __mean, scalar_type __stddev = scalar_type{1.0}) : mean(__mean), stddev(__stddev) {}
        scalar_type mean;
        scalar_type stddev;
    };

    // Constructors
    lognormal_distribution() : lognormal_distribution(scalar_type{0.0}) {}
    explicit lognormal_distribution(scalar_type __mean, scalar_type __stddev = scalar_type{1.0}) : mean_(__mean), stddev_(__stddev), 
                nd_(mean_, stddev_) {}
    explicit lognormal_distribution(const param_type& __params) : mean_(__params.mean), stddev_(__params.stddev), 
                nd_(mean_, stddev_) {}

    // Reset function
    void
    reset()
    {
        nd_.reset();
    }

    // Property functions
    scalar_type
    mean() const
    {
        return mean_;
    }

    scalar_type
    stddev() const
    {
        return stddev_;
    }

    param_type
    param() const
    {
        return param_type(mean_, stddev_);
    }

    void
    param(const param_type& __param)
    {
        mean_ = __param.mean;
        stddev_ = __param.stddev;
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
        return operator()<_Engine>(__engine, param_type(mean_, stddev_));
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
        return operator()<_Engine>(__engine, param_type(mean_, stddev_), __random_nums);
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

    using normal_distr =
        oneapi::dpl::normal_distribution<typename ::std::conditional<(size_of_type_ <= 3), scalar_type, result_type>::type>;

    // Static asserts
    static_assert(::std::is_floating_point<scalar_type>::value,
                  "oneapi::dpl::lognormal_distribution. Error: unsupported data type");

    // Distribution parameters
    scalar_type mean_;
    scalar_type stddev_;
    normal_distr nd_;

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
        return sycl::exp(nd_(__engine));
    }

    // Specialization of the vector generation with size = [1; 3]
    template <int __N, class _Engine>
    typename ::std::enable_if<(__N <= 3), result_type>::type
    generate_vec(_Engine& __engine, const param_type& __params)
    {
        result_type __res;
        for (int i = 0; i < __N; i++)
            __res[i] = sycl::exp(nd_(__engine));
        return __res;
    }

    // Specialization of the vector generation with size = [4; 8; 16]
    template <int __N, class _Engine>
    typename ::std::enable_if<(__N > 3), result_type>::type
    generate_vec(_Engine& __engine, const param_type& __params)
    {
        return sycl::exp(nd_(__engine));
    }

    // Implementation for the N vector's elements generation with size = [4; 8; 16]
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr > 3), result_type>::type
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res = sycl::exp(nd_(__engine, __N));
        return __res;
    }

    // Implementation for the N vector's elements generation with size = [1; 3]
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr <= 3), result_type>::type
    generate_n_elems(_Engine& __engine, const param_type& __params, unsigned int __N)
    {
        result_type __res;
        for (int i = 0; i < __N; i++)
            __res[i] = sycl::exp(nd_(__engine));
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

        __part_vec = generate_n_elems<_Ndistr, _Engine>(__engine, __params, __N);
        return __part_vec;
    }
};
} // namespace dpl
} // namespace oneapi

#endif // #ifndf _ONEDPL_lognormal_DISTRIBUTION