// -*- C++ -*-
//===-- normal_distribution.h ---------------------------------------------===//
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
// Public header file provides implementation for Normal Distribution

#ifndef DPSTD_NORMAL_DISTRIBUTION
#define DPSTD_NORMAL_DISTRIBUTION

namespace oneapi
{
namespace dpl
{
namespace internal
{

static const dp_union_t gaussian_dp_table[2]{
    0x54442D18, 0x401921FB, // Pi * 2
    0x446D71C3, 0xC0874385, // ln(0.494065e-323) = -744.440072
};

static const sp_union_t gaussian_sp_table[2]{
    0x40C90FDB, // Pi * 2
    0xC2CE8ED0, // ln(0.14012984e-44) = -103.278929
};

} // namespace internal

template <class _RealType = double>
class normal_distribution
{
  public:
    // Distribution types
    using result_type = _RealType;
    using scalar_type = internal::element_type_t<result_type>;
    using param_type = typename ::std::pair<scalar_type, scalar_type>;

    // Constructors
    normal_distribution() : normal_distribution(static_cast<scalar_type>(0.0)) {}
    explicit normal_distribution(scalar_type __mean, scalar_type __stddev = static_cast<scalar_type>(1.0))
        : mean_(__mean), stddev_(__stddev) {}
    explicit normal_distribution(const param_type& __params) : mean_(__params.first), stddev_(__params.second) {}

    // Reset function
    void
    reset()
    {
        flag_ = false;
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
    param(const param_type& __parm)
    {
        mean_ = __parm.first;
        stddev_ = __parm.second;
    }

    scalar_type
    min() const
    {
        return -(::std::numeric_limits<scalar_type>::infinity());
    }

    scalar_type
    max() const
    {
        return ::std::numeric_limits<scalar_type>::infinity();
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
    operator()(_Engine& __engine, unsigned int __randoms_num)
    {
        return operator()<_Engine>(__engine, param_type(mean_, stddev_), __randoms_num);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params, unsigned int __randoms_num)
    {
        result_type __part_vec;
        if (__randoms_num < 1)
            return __part_vec;

        int __portion = (__randoms_num > size_of_type_) ? size_of_type_ : __randoms_num;

        __part_vec = result_portion_internal<size_of_type_, _Engine>(__engine, __params, __portion);
        return __part_vec;
    }

  private:
    // Size of type
    static constexpr int size_of_type_ = internal::type_traits_t<result_type>::num_elems;

    // Type of real distribution
    using uniform_result_type =
        typename ::std::conditional<size_of_type_ % 2, sycl::vec<scalar_type, size_of_type_ + 1>, result_type>::type;

    // Distribution parameters
    scalar_type mean_;
    scalar_type stddev_;
    bool flag_ = false;
    scalar_type saved_ln_;
    scalar_type saved_u2_;

    // Static asserts
    static_assert(::std::is_floating_point<scalar_type>::value,
        "oneapi::dpl::normal_distribution. Error: unsupported data type");

    // Real distribution for the conversion
    uniform_real_distribution<uniform_result_type> uniform_real_distribution_;

    // Callback function
    template <typename _Type = float>
    scalar_type
    callback()
    {
        return ((scalar_type*)(internal::gaussian_sp_table))[1];
    }

    template <>
    scalar_type
    callback<double>()
    {
        return ((scalar_type*)(internal::gaussian_dp_table))[1];
    }

    // Get 2 * pi function
    template <typename _Type = float>
    scalar_type
    pi2()
    {
        return ((scalar_type*)(internal::gaussian_sp_table))[0];
    }

    template <>
    scalar_type
    pi2<double>()
    {
        return ((scalar_type*)(internal::gaussian_dp_table))[0];
    }

    // Implementation for generate function
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr != 0), result_type>::type
    generate(_Engine& __engine, const param_type __params)
    {
        return generate_vec_internal(__engine, __params, _Ndistr);
    }

    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr == 0), result_type>::type
    generate(_Engine& __engine, const param_type __params)
    {
        result_type __res;
        result_type __ln;

        if (!flag_)
        {
            result_type __u1 =
                uniform_real_distribution_(__engine, ::std::pair<scalar_type, scalar_type>(static_cast<scalar_type>(0.0),
                                                                                       static_cast<scalar_type>(1.0)));
            result_type __u2 =
                uniform_real_distribution_(__engine, ::std::pair<scalar_type, scalar_type>(static_cast<scalar_type>(0.0),
                                                                                       static_cast<scalar_type>(1.0)));

            __ln = (__u1 == static_cast<scalar_type>(0.0)) ? callback<scalar_type>() : sycl::log(__u1);

            __res = __params.first + __params.second * (sycl::sqrt(-static_cast<scalar_type>(2.0) * __ln) *
                                                  sycl::sin(pi2<scalar_type>() * __u2));

            saved_ln_ = __ln;
            saved_u2_ = __u2;
        }
        else
        {
            __res = __params.first + __params.second * (sycl::sqrt(-static_cast<scalar_type>(2.0) * saved_ln_) *
                                                  sycl::cos(pi2<scalar_type>() * saved_u2_));
        }

        flag_ = !flag_;

        return __res;
    }

    // Implementation for the generate vector function
    template <class _Engine>
    result_type
    generate_vec_internal(_Engine& __engine, const param_type __params, unsigned int __N)
    {

        uniform_result_type __u;
        scalar_type __u1, __u2, __ln;
        scalar_type __mean = __params.first, __stddev = __params.second;
        result_type __res;

        if (!flag_)
        {
            unsigned int __tail = __N % 2;
            __u = uniform_real_distribution_(
                __engine, param_type(static_cast<scalar_type>(0.0), static_cast<scalar_type>(1.0)), __N + __tail);

            for (unsigned int __i = 0; __i < __N - __tail; __i += 2)
            {
                __u1 = __u[__i];
                __u2 = __u[__i + 1];
                __ln = (__u1 == static_cast<scalar_type>(0.0)) ? callback<scalar_type>() : sycl::log(__u1);
                __res[__i] = __mean + __stddev * (sycl::sqrt(-static_cast<scalar_type>(2.0) * __ln) *
                                          sycl::sin(pi2<scalar_type>() * __u2));
                __res[__i + 1] = __mean + __stddev * (sycl::sqrt(-static_cast<scalar_type>(2.0) * __ln) *
                                              sycl::cos(pi2<scalar_type>() * __u2));
            }
            if (__tail)
            {
                __u1 = __u[__N - 1];
                __u2 = __u[__N];
                __ln = (__u1 == static_cast<scalar_type>(0.0)) ? callback<scalar_type>() : sycl::log(__u1);
                __res[__N - 1] = __mean + __stddev * (sycl::sqrt(-static_cast<scalar_type>(2.0) * __ln) *
                                              sycl::sin(pi2<scalar_type>() * __u2));

                saved_ln_ = __ln;
                saved_u2_ = __u2;
                flag_ = true;
            }
        }
        else
        {
            __res[0] = __mean + __stddev * (sycl::sqrt(-static_cast<scalar_type>(2.0) * saved_ln_) *
                                      sycl::cos(pi2<scalar_type>() * saved_u2_));

            flag_ = false;

            unsigned int __tail = (__N - 1u) % 2u;

            __u = uniform_real_distribution_(
                __engine, param_type(static_cast<scalar_type>(0.0), static_cast<scalar_type>(1.0)), __N - 1u + __tail);

            for (unsigned int __i = 1; __i < (__N - __tail); __i += 2)
            {
                __u1 = __u[__i - 1];
                __u2 = __u[__i];
                __ln = (__u1 == static_cast<scalar_type>(0.0)) ? callback<scalar_type>() : sycl::log(__u1);
                __res[__i] = __mean + __stddev * (sycl::sqrt(-static_cast<scalar_type>(2.0) * __ln) *
                                          sycl::sin(pi2<scalar_type>() * __u2));
                __res[__i + 1] = __mean + __stddev * (sycl::sqrt(-static_cast<scalar_type>(2.0) * __ln) *
                                              sycl::cos(pi2<scalar_type>() * __u2));
            }
            if (__tail)
            {
                __u1 = __u[__N - 1];
                __u2 = __u[__N];
                __ln = (__u1 == static_cast<scalar_type>(0.0)) ? callback<scalar_type>() : sycl::log(__u1);
                __res[__N - 1] = __mean + __stddev * (sycl::sqrt(-static_cast<scalar_type>(2.0) * __ln) *
                                              sycl::sin(pi2<scalar_type>() * __u2));

                saved_ln_ = __ln;
                saved_u2_ = __u2;
                flag_ = true;
            }
        }
        return __res;
    }

    // Implementation for result_portion function
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr != 0), result_type>::type
    result_portion_internal(_Engine& __engine, const param_type __params, unsigned int __N)
    {
        return generate_vec_internal(__engine, __params, __N);
    }
};

} // namespace dpl
} // namespace oneapi

#endif // #ifndf DPSTD_NORMAL_DISTRIBUTION