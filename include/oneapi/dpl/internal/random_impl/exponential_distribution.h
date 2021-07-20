#ifndef DPSTD_EXPONENTIAL_DISTRIBUTION
#define DPSTD_EXPONENTIAL_DISTRIBUTION

namespace oneapi
{
namespace dpl
{
template <class _RealType = double>
class exponential_distribution
{
  public:
    // Distribution types
    using result_type = _RealType;
    using scalar_type = internal::element_type_t<_RealType>;
    using param_type = scalar_type;

    // Constructors
    explicit exponential_distribution(scalar_type __lambda = static_cast<scalar_type>(1.0)) : lambda_(__lambda) {}

    // Reset function
    void
    reset()
    {
    }

    // Property functions
    scalar_type
    lambda() const
    {
        return lambda_;
    }

    param_type
    param() const
    {
        return param_type(lambda_);
    }

    void
    param(const param_type& __parm)
    {
        lambda_ = __parm;
    }

    scalar_type
    min() const
    {
        return 0;
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
        return operator()<_Engine>(__engine, param_type(lambda_));
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
        return operator()<_Engine>(__engine, param_type(lambda_), __random_nums);
    }

    template <class _Engine>
    result_type
    operator()(_Engine& __engine, const param_type& __params, unsigned int __random_nums)
    {
        return result_portion_internal<size_of_type_, _Engine>(__engine, __params, __random_nums);
    }

  private:
    // Size of type
    static constexpr int size_of_type_ = internal::type_traits_t<result_type>::num_elems; // ?

    // Static asserts
    static_assert(::std::is_floating_point<scalar_type>::value,
                  "oneapi::dpl::exponential_distribution. Error: unsupported data type");

    // Distribution parameters
    scalar_type lambda_;

    // Implementation for generate function
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr != 0), result_type>::type
    generate(_Engine& __engine, const param_type __params)
    {
        return generate_vec<_Ndistr, _Engine>(__engine, __params);
    }

    // Specialization of the scalar generation
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr == 0), result_type>::type
    generate(_Engine& __engine, const param_type __params)
    {
        result_type __res;
        uniform_real_distribution __u;
        __res = __u(__engine);
        if (__res < 0)
            __res = 0;
        else
            __res = ((-1) / __params) * sycl::log(__res);
        return __res;
    }

    // Specialization of the vector generation
    template <int __N, class _Engine>
    result_type
    generate_vec(_Engine& __engine, const param_type __params)
    {
        return generate_n_elems<_Engine>(__engine, __params, __N);
    }

    // Implementation for the N vector's elements generation
    template <class _Engine>
    result_type
    generate_n_elems(_Engine& __engine, const param_type __params, unsigned int __N)
    {
        result_type __res;
        uniform_real_distribution __u;
        for (int i = 0; i < __N; i++)
        {
            __res[i] = __u(__engine);
            if (__res[i] < 0)
                __res[i] = 0;
            else
                __res[i] = ((-1) / __params) * sycl::log(__res[i]);
        }
        return __res;
    }

    // Implementation for result_portion function
    template <int _Ndistr, class _Engine>
    typename ::std::enable_if<(_Ndistr != 0), result_type>::type
    result_portion_internal(_Engine& __engine, const param_type __params, unsigned int __N)
    {
        result_type __part_vec;
        if (__N == 0)
            return __part_vec;
        else if (__N >= _Ndistr)
            return operator()(__engine);

        __part_vec = generate_n_elems(__engine, __params, __N);
        return __part_vec;
    }
};
} // namespace dpl
} // namespace oneapi

#endif // #ifndf DPSTD_EXPONENTIAL_DISTRIBUTION