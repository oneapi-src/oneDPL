Random Number Generators
########################

|onedpl_long| (|onedpl_short|) offers support of random number generation, including:

- Random number engines, which generate unsigned integer sequences of random numbers.
- Random number distributions (example: ``uniform_real_distribution``), which converts the output of
  random number engines into various statistical distributions.

Random Number Engines
---------------------

Random number engines use seed data as an entropy source to generate pseudo-random numbers. 
|onedpl_short| provides several class templates for customized engines, they are defined in the header
``<oneapi/dpl/random>``.

============================== =========================================================================================================
Engine                         Description
============================== =========================================================================================================
``linear_congruential_engine`` Implements a linear congruential algorithm
``subtract_with_carry_engine`` Implements a subtract-with-carry algorithm
``discard_block_engine``       Implements a discard block adaptor
============================== =========================================================================================================

Predefined Random Number Engines
--------------------------------

Predefined random number engines are instantiations of random number engines class templates. 
The types below are defined in the header ``<oneapi/dpl/random>`` under the ``oneapi::dpl::`` namespace.

===================================================================== =========================================================================================================
Type                                                                  Description
===================================================================== =========================================================================================================
``minstd_rand0``                                                      ``oneapi::dpl::linear_congruential_engine<std::uint32_t, 16807, 0, 2147483647>``
``minstd_rand``                                                       ``oneapi::dpl::linear_congruential_engine<std::uint32_t, 48271, 0, 2147483647>``
``ranlux24_base``                                                     ``oneapi::dpl::subtract_with_carry_engine<std::uint32_t, 24, 10, 24>``
``ranlux48_base``                                                     ``oneapi::dpl::subtract_with_carry_engine<std::uint64_t, 48, 5, 12>``
``ranlux24``                                                          ``oneapi::dpl::discard_block_engine<ranlux24_base, 223, 23>``
``ranlux48``                                                          ``oneapi::dpl::discard_block_engine<ranlux48_base, 389, 11>``
===================================================================== =========================================================================================================

The engines described below can efficiently generate vectors of random numbers. These types are
defined in the header ``<oneapi/dpl/random>`` under the ``oneapi::dpl::`` namespace.

===================================================================== =========================================================================================================
Type                                                                  Description
===================================================================== =========================================================================================================
``template<std::int32_t N> minstd_rand0_vec<N>``                      ``oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, N>, 16807, 0, 2147483647>``
                                                                      minstd_rand0 for a vector generation case
``template<std::int32_t N> minstd_rand_vec<N>``                       ``oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, N>, 48271, 0, 2147483647>``
                                                                      minstd_rand for a vector generation case
``template<std::int32_t N> ranlux24_base_vec<N>``                     ``oneapi::dpl::subtract_with_carry_engine<sycl::vec<std::uint32_t, N>, 24, 10, 24>``
                                                                      ranlux24_base for a vector generation case
``template<std::int32_t N> ranlux48_base_vec<N>``                     ``oneapi::dpl::subtract_with_carry_engine<sycl::vec<std::uint64_t, N>, 48, 5, 12>``
                                                                      ranlux48_base for a vector generation case
``template<std::int32_t N> ranlux24_vec<N>``                          ``oneapi::dpl::discard_block_engine<ranlux24_base_vec<N>, 223, 23>``
                                                                      ranlux24 for a vector generation case
``template<std::int32_t N> ranlux48_vec<N>``                          ``oneapi::dpl::discard_block_engine<ranlux48_base_vec<N>, 389, 11>``
                                                                      ranlux48 for vector generation case
===================================================================== =========================================================================================================

Random Number Distributions
---------------------------

Random number distributions process the output of random number engines in such a way that the
resulting output is distributed according to a defined statistical probability density function. They
are defined in the header ``<oneapi/dpl/random>`` under the ``oneapi::dpl::`` namespace.

============================== =========================================================================================================
Distribution                   Description
============================== =========================================================================================================
``uniform_int_distribution``   Produces integer values evenly distributed across a range
``uniform_real_distribution``  Produces real values evenly distributed across a range
``normal_distribution``        Produces real values according to the Normal (Gaussian) distribution
``exponential_distribution``   Produces real values according to the Exponential distribution
``bernoulli_distribution``     Produces bool values according to the Bernoulli distribution
``geometric_distribution``     Produces integer values according to the Geometric distribution
``weibull_distribution``       Produces real values according to the Weibull distribution
``lognormal_distribution``     Produces real values according to the Lognormal distribution
``extreme_value_distribution`` Produces real values according to the Extreme value (Gumbel) distribution
``cauchy_distribution``        Produces real values according to the Cauchy distribution
============================== =========================================================================================================

Usage Model of |onedpl_short| Random Number Generation Functionality
--------------------------------------------------------------------

Random number generation is available for SYCL* device-side and host-side code. For example:

.. code:: cpp

    #include <oneapi/dpl/random>
    #include <sycl/sycl.hpp>
    #include <iostream>
    #include <vector>

    int main() {
        sycl::queue queue(sycl::default_selector_v);

        std::int64_t nsamples = 100;
        std::uint32_t seed = 777;
        std::vector<float> x(nsamples);
        {
            sycl::buffer<float, 1> x_buf(x.data(), sycl::range<1>(x.size()));

            queue.submit([&] (sycl::handler &cgh) {

                auto x_acc =
                x_buf.template get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<class count_kernel>(sycl::range<1>(nsamples),
                    [=](sycl::item<1> idx) {
                    std::uint64_t offset = idx.get_linear_id();

                    // Create minstd_rand engine
                    oneapi::dpl::minstd_rand engine(seed, offset);

                    // Create float uniform_real_distribution distribution
                    oneapi::dpl::uniform_real_distribution<float> distr;

                    // Generate float random number
                    auto res = distr(engine);

                    // Store results to x_acc
                    x_acc[idx] = res;
                });
            });
        }

        std::cout << "\nFirst 5 samples of minstd_rand with scalar generation" << std::endl;
        for(int i = 0; i < 5; i++) {
            std::cout << x.begin()[i] << std::endl;
        }

        std::cout << "\nLast 5 samples of minstd_rand with scalar generation" << std::endl;
        for(int i = 0; i < 5; i++) {
            std::cout << x.rbegin()[i] << std::endl;
        }
        return 0;
    }
