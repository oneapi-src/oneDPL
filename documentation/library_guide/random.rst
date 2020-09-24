Random Number Generators
########################

An Introduction to random number generation usage in oneDPL.

oneDPL library offers support of random number generation, including:

- Random number engines, which generate unsigned integer sequences of random numbers
- Random number distributions (e.g. uniform_real_distribution), which convert the output of random number engines into various statistical distributions

Random number engines
---------------------

Random number engines use seed data as entropy source to generate pseudo-random numbers.
Customized class templates which are available for oneDPL:

Defined in header ``<oneapi/dpl/random>``

============================== =========================================================================================================
Engine                         Description
============================== =========================================================================================================
``linear_congruential_engine`` implements linear congruential algorithm
``subtract_with_carry_engine`` implements subtract with carry algorithm
``discard_block_engine``       implements discard block adaptor
============================== =========================================================================================================

Predefined random number egnines
-----------------------------------

Predefined instantiations of random number engines class templates

Defined in header ``<oneapi/dpl/random>`` under ``oneapi::std::`` namespace

===================================================================== =========================================================================================================
Type                                                                  Description
===================================================================== =========================================================================================================
``minstd_rand0``                                                      ``oneapi::std::linear_congruential_engine<std::uint32_t, 16807, 0, 2147483647>``
``minstd_rand``                                                       ``oneapi::std::linear_congruential_engine<std::uint32_t, 48271, 0, 2147483647>``
``ranlux24_base``                                                     ``oneapi::std::subtract_with_carry_engine<std::uint32_t, 24, 10, 24>``
``ranlux48_base``                                                     ``oneapi::std::subtract_with_carry_engine<std::uint64_t, 48, 5, 12>``
``ranlux24``                                                          ``oneapi::std::discard_block_engine<ranlux24_base, 223, 23>``
``ranlux48``                                                          ``oneapi::std::discard_block_engine<ranlux48_base, 389, 11>``
===================================================================== =========================================================================================================

Defined in header ``<oneapi/dpl/random>`` under ``oneapi::dpl::`` namespace

===================================================================== =========================================================================================================
Type                                                                  Description
===================================================================== =========================================================================================================
``template<std::int32_t N> minstd_rand0_vec<N>``                      ``oneapi::std::linear_congruential_engine<sycl::vec<std::uint32_t, N>, 16807, 0, 2147483647>``
                                                                      minstd_rand0 for vector genertion case
``template<std::int32_t N> minstd_rand_vec<N>``                       ``oneapi::std::linear_congruential_engine<sycl::vec<std::uint32_t, N>, 48271, 0, 2147483647>``
                                                                      minstd_rand for vector genertion case
``template<std::int32_t N> ranlux24_base_vec<N>``                     ``oneapi::std::subtract_with_carry_engine<sycl::vec<std::uint32_t, N>, 24, 10, 24>``
                                                                      ranlux24_base for vector genertion case
``template<std::int32_t N> ranlux48_base_vec<N>``                     ``oneapi::std::subtract_with_carry_engine<sycl::vec<std::uint64_t, N>, 48, 5, 12>``
                                                                      ranlux48_base for vector genertion case
``template<std::int32_t N> ranlux24_vec<N>``                          ``oneapi::std::discard_block_engine<ranlux24_base_vec<N>, 223, 23>``
                                                                      ranlux24 for vector genertion case
``template<std::int32_t N> ranlux48_vec<N>``                          ``oneapi::std::discard_block_engine<ranlux48_base_vec<N>, 389, 11>``
                                                                      ranlux48 for vector genertion case
===================================================================== =========================================================================================================

Random number distributions
---------------------------

Random number distributions process the output of random number engines in such a way that resulting output is distributed according to a defined statistical probability density function

Defined in header ``<oneapi/dpl/random>``

============================== =========================================================================================================
Distribution                   Description
============================== =========================================================================================================
``uniform_int_distribution``   produces integer values evenly distributed across a range
``uniform_real_distribution``  produces real values evenly distributed across a range
``normal_distribution``        produces real values according to the Normal (Gaussian) distribution
============================== =========================================================================================================

Usage model of oneDPL random number generation functionality
------------------------------------------------------------

Random number generation may work for both DPC++ device-side and host-side code.

Example is represented below:

.. code:: cpp

    #include <iostream>
    #include <vector>
    #include <CL/sycl.hpp>
    #include <dstd/random>

    int main() {
        sycl::queue queue(sycl::default_selector{});

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
                    oneapi::std::minstd_rand engine(seed, offset);

                    // Create float uniform_real_distribution distribution
                    oneapi::std::uniform_real_distribution<float> distr;

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
