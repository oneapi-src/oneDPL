// -*- C++ -*-
//===-- philox_test.pass.cpp ----------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test for Philox random number generation engine - comparison of 10 000th element

#include "support/utils.h"
#include "support/test_config.h"
#include "common_for_conformance_tests.hpp"

#include <random>

#include <oneapi/dpl/random>

namespace ex = oneapi::dpl::experimental;

/* Corner-case testing functions */
template <typename Engine>
void api_test();

template <typename Engine>
void seed_test();

template <typename Engine>
void discard_test();

template <typename Engine>
void set_counter_conformance_test();

template <typename Engine>
void skip_test();

template <typename Engine>
void counter_overflow_test();

template <typename Engine>
void discard_overflow_test();

int
main()
{
    /* ----- Simple conformance testing -----*/
    std::cout << "----- Simple conformance testing -----" << std::endl;

    sycl::queue queue = TestUtils::get_test_queue();
    // Reference values
    std::uint_fast32_t philox4_32_ref = 1955073260;
    std::uint_fast64_t philox4_64_ref = 3409172418970261260;
    int err = 0;

    // Generate 10 000th element for philox4_32
    err += test<ex::philox4x32, 10000, 1>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<1>, 10000, 1>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<2>, 10000, 2>(queue) != philox4_32_ref;
    // In case of philox4x32_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<ex::philox4x32_vec<3>, 10002, 3>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<4>, 10000, 4>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<8>, 10000, 8>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<16>, 10000, 16>(queue) != philox4_32_ref;

    EXPECT_TRUE(!err, "Test FAILED");

    // Generate 10 000th element for philox4_64
    err += test<ex::philox4x64, 10000, 1>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<1>, 10000, 1>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<2>, 10000, 2>(queue) != philox4_64_ref;
    // In case of philox4x64_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<ex::philox4x64_vec<3>, 10002, 3>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<4>, 10000, 4>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<8>, 10000, 8>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<16>, 10000, 16>(queue) != philox4_64_ref;

    EXPECT_TRUE(!err, "Test FAILED");
    std::cout << "passed" << std::endl;
    
    /* ----- Corner-case testing -----*/
    std::cout << "----- Corner-case testing -----" << std::endl;
    
    std::cout << "void api_test() [Engine = philox4x32]";
    api_test<ex::philox4x32>();
    std::cout << "void api_test() [Engine = philox4x64]";
    api_test<ex::philox4x64>();

    std::cout << "void seed_test() [Engine = philox4x32]";
    seed_test<ex::philox4x32>();
    std::cout << "void seed_test() [Engine = philox4x64]";
    seed_test<ex::philox4x64>();

    std::cout << "void discard_test() [Engine = philox4x32]";
    discard_test<ex::philox4x32>();
    std::cout << "void discard_test() [Engine = philox4x64]";
    discard_test<ex::philox4x64>();

    std::cout << "void set_counter_conformance_test() [Engine = philox4x32]";
    set_counter_conformance_test<ex::philox4x32>();
    std::cout << "void set_counter_conformance_test() [Engine = philox4x64]";
    set_counter_conformance_test<ex::philox4x64>();

    std::cout << "void skip_test() [Engine = philox4x32]";
    skip_test<ex::philox4x32>();
    std::cout << "void skip_test() [Engine = philox4x64]";
    skip_test<ex::philox4x64>();

    std::cout << "void counter_overflow_test() [Engine = philox4x32]";
    counter_overflow_test<ex::philox4x32>();
    std::cout << "void counter_overflow_test() [Engine = philox4x64]";
    counter_overflow_test<ex::philox4x64>();

    std::cout << "void discard_overflow_test() [Engine = philox4x32]";
    discard_overflow_test<ex::philox4x32>();
    std::cout << "void discard_overflow_test() [Engine = philox4x64]";
    discard_overflow_test<ex::philox4x64>();

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}

template <typename Engine>
void api_test() {
    {
        Engine engine;
        engine.seed();
    }
    {
        Engine engine(1);
        engine.seed(1);
    }
    {
        Engine engine;
        Engine engine2;
        if(!(engine == engine2) || (engine != engine2)) {
            std::cout << " failed !=, == for the same engines" << std::endl;
            return;
        }
        engine2.seed(42);
        if((engine == engine2) || !(engine != engine2)) {
            std::cout << " failed !=, == for the different engines" << std::endl;
            return;
        }
    }
    {
        std::ostringstream os;
        Engine engine;
        os << engine << std::endl;
        Engine engine2;
        engine2();
        std::istringstream in(os.str());
        in >> engine2;
        if(engine != engine2) {
            std::cout << " failed for >> << operators" << std::endl;
            return;
        }
    }
    {
        Engine engine;
        engine.min();
        engine.max();
    }
    std::cout << " passed" << std::endl;
}

template <typename Engine>
void seed_test() {
    for(int i = 1; i < 5; i++) { // make sure that the state is reset properly for all idx positions
        Engine engine;
        typename Engine::result_type res;
        for(int j = 0; j < i - 1; j++) {
            engine();
        }
        res = engine();
        engine.seed();
        for(int j = 0; j < i - 1; j++) {
            engine();
        }
        if(res != engine()) {
            std::cout << " failed while generating " << i  << " elements" << std::endl;
        }
    }
    std::cout << " passed" << std::endl;
}

template <typename Engine>
void discard_test() {
    {
        constexpr size_t n = 10; // arbitrary length we want to check
        typename Engine::result_type reference[n];
        Engine engine;
        for(int i = 0; i < n; i++) {
            reference[i] = engine();
        }
        for(int i = 0; i < n; i++) {
            engine.seed();
            engine.discard(i);
            for(size_t j = i; j < n; j++) {
                if(reference[j] != engine()) {
                    std::cout << " failed with error in element " << j << " discard " << i << std::endl;
                    break;
                }
            }
        }
        std::cout << " passed step 1 discard from the intial state" << std::endl;

        for(int i = 1; i < n; i++) {
            for(int j = 1; j < i; j++) {
                engine.seed();
                for(size_t k = 0; k < i - j; k++) {
                    engine();
                }
                engine.discard(j);
                if(reference[i] != engine()) {
                    std::cout << " failed on step " << i << " " << j << std::endl;
                    break;
                }
            }
        }
        std::cout << " passed step 2 discard after generation" << std::endl;
    }
}

template <typename Engine>
void set_counter_conformance_test() {
    Engine engine;
    std::array<typename Engine::result_type, Engine::word_count> counter;
    for(int i = 0; i < Engine::word_count - 1; i++) {
        counter[i] = 0;
    }

    counter[Engine::word_count - 1] = 2499; // to get 10'000 element
    engine.set_counter(counter);

    for(int i = 0; i < Engine::word_count - 1; i++) {
        engine();
    }

    typename Engine::result_type reference;
    if(std::is_same_v<Engine, ex::philox4x32>) {
        reference = 1955073260;
    }
    else {
        reference = 3409172418970261260;
    }
    if(engine() == reference) {
        std::cout << " passed" << std::endl;
    } else {
        std::cout << " failed" << std::endl;
    }
}

template <typename Engine>
void skip_test() {
    using T = typename Engine::result_type;
    for(T i = 1; i <= Engine::word_count + 1; i++) {
        Engine engine1;
        std::array<T, Engine::word_count> counter = {0, 0, 0, i / Engine::word_count};
        engine1.set_counter(counter);
        for(T j = 0; j < i % Engine::word_count; j++) {
            engine1();
        }

        Engine engine2;
        engine2.discard(i);

        if(engine1() != engine2()) {
            std::cout << " failed for " << i << " skip" << std::endl;
            return;
        }
    }
    std::cout << " passed" << std::endl;
}

template <typename Engine>
void counter_overflow_test() {
    using T = typename Engine::result_type;
    Engine engine1;
    std::array<T, Engine::word_count> counter;
    for(int i = 0; i < Engine::word_count; i++) {
        counter[i] = std::numeric_limits<T>::max();
    }

    engine1.set_counter(counter);
    for(int i = 0; i < Engine::word_count; i++) {
        engine1();
    } // all counters overflowed == start from 0 0 0 0

    Engine engine2;

    if(engine1() == engine2()) {
        std::cout << " passed" << std::endl;
    } else {
        std::cout << " failed" << std::endl;
    }
}

template <typename Engine>
void discard_overflow_test() {
    using T = typename Engine::result_type;
    Engine engine1;
    std::array<T, Engine::word_count> counter;

    for(int i = 0; i < Engine::word_count; i++) {
        counter[i] = 0;
    }

    if(std::is_same_v<Engine, ex::philox4x32>) {
        counter[1] = 1;
    }
    else if(std::is_same_v<Engine, ex::philox4x64>) {
        counter[2] = 1;
    }

    engine1.set_counter(counter);

    Engine engine2;

    for(int i = 0; i < Engine::word_count; i++) {
        engine2();
    }
    for(int i = 0; i < Engine::word_count; i++) {
        engine2.discard(std::numeric_limits<unsigned long long>::max());
    }

    if(engine1() == engine2()) {
        std::cout << " passed" << std::endl;
    } else {
        std::cout << " failed" << std::endl;
    }
}
