// -*- C++ -*-
//===-- philox_unit_test.pass.cpp -----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Unit test for Philox random number generation engine - corner-case testing functions

#include "support/utils.h"
#include "support/test_config.h"

#include <random>
#include <oneapi/dpl/random>

namespace ex = oneapi::dpl::experimental;

/* Corner-case testing functions */
template <typename Engine>
int api_test();

template <typename Engine>
int seed_test();

template <typename Engine>
int discard_test();

template <typename Engine>
int set_counter_conformance_test();

template <typename Engine>
int skip_test();

template <typename Engine>
int counter_overflow_test();

template <typename Engine>
int discard_overflow_test();

int
main()
{
    int err = 0;
    
    std::cout << "void api_test() [Engine = philox4x32]";
    err += api_test<ex::philox4x32>();
    std::cout << "void api_test() [Engine = philox4x64]";
    err += api_test<ex::philox4x64>();

    std::cout << "void seed_test() [Engine = philox4x32]";
    err += seed_test<ex::philox4x32>();
    std::cout << "void seed_test() [Engine = philox4x64]";
    err += seed_test<ex::philox4x64>();

    std::cout << "void discard_test() [Engine = philox4x32]";
    err += discard_test<ex::philox4x32>();
    std::cout << "void discard_test() [Engine = philox4x64]";
    err += discard_test<ex::philox4x64>();

    std::cout << "void set_counter_conformance_test() [Engine = philox4x32]";
    err += set_counter_conformance_test<ex::philox4x32>();
    std::cout << "void set_counter_conformance_test() [Engine = philox4x64]";
    err += set_counter_conformance_test<ex::philox4x64>();

    std::cout << "void skip_test() [Engine = philox4x32]";
    err += skip_test<ex::philox4x32>();
    std::cout << "void skip_test() [Engine = philox4x64]";
    err += skip_test<ex::philox4x64>();

    std::cout << "void counter_overflow_test() [Engine = philox4x32]";
    err += counter_overflow_test<ex::philox4x32>();
    std::cout << "void counter_overflow_test() [Engine = philox4x64]";
    err += counter_overflow_test<ex::philox4x64>();

    std::cout << "void discard_overflow_test() [Engine = philox4x32]";
    err += discard_overflow_test<ex::philox4x32>();
    std::cout << "void discard_overflow_test() [Engine = philox4x64]";
    err += discard_overflow_test<ex::philox4x64>();

    EXPECT_TRUE(!err, "Test FAILED");

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}

/*
 * All functions below have the following return values:
 *           0 in case of success
 *           1 in case of failure
 */

template <typename Engine>
int api_test() {
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
            return 1;
        }
        engine2.seed(42);
        if((engine == engine2) || !(engine != engine2)) {
            std::cout << " failed !=, == for the different engines" << std::endl;
            return 1;
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
            return 1;
        }
    }
    {
        Engine engine;
        engine.min();
        engine.max();
    }

    std::cout << " passed" << std::endl;
    return 0;
}

template <typename Engine>
int seed_test() {
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
            return 1;
        }
    }
    
    std::cout << " passed" << std::endl;
    return 0;
}

template <typename Engine>
int discard_test() {
    int ret = 0;

    constexpr std::size_t n = 10; // arbitrary length we want to check
    typename Engine::result_type reference[n];
    Engine engine;
    for(int i = 0; i < n; i++) {
        reference[i] = engine();
    }
    for(int i = 0; i < n; i++) {
        engine.seed();
        engine.discard(i);
        for(std::size_t j = i; j < n; j++) {
            if(reference[j] != engine()) {
                std::cout << " failed with error in element " << j << " discard " << i << std::endl;
                ret++;
                break;
            }
        }
    }

    for(int i = 1; i < n; i++) {
        for(int j = 1; j < i; j++) {
            engine.seed();
            for(std::size_t k = 0; k < i - j; k++) {
                engine();
            }
            engine.discard(j);
            if(reference[i] != engine()) {
                std::cout << " failed on step " << i << " " << j << std::endl;
                ret++;
                break;
            }
        }
    }

    if(!ret) {
        std::cout << " passed" << std::endl;
    }

    return ret;
}

template <typename Engine>
int set_counter_conformance_test() {
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
        return 0;
    } else {
        std::cout << " failed" << std::endl;
        return 1;
    }
}

template <typename Engine>
int skip_test() {
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
            return 1;
        }
    }

    std::cout << " passed" << std::endl;
    return 0;
}

template <typename Engine>
int counter_overflow_test() {
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
        return 0;
    } else {
        std::cout << " failed" << std::endl;
        return 1;
    }
}

template <typename Engine>
int discard_overflow_test() {
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
        return 0;
    } else {
        std::cout << " failed" << std::endl;
        return 1;
    }
}
