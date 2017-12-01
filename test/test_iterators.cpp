/*
    Copyright (c) 2017 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#include <vector>
#include <iostream>

#include "pstl/algorithm"
#include "pstl/iterators.h"
#include "test/utils.h"

using namespace TestUtils;

//common checks of a random access iterator functionality
template <typename RandomIt>
void test_random_iterator(RandomIt it) {
    auto it2 = it;
    EXPECT_TRUE(*it == it2[0], "wrong result with */[] operations for an iterator");
    EXPECT_TRUE((++it, --it) == it2, "wrong result with prefix ++/-- operations for an iterator");

    auto res = it - it2++;
    EXPECT_TRUE(res == 0, "wrong result with postfix ++ operation for an iterator");
    res = it - it2;
    EXPECT_TRUE(res == -1, "wrong result with postfix ++ operation for an iterator");

    res = it - it2--;
    EXPECT_TRUE(res == -1, "wrong result with postfix -- operation for an iterator");
    res = it - it2;
    EXPECT_TRUE(res == 0, "wrong result with postfix -- operation for an iterator");

    it2 = 1 + it + 1;
    EXPECT_TRUE(it == it2 + (it-it2), "wrong result with +/- == operations for an iterator");
    EXPECT_TRUE(it2 - it == 2, "wrong result with difference between iterators");
    EXPECT_TRUE((it += 2) == it2, "wrong result with += operator iterators");
    EXPECT_TRUE((it -= 2) == (it2-=2), "wrong result with -= operator iterators");

    EXPECT_TRUE(it <= it2, "wrong result with <= operator iterators");
    EXPECT_TRUE(it >= it2, "wrong result with >= operator iterators");

    ++it2;
    EXPECT_TRUE(it < it2, "wrong result with < operator iterators");
    EXPECT_TRUE(it2 > it, "wrong result with > operator iterators");
    EXPECT_TRUE(it != it2, "wrong result with != operator iterators");
}

struct test_counting_iterator {
    template <typename Policy, typename T, typename IntType>
    void operator()( Policy exec, Sequence<T>& in, IntType begin, IntType end, const T& value) {

        auto b = pstl::counting_iterator<IntType>(begin);
        auto e = pstl::counting_iterator<IntType>(end);

        //checks in using
        std::for_each(exec, b, e, [&in, &value](IntType i) { in[i] = value; });
        auto res = std::all_of(in.begin(), in.end(), [&value](const T& a) {return a==value;});
        EXPECT_TRUE(res, "wrong result with counting_iterator iterator");

        //explicit checks of the counting iterator specific
        EXPECT_TRUE(b[0]==0, "wrong result with operator[] for an iterator");
        EXPECT_TRUE(*(b + 1) == 1, "wrong result with operator+ for an iterator");
        EXPECT_TRUE(*(b+=1) == 1, "wrong result with operator+= for an iterator");

        b = pstl::counting_iterator<IntType>(begin);
        test_random_iterator(b);
    }
};

struct test_zip_iterator {
    template <typename Policy, typename T1, typename T2>
    void operator()( Policy exec, Sequence<T1>& in1, Sequence<T2>& in2) {
        auto b = pstl::make_zip_iterator(in1.begin(), in2.begin());
        auto e = pstl::make_zip_iterator(in1.end(), in2.end());

        //checks in using
        std::for_each(exec, b, e, [](const std::tuple<T1&, T2&>& a) { std::get<0>(a) = 1, std::get<1>(a) = 1;});
        auto res = std::all_of(b, e, [](const std::tuple<T1&, T2&>& a) {return std::get<0>(a) == 1 && std::get<1>(a) == 1;});
        EXPECT_TRUE(res, "wrong result with zip_iterator iterator");

        test_random_iterator(b);
    }
};

template <typename T, typename IntType>
void test_iterator_by_type(IntType n) {

    const IntType beg = 0;
    const IntType end = n;
    Sequence<T> in(end-beg, [](size_t)->T { return T(0); }); //fill with zeros
    T value = -1;
    test_counting_iterator()(pstl::execution::seq, in, beg, end, value);
    test_counting_iterator()(pstl::execution::unseq, in, beg, end, value);
    test_counting_iterator()(pstl::execution::par, in, beg, end, value);
    test_counting_iterator()(pstl::execution::par_unseq, in, beg, end, value);

    Sequence<IntType> in2(end-beg, [](size_t)->IntType { return IntType(0); }); //fill with zeros
    
    // Zip Iterator doesn't work correctly with unseq and par_unseq policies with compilers older than icc 18
    #if (__INTEL_COMPILER && __INTEL_COMPILER<1800)
        test_zip_iterator()(pstl::execution::seq, in, in2);
        #if __PSTL_USE_PAR_POLICIES
            test_zip_iterator()(pstl::execution::par, in, in2);
        #endif
    #else
    test_zip_iterator()(pstl::execution::seq, in, in2);
    test_zip_iterator()(pstl::execution::unseq, in, in2);
    test_zip_iterator()(pstl::execution::par, in, in2);
    test_zip_iterator()(pstl::execution::par_unseq, in, in2);
    #endif
}

int32_t main() {

    const auto n1 = 1000;
    const auto n2 = 100000;

    test_iterator_by_type<int16_t, int16_t>(n1);
    test_iterator_by_type<int16_t, int32_t>(n2);
    test_iterator_by_type<int16_t, int64_t>(n2);

    test_iterator_by_type<float64_t, int16_t>(n1);
    test_iterator_by_type<float64_t, int32_t>(n2);
    test_iterator_by_type<float64_t, int64_t>(n2);

    std::cout << "done" << std::endl;
    return 0;
}
