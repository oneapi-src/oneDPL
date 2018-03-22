/*
    Copyright (c) 2017-2018 Intel Corporation

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

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

using namespace TestUtils;

template<typename In1, typename In2, typename Out>
class TheOperation {
    Out val;
public:
    TheOperation(Out v) : val(v) {}
    Out operator()(const In1& x, const In2& y) const {return Out(val+x-y);}
};

template<typename InputIterator1, typename InputIterator2, typename OutputIterator>
void check_and_reset(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator out_first) {
    typedef typename std::iterator_traits<OutputIterator>::value_type Out;
    typename std::iterator_traits<OutputIterator>::difference_type k = 0;
    for (; first1 != last1; ++first1, ++first2, ++out_first, ++k) {
        // check
        Out expected = Out(1.5) + *first1 - *first2;
        Out actual = *out_first;
        if(std::is_floating_point<Out>::value) {
            EXPECT_TRUE((expected  > actual ? expected - actual : actual - expected) < 1e7, "wrong value in output sequence");
        } else {
            EXPECT_EQ(expected, actual, "wrong value in output sequence");
        }
        // reset
        *out_first = k%7!=4 ? 7*k-5 : 0;
    }
}

struct test_one_policy {
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryOp>
    void operator()( Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
        OutputIterator out_first, BinaryOp op) {
        auto orrr = std::transform( exec, first1, last1, first2, out_first, op );
//        EXPECT_TRUE( orrr==out.end(), "transform returned wrong iterator" );
        check_and_reset(first1, last1, first2, out_first);
    }
};

template <typename In1, typename In2, typename Out>
void test() {
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<In1> in1(n, [](size_t k) {
            return k%5!=1 ? 3*k-7 : 0;
        });
        Sequence<In2> in2(n, [](size_t k) {
            return k%7!=2 ? 5*k-5 : 0;
        });

        Sequence<Out> out( n, [](size_t k){return -1;}  );

        //const operator()
        auto flip_const = TheOperation<In1,In2,Out>(Out(1.5));
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.end(), in2.begin(), in2.end(), out.begin(), flip_const);
        invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cend(), in2.cbegin(), in2.cend(), out.begin(), flip_const);

        //non-const operator()
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.end(), in2.begin(), in2.end(), out.begin(), non_const(flip_const));
        invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cend(), in2.cbegin(), in2.cend(), out.begin(), non_const(flip_const));

        //lambda
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.end(), in2.begin(), in2.end(), out.begin(), [](const In1 x, In2& y) {return Out(Out(1.5) + x - y); });
        invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cend(), in2.cbegin(), in2.cend(), out.begin(), [](const In1 x, In2& y) {return Out(Out(1.5) + x - y); });
    }
}

int32_t main( ) {
    test<int32_t, int32_t, int32_t>();
    test<int32_t, float32_t, float32_t>();
    test<float32_t, float32_t, float32_t>();
    test<int64_t, float64_t, float32_t>();
    test<int8_t, float64_t, int8_t>();
    std::cout << "done" << std::endl;
    return 0;
}
