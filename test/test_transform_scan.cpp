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
#include "pstl/numeric"
#include "test/utils.h"
#include "pstl/internal/numeric_impl.h" //for usage a serial algo version

using namespace TestUtils;

// Most of the framework required for testing inclusive and exclusive transform-scans is identical,
// so the tests for both are in this file.  Which is being tested is controlled by the global
// flag inclusive, which is set to each alternative by main().
static bool inclusive;

template<typename Iterator, typename Size, typename T>
void check_and_reset(Iterator expected_first, Iterator out_first, Size n, T trash) {
    EXPECT_EQ_N(expected_first, out_first, n, inclusive ? "result from transform_inclusive_scan" : "result from transform_exclusive_scan");
    std::fill_n(out_first, n, trash);
}

struct test_transform_scan {
    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp, typename T, typename BinaryOp>
    typename std::enable_if<!TestUtils::isReverse<InputIterator>::value, void>::type
        operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
            OutputIterator expected_first, OutputIterator expected_last, Size n, UnaryOp unary_op, T init, BinaryOp binary_op, T trash ) {
        using namespace std;

        auto orr1 = inclusive ?
            transform_inclusive_scan(pstl::execution::seq, first, last, expected_first, binary_op, unary_op, init) :
            transform_exclusive_scan(pstl::execution::seq, first, last, expected_first, init, binary_op, unary_op);
        auto orr2 = inclusive ?
            transform_inclusive_scan(exec, first, last, out_first, binary_op, unary_op, init) :
            transform_exclusive_scan(exec, first, last, out_first, init, binary_op, unary_op);
        EXPECT_TRUE( out_last==orr2, "transform...scan returned wrong iterator" );
        check_and_reset(expected_first, out_first, n, trash );

        // Checks inclusive scan if init is not provided
        if(inclusive && n > 0) {
            orr1 = transform_inclusive_scan(pstl::execution::seq, first, last, expected_first, binary_op, unary_op);
            orr2 = transform_inclusive_scan(exec, first, last, out_first, binary_op, unary_op);
            EXPECT_TRUE(out_last == orr2, "transform...scan returned wrong iterator");
            check_and_reset(expected_first, out_first, n, trash);
        }
    }

    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp, typename T, typename BinaryOp>
    typename std::enable_if<TestUtils::isReverse<InputIterator>::value, void>::type
        operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last,
            OutputIterator expected_first, OutputIterator expected_last, Size n, UnaryOp unary_op, T init, BinaryOp binary_op, T trash) {
    }
};

const uint32_t encryption_mask = 0x314;

template <typename In, typename Out, typename UnaryOp, typename BinaryOp>
void test( UnaryOp unary_op, Out init, BinaryOp binary_op, Out trash ) {
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<In> in(n, [](size_t k) {
            return In(k ^ encryption_mask);
        });

        Out tmp = init;
        Sequence<Out> expected(n, [&](size_t k)->Out {
            if( inclusive ) {
                tmp = binary_op(tmp, unary_op(in[k]));
                return tmp;
            } else {
                Out val = tmp;
                tmp = binary_op(tmp, unary_op(in[k]));
                return val;
            }
        });

        Sequence<Out> out(n, [&](size_t) {return trash;});

        auto result = inclusive ?
                      pstl::internal::brick_transform_scan(in.cbegin(), in.cend(), out.fbegin(), unary_op, init, binary_op, std::true_type()/*inclusive*/) :
                      pstl::internal::brick_transform_scan(in.cbegin(), in.cend(), out.fbegin(), unary_op, init, binary_op, std::false_type()/*exclusive*/);
        check_and_reset( expected.begin(), out.begin(), out.size(), trash );

        invoke_on_all_policies(test_transform_scan(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(), expected.end(), in.size(), unary_op, init, binary_op, trash);
        invoke_on_all_policies(test_transform_scan(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(), expected.end(), in.size(), unary_op, init, binary_op, trash);
    }
}

// Unary op
class ToMonoidElement {
    uint32_t decryption_mask;
public:
    ToMonoidElement(uint32_t decryption_mask_, OddTag) : decryption_mask(decryption_mask_) {}
    MonoidElement operator()(uint32_t x ) const {
        uint32_t y = x ^ decryption_mask;
        return MonoidElement(y, y+1, OddTag());
    }
};

int32_t main( ) {
    for(int32_t mode=0; mode<2; ++mode ) {
        inclusive = mode!=0;
        test<uint32_t, MonoidElement>(ToMonoidElement(encryption_mask,OddTag()), MonoidElement(~0u,0u,OddTag()), AssocOp(OddTag()), MonoidElement(666,666,OddTag()));
    }
    std::cout << "done" << std::endl;
    return 0;
}
