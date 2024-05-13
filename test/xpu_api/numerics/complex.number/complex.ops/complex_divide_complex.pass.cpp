//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   operator/(const complex<T>& lhs, const complex<T>& rhs);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& lhs, const dpl::complex<T>& rhs, dpl::complex<T> x)
{
    assert(is_about(lhs / rhs, x));
}

template <class T>
void
test()
{
    dpl::complex<T> lhs(-4.0f, 7.5f);
    dpl::complex<T> rhs(1.5f, 2.5f);
    dpl::complex<T>   x(1.5f, 2.5f);
    test(lhs, rhs, x);
}

template <class TChecker>
void test_edges(TChecker& check_obj)
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            dpl::complex<double> r = testcases[i] / testcases[j];
            switch (classify(testcases[i]))
            {
            case zero:
                switch (classify(testcases[j]))
                {
                case zero:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case non_zero:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == zero);
                    break;
                case inf:
#if !_PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == zero);
#endif // _PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    break;
                case NaN:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case non_zero_nan:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                }
                break;
            case non_zero:
                switch (classify(testcases[j]))
                {
                case zero:
#if !_PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    break;
                case non_zero:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == non_zero);
                    break;
                case inf:
#if !_PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    if (classify(r) != zero)
                    {
                        //  0       dpl::complex<double>(1.e-6, 1.e-6)
                        // 33       dpl::complex<double>(-2, NAN)
                        std::cout << "i = " << i << ", j == " << j << std::endl;
                        CALL_CHECK_OBJ_I_J(check_obj, i, j, false);
                    }
#endif // _PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    break;
                case NaN:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case non_zero_nan:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                }
                break;
            case inf:
                switch (classify(testcases[j]))
                {
                case zero:
#if !_PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    break;
                case non_zero:
#if !_PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    break;
                case inf:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case NaN:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case non_zero_nan:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                }
                break;
            case NaN:
                switch (classify(testcases[j]))
                {
                case zero:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case non_zero:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case inf:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case NaN:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case non_zero_nan:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                }
                break;
            case non_zero_nan:
                switch (classify(testcases[j]))
                {
                case zero:
#if !_PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN
                    break;
                case non_zero:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case inf:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case NaN:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                case non_zero_nan:
                    CALL_CHECK_OBJ_I_J(check_obj, i, j, classify(r) == NaN);
                    break;
                }
                break;
            }
        }
    }
}

template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime>
int
run_test();
int
main(int, char**)
{
    std::cout << "Run test on host" << std::endl;
    run_test<::std::true_type, ::std::true_type>();
    //using HasDoubleTypeSupportInRuntime = ::std::true_type;
    //using HasntDoubleTypeSupportInRuntime = ::std::false_type;
    //using HasntLongDoubleSupportInCompiletime = ::std::false_type;
    //std::cout << "Run test on device" << std::endl;
    //TestUtils::run_test_in_kernel(
    //    [&]() { run_test<HasDoubleTypeSupportInRuntime, HasntLongDoubleSupportInCompiletime>(); },
    //    [&]() { run_test<HasntDoubleTypeSupportInRuntime, HasntLongDoubleSupportInCompiletime>(); });
    return TestUtils::done();
}
template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime, class TChecker>
int
run_test()
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT_REF_CAPT(test_edges(check_obj))

  return 0;
}
