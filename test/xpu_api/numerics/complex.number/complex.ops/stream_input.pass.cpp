//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-localization

// <complex>

// template<class T, class charT, class traits>
//   basic_istream<charT, traits>&
//   operator>>(basic_istream<charT, traits>& is, complex<T>& x);

#include "support/test_complex.h"

#include <sstream>

void
run_test()
{
    {
        std::istringstream is("5");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(5, 0));
        assert(is.eof());
    }
    {
        std::istringstream is(" 5 ");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(5, 0));
        assert(is.good());
    }
    {
        std::istringstream is(" 5, ");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(5, 0));
        assert(is.good());
    }
    {
        std::istringstream is(" , 5, ");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        std::istringstream is("5.5 ");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        std::istringstream is(" ( 5.5 ) ");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        std::istringstream is("  5.5)");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        std::istringstream is("(5.5 ");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        std::istringstream is("(5.5,");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        std::istringstream is("( -5.5 , -6.5 )");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(-5.5, -6.5));
        assert(!is.eof());
    }
    {
        std::istringstream is("(-5.5,-6.5)");
        dpl::complex<double> c;
        is >> c;
        assert(c == dpl::complex<double>(-5.5, -6.5));
        assert(!is.eof());
    }
}

int main(int, char**)
{
    // Run on host
    run_test();

    return TestUtils::done();
}
