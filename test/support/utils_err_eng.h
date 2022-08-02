// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __UTILS_ERR_ENG_H
#define __UTILS_ERR_ENG_H

#include "utils.h"

#define EXPECT_TRUE_EE(errors, condition, message)                     \
    errors.expect_true(condition, __FILE__, __LINE__, message);

#define EXPECT_EQ_TYPE_EE(errors, expected_type, test_val)             \
    errors.template expect_eq_type<expected_type>(test_val, __FILE__, __LINE__);

namespace TestUtils
{
    inline
    const char* get_type_name(int)
    {
        static const char name[] = "int";
        return name;
    }

    inline
    const char* get_type_name(float)
    {
        static const char name[] = "float";
        return name;
    }

    inline
    const char* get_type_name(double)
    {
        static const char name[] = "double";
        return name;
    }

    inline const char*
    get_type_name(long double)
    {
        static const char name[] = "long double";
        return name;
    }

    template <typename T>
    const char* get_type_name(T)
    {
        static const char name[] = "unknown";
        return name;
    }

    ////////////////////////////////////////////////////////////////////////////
    /// struct ErrorsContainerOnHost - error container for host tests
    struct ErrorsContainerOnHost
    {
        void expect_true(bool condition, const char* file, std::int32_t line, const char* msg)
        {
            std::string str;
            if (msg)
                str.append(msg);
            str += " (on host)";

            expect(true, condition, file, line, str.c_str(), false);
        }

        template <typename TExpected, typename TTestedVal>
        typename ::std::enable_if<!::std::is_same<typename ::std::decay<TTestedVal>::type, TExpected>::value>::type
        expect_eq_type(TTestedVal testedVal, const char* file, std::int32_t line, const char* msg = "Wrong type")
        {
            std::string str;
            if (msg)
                str.append(msg);
            str += ": ";
            str += get_type_name(TTestedVal());
            str += ", excepted: ";
            str += get_type_name(TExpected());
            str += " (on host)";

            expect(true, false, file, line, str.c_str(), false);

            bHaveErrors = true;
        }

        template <typename TExpected, typename TTestedVal>
        typename ::std::enable_if<::std::is_same<typename ::std::decay<TTestedVal>::type, TExpected>::value>::type
        expect_eq_type(TTestedVal, const char*, std::int32_t, const char* msg = "ok")
        {
            // Type is ok
        }

        bool bHaveErrors = false;
    };

#if TEST_DPCPP_BACKEND_PRESENT

    constexpr ::std::size_t kMaxLenFileName = 255;
    constexpr ::std::size_t kMaxLenMessage  = 50;
    constexpr ::std::size_t kMaxKernelErrorsCount = 3;

    ////////////////////////////////////////////////////////////////////////////
    /// struct ErrorInfo - describe error info
    struct ErrorInfo
    {
        char file[kMaxLenFileName] = { };       // File name
        int  line                  = 0;         // Line no
        char msg[kMaxLenMessage]   = { };       // Error message
        bool isError               = false;     // Sign that no more errors in owner buffer: true setup to last not filled error record
    };

    ////////////////////////////////////////////////////////////////////////////
    /// struct ErrorContainer_HostPart - host part of error container for kernel tests
    template <::std::size_t max_errors_count = kMaxKernelErrorsCount>
    struct ErrorContainer_HostPart
    {
        ErrorInfo errors[max_errors_count] = { };

        cl::sycl::buffer<ErrorInfo, 1> get_sycl_buffer()
        {
            const cl::sycl::range<1> numOfItems{ max_errors_count };
            return cl::sycl::buffer<ErrorInfo, 1>(&errors[0], numOfItems);
        }

        bool have_errors() const
        {
            return errors[0].isError;
        }

        /// Process errors, occurred in Kernel, end exit if has errors
        void process_errors()
        {
            for (const auto& error : errors)
            {
                if (!error.isError)
                    break;

                expect(true, false,    // true != false -> is error
                       error.file,
                       error.line,
                       error.msg,
                       false);         // Do not exit on error
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////
    /// struct ErrorContairer_KernelPart - kernel part of error container for kernel tests
    template <typename TAccessor, ::std::size_t max_errors_count = kMaxKernelErrorsCount>
    struct ErrorContairer_KernelPart
    {
        TAccessor error_buf_accessor;
        unsigned index = 0;

        ErrorContairer_KernelPart(TAccessor acc)
            : error_buf_accessor(acc)
        {
        }

        template <typename T>
        std::size_t get_buf_size(const T& buf) const
        {
            return sizeof(buf) / sizeof(buf[0]);
        }

        void expect_true(bool condition, const char* file, std::int32_t line, const char* msg)
        {
            if (!condition)
                add_error_info(file, line, msg);
        }

        template <typename TExpected, typename TTestedVal>
        typename ::std::enable_if<!::std::is_same<typename ::std::decay<TTestedVal>::type, TExpected>::value>::type
        expect_eq_type(TTestedVal testedVal, const char* file, std::int32_t line, const char* msg = "Wrong type")
        {
            ErrorInfo eiTemp;
            copy_string(eiTemp.msg, get_buf_size(eiTemp.msg), msg);
            copy_string(eiTemp.msg + get_str_len(eiTemp.msg), get_buf_size(eiTemp.msg) - get_str_len(eiTemp.msg), ": ");
            copy_string(eiTemp.msg + get_str_len(eiTemp.msg), get_buf_size(eiTemp.msg) - get_str_len(eiTemp.msg), get_type_name(TTestedVal()));
            copy_string(eiTemp.msg + get_str_len(eiTemp.msg), get_buf_size(eiTemp.msg) - get_str_len(eiTemp.msg), ", excepted: ");
            copy_string(eiTemp.msg + get_str_len(eiTemp.msg), get_buf_size(eiTemp.msg) - get_str_len(eiTemp.msg), get_type_name(TExpected()));

            add_error_info(file, line, eiTemp.msg);
        }

        template <typename TExpected, typename TTestedVal>
        typename ::std::enable_if<::std::is_same<typename ::std::decay<TTestedVal>::type, TExpected>::value>::type
        expect_eq_type(TTestedVal, const char*, std::int32_t, const char* msg = "ok")
        {
            // Type is ok
        }

        void add_error_info(const char* file, std::int32_t line, const char* msg)
        {
            assert(index < max_errors_count);
            if (index == max_errors_count)
                return;

            auto& ei = error_buf_accessor[index];
            copy_string(ei.file, get_buf_size(ei.file), file);
            ei.line = line;
            copy_string(ei.msg, get_buf_size(ei.msg), msg);
            copy_string(ei.msg + get_str_len(ei.msg), get_buf_size(ei.msg) - get_str_len(ei.msg), " (in Kernel)");
            ei.isError = true;

            ++index;
        }

        ::std::size_t get_str_len(const char* buf) const
        {
            ::std::size_t len = 0;

            for (; *buf != 0x00; ++buf)
                ++len;

            return len;
        }

        void copy_string(char* dest, std::size_t dest_size, const char* src) const
        {
            assert(dest != nullptr);

            if (src == nullptr)
                return;

            if (dest_size == 0)
                return;

            auto pDest = dest;
            auto pSrc = src;

            bool bNeedBreak = false;
            for (size_t i = 0; i < (dest_size - 1) && !bNeedBreak; ++i)
            {
                *pDest = *pSrc;

                bNeedBreak = *pSrc == 0x00;

                ++pDest;
                ++pSrc;
            }

            *pDest = 0x00;
        }
    };
#endif

} /* namespace TestUtils */

#endif // __UTILS_ERR_ENG_H
