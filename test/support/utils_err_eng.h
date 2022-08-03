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

#include <vector>

#include "utils.h"

#define EXPECT_TRUE_EE(errors, condition, message)                     \
    errors.expect_true(condition, __FILE__, __LINE__, message);

namespace TestUtils
{
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
    struct ErrorContainer_HostPart
    {
        std::vector<ErrorInfo> errors;

        ErrorContainer_HostPart(::std::size_t _max_errors_count)
            : errors(_max_errors_count)
        {
        }

        static cl::sycl::buffer<ErrorInfo, 1> get_sycl_buffer(ErrorContainer_HostPart& host_part)
        {
            const cl::sycl::range<1> numOfItems{host_part.errors.size()};
            return cl::sycl::buffer<ErrorInfo, 1>(host_part.errors.data(), numOfItems);
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
    template <typename TAccessor>
    struct ErrorContairer_KernelPart
    {
        const ::std::size_t max_errors_count = 0;
        TAccessor error_buf_accessor;
        unsigned index = 0;

        ErrorContairer_KernelPart(TAccessor acc, ::std::size_t _max_errors_count)
            : error_buf_accessor(acc), max_errors_count(_max_errors_count)
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
