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

#define EXPECT_TRUE_EE(errorEngine, condition, message) errorEngine.expect_true(condition, __FILE__, __LINE__, message);

namespace TestUtils
{
    ////////////////////////////////////////////////////////////////////////////
    /// struct ErrorEngineHost - error engine for host tests
    struct ErrorEngineHost
    {
        void expect_true(bool condition, const char* file, std::int32_t line, const char* msg)
        {
            expect(true, condition, file, line, msg);
        }
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
    /// struct ErrorEngine_HostPart - host part of error engine for kernel tests
    template <::std::size_t max_errors_count = kMaxKernelErrorsCount>
    struct ErrorEngine_HostPart
    {
        ErrorInfo errors[max_errors_count] = { };

        cl::sycl::buffer<ErrorInfo, 1> get_sycl_buffer()
        {
            const cl::sycl::range<1> numOfItems{ max_errors_count };
            return cl::sycl::buffer<ErrorInfo, 1>(&errors[0], numOfItems);
        }

        /// Process errors, occurred in Kernel, end exit if has errors
        void process_errors()
        {
            bool have_error = false;

            for (const auto& error : errors)
            {
                if (!error.isError)
                    break;

                if (!have_error)
                    std::cout << "Errors in Kernel:" << std::endl;

                expect(true, false,    // true != false -> is error
                       error.file,
                       error.line,
                       error.msg,
                       false);         // Do not exit on error
                have_error = true;     // but save sign to exit later
            }

            if (have_error)
                exit_on_error();
        }
    };

    ////////////////////////////////////////////////////////////////////////////
    /// struct ErrorEngine_KernelPart - kernel part of error engine for kernel tests
    template <typename TAccessor, ::std::size_t max_errors_count = kMaxKernelErrorsCount>
    struct ErrorEngine_KernelPart
    {
        TAccessor error_buf_accessor;
        unsigned index = 0;

        ErrorEngine_KernelPart(TAccessor acc)
            : error_buf_accessor(acc)
        {
        }

        template <typename T>
        std::size_t get_buf_size(const T& buf)
        {
            return sizeof(buf) / sizeof(buf[0]);
        }

        void expect_true(bool condition, const char* file, std::int32_t line, const char* msg)
        {
            if (condition)
                return;

            assert(index < max_errors_count);
            if (index == max_errors_count)
                return;

            copy_string(error_buf_accessor[index].file, get_buf_size(error_buf_accessor[index].file), file);
            error_buf_accessor[index].line = line;
            copy_string(error_buf_accessor[index].msg, get_buf_size(error_buf_accessor[index].msg), msg);
            error_buf_accessor[index].isError = true;

            ++index;
        }

        void copy_string(char* dest, std::size_t dest_size, const char* src)
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
