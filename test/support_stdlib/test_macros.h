// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_MACROS_HPP
#define SUPPORT_TEST_MACROS_HPP

// Attempt to get STL specific macros like _LIBCPP_VERSION using the most
// minimal header possible. If we're testing libc++, we should use `<__config>`.
// If <__config> isn't available, fall back to <ciso646>.
#ifdef __has_include
# if __has_include("<__config>")
#   include <__config>
#   define TEST_IMP_INCLUDED_HEADER
# endif
#endif
#ifndef TEST_IMP_INCLUDED_HEADER
#include <ciso646>
#endif

#define TEST_STRINGIZE_IMPL(x) #x
#define TEST_STRINGIZE(x) TEST_STRINGIZE_IMPL(x)

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#endif

#define TEST_CONCAT1(X, Y) X##Y
#define TEST_CONCAT(X, Y) TEST_CONCAT1(X, Y)

#ifdef __has_feature
#define TEST_HAS_FEATURE(X) __has_feature(X)
#else
#define TEST_HAS_FEATURE(X) 0
#endif

#ifndef __has_include
#define __has_include(...) 0
#endif

#ifdef __has_extension
#define TEST_HAS_EXTENSION(X) __has_extension(X)
#else
#define TEST_HAS_EXTENSION(X) 0
#endif

#ifdef __has_warning
#define TEST_HAS_WARNING(X) __has_warning(X)
#else
#define TEST_HAS_WARNING(X) 0
#endif

#ifdef __has_builtin
#define TEST_HAS_BUILTIN(X) __has_builtin(X)
#else
#define TEST_HAS_BUILTIN(X) 0
#endif
#ifdef __is_identifier
// '__is_identifier' returns '0' if '__x' is a reserved identifier provided by
// the compiler and '1' otherwise.
#define TEST_HAS_BUILTIN_IDENTIFIER(X) !__is_identifier(X)
#else
#define TEST_HAS_BUILTIN_IDENTIFIER(X) 0
#endif

#if defined(__EDG__)
# define TEST_COMPILER_EDG
#elif defined(__clang__)
# define TEST_COMPILER_CLANG
# if defined(__apple_build_version__)
#  define TEST_COMPILER_APPLE_CLANG
# endif
#elif defined(_MSC_VER)
# define TEST_COMPILER_C1XX
#elif defined(__GNUC__)
# define TEST_COMPILER_GCC
#endif

#if defined(__apple_build_version__)
#define TEST_APPLE_CLANG_VER (__clang_major__ * 100) + __clang_minor__
#elif defined(__clang_major__)
#define TEST_CLANG_VER (__clang_major__ * 100) + __clang_minor__
#elif defined(__GNUC__)
#define TEST_GCC_VER (__GNUC__ * 100 + __GNUC_MINOR__)
#define TEST_GCC_VER_NEW (TEST_GCC_VER * 10 + __GNUC_PATCHLEVEL__)
#endif

/* Make a nice name for the standard version */
#ifndef TEST_STD_VER
#if  __cplusplus <= 199711L
# define TEST_STD_VER 3
#elif __cplusplus <= 201103L
# define TEST_STD_VER 11
#elif __cplusplus <= 201402L
# define TEST_STD_VER 14
#elif __cplusplus <= 201703L
# define TEST_STD_VER 17
#elif __cplusplus <= 202002L
# define TEST_STD_VER 20
#else
# define TEST_STD_VER 99    // greater than current standard
// This is deliberately different than _LIBCPP_STD_VER to discourage matching them up.
#endif
#endif

// Attempt to deduce the GLIBC version
#if (defined(__has_include) && __has_include(<features.h>)) || \
    defined(__linux__)
#include <features.h>
#if defined(__GLIBC_PREREQ)
#define TEST_HAS_GLIBC
#define TEST_GLIBC_PREREQ(major, minor) __GLIBC_PREREQ(major, minor)
#endif
#endif

#if TEST_STD_VER >= 11
#define TEST_ALIGNOF(...) alignof(__VA_ARGS__)
#define TEST_ALIGNAS(...) alignas(__VA_ARGS__)
#define TEST_CONSTEXPR constexpr
#define TEST_NOEXCEPT noexcept
#define TEST_NOEXCEPT_FALSE noexcept(false)
#define TEST_NOEXCEPT_COND(...) noexcept(__VA_ARGS__)
# if TEST_STD_VER >= 14
#   define TEST_CONSTEXPR_CXX14 constexpr
# else
#   define TEST_CONSTEXPR_CXX14
# endif
# if TEST_STD_VER > 14
#   define TEST_THROW_SPEC(...)
# else
#   define TEST_THROW_SPEC(...) throw(__VA_ARGS__)
# endif
#else
#if defined(TEST_COMPILER_CLANG)
# define TEST_ALIGNOF(...) _Alignof(__VA_ARGS__)
#else
# define TEST_ALIGNOF(...) __alignof(__VA_ARGS__)
#endif
#define TEST_ALIGNAS(...) __attribute__((__aligned__(__VA_ARGS__)))
#define TEST_CONSTEXPR
#define TEST_CONSTEXPR_CXX14
#define TEST_NOEXCEPT throw()
#define TEST_NOEXCEPT_FALSE
#define TEST_NOEXCEPT_COND(...)
#define TEST_THROW_SPEC(...) throw(__VA_ARGS__)
#endif

// Sniff out to see if the underling C library has C11 features
// Note that at this time (July 2018), MacOS X and iOS do NOT.
// This is cribbed from __config; but lives here as well because we can't assume libc++
#if __ISO_C_VISIBLE >= 2011 || TEST_STD_VER >= 11
#  if defined(__FreeBSD__)
//  Specifically, FreeBSD does NOT have timespec_get, even though they have all
//  the rest of C11 - this is PR#38495
#    define TEST_HAS_C11_FEATURES
#  elif defined(__Fuchsia__) || defined(__wasi__)
#    define TEST_HAS_C11_FEATURES
#    define TEST_HAS_TIMESPEC_GET
#  elif defined(__linux__)
// This block preserves the old behavior used by include/__config:
// _LIBCPP_GLIBC_PREREQ would be defined to 0 if __GLIBC_PREREQ was not
// available. The configuration here may be too vague though, as Bionic, uClibc,
// newlib, etc may all support these features but need to be configured.
#    if defined(TEST_GLIBC_PREREQ)
#      if TEST_GLIBC_PREREQ(2, 17)
#        define TEST_HAS_TIMESPEC_GET
#        define TEST_HAS_C11_FEATURES
#      endif
#    elif defined(_LIBCPP_HAS_MUSL_LIBC)
#      define TEST_HAS_C11_FEATURES
#      define TEST_HAS_TIMESPEC_GET
#    endif
#  elif defined(_WIN32)
#    if defined(_MSC_VER) && !defined(__MINGW32__)
#      define TEST_HAS_C11_FEATURES // Using Microsoft's C Runtime library
#      define TEST_HAS_TIMESPEC_GET
#    endif
#  endif
#endif

/* Features that were introduced in C++14 */
#if TEST_STD_VER >= 14
#define TEST_HAS_EXTENDED_CONSTEXPR
#define TEST_HAS_VARIABLE_TEMPLATES
#endif

/* Features that were introduced in C++17 */
#if TEST_STD_VER >= 17
#endif

/* Features that were introduced after C++17 */
#if TEST_STD_VER > 17
#endif


#define TEST_ALIGNAS_TYPE(...) TEST_ALIGNAS(TEST_ALIGNOF(__VA_ARGS__))

#if !TEST_HAS_FEATURE(cxx_rtti) && !defined(__cpp_rtti) \
    && !defined(__GXX_RTTI)
#define TEST_HAS_NO_RTTI
#endif

#if !TEST_HAS_FEATURE(cxx_exceptions) && !defined(__cpp_exceptions) \
     && !defined(__EXCEPTIONS)
#define TEST_HAS_NO_EXCEPTIONS
#endif

#if TEST_HAS_FEATURE(address_sanitizer) || TEST_HAS_FEATURE(memory_sanitizer) || \
    TEST_HAS_FEATURE(thread_sanitizer)
#define TEST_HAS_SANITIZERS
#endif

#if defined(_LIBCPP_NORETURN)
#define TEST_NORETURN _LIBCPP_NORETURN
#else
#define TEST_NORETURN [[noreturn]]
#endif

#if defined(_LIBCPP_HAS_NO_ALIGNED_ALLOCATION) || \
  (!(TEST_STD_VER > 14 || \
    (defined(__cpp_aligned_new) && __cpp_aligned_new >= 201606L)))
#define TEST_HAS_NO_ALIGNED_ALLOCATION
#endif

#if TEST_STD_VER > 17
#define TEST_CONSTINIT constinit
#elif defined(_LIBCPP_CONSTINIT)
#define TEST_CONSTINIT _LIBCPP_CONSTINIT
#else
#define TEST_CONSTINIT
#endif

#if !defined(__cpp_impl_three_way_comparison) \
    && (!defined(_MSC_VER) || defined(__clang__) || _MSC_VER < 1920 || _MSVC_LANG <= 201703L)
#define TEST_HAS_NO_SPACESHIP_OPERATOR
#endif

#if defined(_LIBCPP_SAFE_STATIC)
#define TEST_SAFE_STATIC _LIBCPP_SAFE_STATIC
#else
#define TEST_SAFE_STATIC
#endif

// FIXME: Fix this feature check when either (A) a compiler provides a complete
// implementation, or (b) a feature check macro is specified
#if !defined(_MSC_VER) || defined(__clang__) || _MSC_VER < 1920 || _MSVC_LANG <= 201703L
#define TEST_HAS_NO_SPACESHIP_OPERATOR
#endif

#if TEST_STD_VER < 11
#define ASSERT_NOEXCEPT(...)
#define ASSERT_NOT_NOEXCEPT(...)
#else
#define ASSERT_NOEXCEPT(...) \
    static_assert(noexcept(__VA_ARGS__), "Operation must be noexcept")

#define ASSERT_NOT_NOEXCEPT(...) \
    static_assert(!noexcept(__VA_ARGS__), "Operation must NOT be noexcept")
#endif

/* Macros for testing libc++ specific behavior and extensions */
#if defined(_LIBCPP_VERSION)
#define LIBCPP_ASSERT(...) assert(__VA_ARGS__)
#define LIBCPP_STATIC_ASSERT(...) static_assert(__VA_ARGS__)
#define LIBCPP_ASSERT_NOEXCEPT(...) ASSERT_NOEXCEPT(__VA_ARGS__)
#define LIBCPP_ASSERT_NOT_NOEXCEPT(...) ASSERT_NOT_NOEXCEPT(__VA_ARGS__)
#define LIBCPP_ONLY(...) __VA_ARGS__
#else
#define LIBCPP_ASSERT(...) static_assert(true, "")
#define LIBCPP_STATIC_ASSERT(...) static_assert(true, "")
#define LIBCPP_ASSERT_NOEXCEPT(...) static_assert(true, "")
#define LIBCPP_ASSERT_NOT_NOEXCEPT(...) static_assert(true, "")
#define LIBCPP_ONLY(...) static_assert(true, "")
#endif

#define TEST_IGNORE_NODISCARD (void)

namespace test_macros_detail {
template <class T, class U>
struct is_same { enum { value = 0};} ;
template <class T>
struct is_same<T, T> { enum {value = 1}; };
} // namespace test_macros_detail

#define ASSERT_SAME_TYPE(...) \
    static_assert((test_macros_detail::is_same<__VA_ARGS__>::value), \
                 "Types differ unexpectedly")

#ifndef TEST_HAS_NO_EXCEPTIONS
#define TEST_THROW(...) throw __VA_ARGS__
#else
#if defined(__GNUC__)
#define TEST_THROW(...) __builtin_abort()
#else
#include <stdlib.h>
#define TEST_THROW(...) ::abort()
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
template <class Tp>
inline
void DoNotOptimize(Tp const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
inline void DoNotOptimize(Tp& value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}
#else
#include <intrin.h>
template <class Tp>
inline void DoNotOptimize(Tp const& value) {
  const volatile void* volatile unused = __builtin_addressof(value);
  static_cast<void>(unused);
  _ReadWriteBarrier();
}
#endif

#if defined(__GNUC__)
#define TEST_ALWAYS_INLINE __attribute__((always_inline))
#define TEST_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define TEST_ALWAYS_INLINE __forceinline
#define TEST_NOINLINE __declspec(noinline)
#else
#define TEST_ALWAYS_INLINE
#define TEST_NOINLINE
#endif

#ifdef _WIN32
#define TEST_NOT_WIN32(...) ((void)0)
#else
#define TEST_NOT_WIN32(...) __VA_ARGS__
#endif

#if defined(TEST_WINDOWS_DLL) ||defined(__MVS__) || defined(_AIX)
// Macros for waiving cases when we can't count allocations done within
// the library implementation.
//
// On Windows, when libc++ is built as a DLL, references to operator new/delete
// within the DLL are bound at link time to the operator new/delete within
// the library; replacing them in the user executable doesn't override the
// calls within the library.
//
// The same goes on IBM zOS.
// The same goes on AIX.
#define ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(...) ((void)(__VA_ARGS__))
#define TEST_SUPPORTS_LIBRARY_INTERNAL_ALLOCATIONS 0
#else
#define ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(...) assert(__VA_ARGS__)
#define TEST_SUPPORTS_LIBRARY_INTERNAL_ALLOCATIONS 1
#endif

#if (defined(TEST_WINDOWS_DLL) && !defined(_MSC_VER)) ||                      \
    defined(__MVS__)
// Normally, a replaced e.g. 'operator new' ends up used if the user code
// does a call to e.g. 'operator new[]'; it's enough to replace the base
// versions and have it override all of them.
//
// When the fallback operators are located within the libc++ library and we
// can't override the calls within it (see above), this fallback mechanism
// doesn't work either.
//
// On Windows, when using the MSVC vcruntime, the operator new/delete fallbacks
// are linked separately from the libc++ library, linked statically into
// the end user executable, and these fallbacks work even in DLL configurations.
// In MinGW configurations when built as a DLL, and on zOS, these fallbacks
// don't work though.
#define ASSERT_WITH_OPERATOR_NEW_FALLBACKS(...) ((void)(__VA_ARGS__))
#else
#define ASSERT_WITH_OPERATOR_NEW_FALLBACKS(...) assert(__VA_ARGS__)
#endif

#ifdef _WIN32
#define TEST_WIN_NO_FILESYSTEM_PERMS_NONE
#endif

// Support for carving out parts of the test suite, like removing wide characters, etc.
#if defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   define TEST_HAS_NO_WIDE_CHARACTERS
#endif

#if defined(_LIBCPP_HAS_NO_UNICODE)
#   define TEST_HAS_NO_UNICODE
#elif defined(_MSVC_EXECUTION_CHARACTER_SET) && _MSVC_EXECUTION_CHARACTER_SET != 65001
#   define TEST_HAS_NO_UNICODE
#endif

#if defined(_LIBCPP_HAS_NO_INT128) || defined(_MSVC_STL_VERSION)
#   define TEST_HAS_NO_INT128
#endif

#if defined(_LIBCPP_HAS_NO_LOCALIZATION)
#  define TEST_HAS_NO_LOCALIZATION
#endif

#if TEST_STD_VER <= 17 || !defined(__cpp_char8_t)
#  define TEST_HAS_NO_CHAR8_T
#endif

#if defined(_LIBCPP_HAS_NO_THREADS)
#  define TEST_HAS_NO_THREADS
#endif

#if defined(_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY)
#  define TEST_HAS_NO_FILESYSTEM_LIBRARY
#endif

#if defined(_LIBCPP_HAS_NO_FGETPOS_FSETPOS)
#  define TEST_HAS_NO_FGETPOS_FSETPOS
#endif

#if defined(TEST_COMPILER_CLANG)
#  define TEST_DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#  define TEST_DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#  define TEST_CLANG_DIAGNOSTIC_IGNORED(str) _Pragma(TEST_STRINGIZE(clang diagnostic ignored str))
#  define TEST_GCC_DIAGNOSTIC_IGNORED(str)
#  define TEST_MSVC_DIAGNOSTIC_IGNORED(num)
#elif defined(TEST_COMPILER_GCC)
#  define TEST_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#  define TEST_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#  define TEST_CLANG_DIAGNOSTIC_IGNORED(str)
#  define TEST_GCC_DIAGNOSTIC_IGNORED(str) _Pragma(TEST_STRINGIZE(GCC diagnostic ignored str))
#  define TEST_MSVC_DIAGNOSTIC_IGNORED(num)
#elif defined(TEST_COMPILER_MSVC)
#  define TEST_DIAGNOSTIC_PUSH _Pragma("warning(push)")
#  define TEST_DIAGNOSTIC_POP _Pragma("warning(pop)")
#  define TEST_CLANG_DIAGNOSTIC_IGNORED(str)
#  define TEST_GCC_DIAGNOSTIC_IGNORED(str)
#  define TEST_MSVC_DIAGNOSTIC_IGNORED(num) _Pragma(TEST_STRINGIZE(warning(disable: num)))
#else
#  define TEST_DIAGNOSTIC_PUSH
#  define TEST_DIAGNOSTIC_POP
#  define TEST_CLANG_DIAGNOSTIC_IGNORED(str)
#  define TEST_GCC_DIAGNOSTIC_IGNORED(str)
#  define TEST_MSVC_DIAGNOSTIC_IGNORED(num)
#endif

#if __has_cpp_attribute(msvc::no_unique_address)
#define TEST_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#elif __has_cpp_attribute(no_unique_address)
#define TEST_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define TEST_NO_UNIQUE_ADDRESS
#endif

#endif // SUPPORT_TEST_MACROS_HPP
