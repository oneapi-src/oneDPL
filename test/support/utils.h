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

#ifndef _UTILS_H
#define _UTILS_H

// File contains common utilities that tests rely on

// Do not #include <algorithm>, because if we do we will not detect accidental dependencies.
#include "test_config.h"

#include _PSTL_TEST_HEADER(execution)

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <complex>
#include <type_traits>
#include <memory>
#include <sstream>
#include <vector>

#include "utils_const.h"
#include "iterator_utils.h"
#include "utils_sequence.h"
#include "utils_test_base.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "utils_sycl.h"
#endif

namespace TestUtils
{

typedef double float64_t;
typedef float float32_t;

template <class T, ::std::size_t N>
constexpr size_t
const_size(const T (&)[N]) noexcept
{
    return N;
}

// Handy macros for error reporting
#define EXPECT_TRUE(condition, message) ::TestUtils::expect(true, condition, __FILE__, __LINE__, message)
#define EXPECT_FALSE(condition, message) ::TestUtils::expect(false, condition, __FILE__, __LINE__, message)

// Check that expected and actual are equal and have the same type.
#define EXPECT_EQ(expected, actual, message)                                                                           \
    ::TestUtils::expect_equal_val(expected, actual, __FILE__, __LINE__, message)

// Check that sequences started with expected and actual and have had size n are equal and have the same type.
#define EXPECT_EQ_N(expected, actual, n, message)                                                                      \
    ::TestUtils::expect_equal(expected, actual, n, __FILE__, __LINE__, message)

// Check the expected and actual ranges are equal.
#define EXPECT_EQ_RANGES(expected, actual, message)                                                                    \
    ::TestUtils::expect_equal(expected, actual, __FILE__, __LINE__, message)

// Issue error message from outstr, adding a newline.
// Real purpose of this routine is to have a place to hang a breakpoint.
inline void
issue_error_message(::std::stringstream& outstr)
{
    outstr << ::std::endl;
    ::std::cerr << outstr.str();
    ::std::exit(EXIT_FAILURE);
}

inline void
expect(bool expected, bool condition, const char* file, std::int32_t line, const char* message)
{
    if (condition != expected)
    {
        ::std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message;
        issue_error_message(outstr);
    }
}

// Do not change signature to const T&.
// Function must be able to detect const differences between expected and actual.
template <typename T>
void
expect_equal_val(T& expected, T& actual, const char* file, std::int32_t line, const char* message)
{
    if (!(expected == actual))
    {
        ::std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message << ", expected " << expected << " got "
               << actual;
        issue_error_message(outstr);
    }
}

template <typename R1, typename R2>
void
expect_equal(const R1& expected, const R2& actual, const char* file, std::int32_t line, const char* message)
{
    size_t n = expected.size();
    size_t m = actual.size();
    if (n != m)
    {
        ::std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message << ", expected sequence of size " << n
               << " got sequence of size " << m;
        issue_error_message(outstr);
        return;
    }
    size_t error_count = 0;
    for (size_t k = 0; k < n && error_count < 10; ++k)
    {
        if (!(expected[k] == actual[k]))
        {
            ::std::stringstream outstr;
            outstr << "error at " << file << ":" << line << " - " << message << ", at index " << k << " expected "
                   << expected[k] << " got " << actual[k];
            issue_error_message(outstr);
            ++error_count;
        }
    }
}

template <typename T>
void
expect_equal_val(Sequence<T>& expected, Sequence<T>& actual, const char* file, std::int32_t line, const char* message)
{
    expect_equal(expected, actual, file, line, message);
}

template <typename Iterator1, typename Iterator2, typename Size>
void
expect_equal(Iterator1 expected_first, Iterator2 actual_first, Size n, const char* file, std::int32_t line,
             const char* message)
{
    size_t error_count = 0;
    for (size_t k = 0; k < n && error_count < 10; ++k, ++expected_first, ++actual_first)
    {
        if (!(*expected_first == *actual_first))
        {
            ::std::stringstream outstr;
            outstr << "error at " << file << ":" << line << " - " << message << ", at index " << k;
            issue_error_message(outstr);
            ++error_count;
        }
    }
}

struct MemoryChecker
{
    // static counters and state tags
    static ::std::atomic<::std::size_t> alive_object_counter; // initialized outside
    // since it can truncate the value on 32-bit platforms
    // we have to explicitly cast it to desired type to avoid any warnings
    static constexpr ::std::size_t alive_state = ::std::size_t(0xAAAAAAAAAAAAAAAA);
    static constexpr ::std::size_t dead_state = 0; // only used as a set value to cancel alive_state

    ::std::int32_t _value; // object value used for algorithms
    ::std::size_t _state;  // state tag used for checks

    // ctors, dtors, assign ops
    explicit MemoryChecker(::std::int32_t value = 0) : _value(value)
    {
        // check for EXPECT_TRUE(state() != alive_state, ...) has not been done since we cannot guarantee that
        // raw memory for object being constructed does not have a bit sequence being equal to alive_state

        // set constructed state and increment counter for living object
        inc_alive_objects();
        _state = alive_state;
    }
    MemoryChecker(MemoryChecker&& other) : _value(other.value())
    {
        // check for EXPECT_TRUE(state() != alive_state, ...) has not been done since
        // compiler can optimize out the move ctor call that results in false positive failure
        EXPECT_TRUE(other.state() == alive_state, "wrong effect from MemoryChecker(MemoryChecker&&): attempt to "
                                                  "construct an object from non-existing object");
        // set constructed state and increment counter for living object
        inc_alive_objects();
        _state = alive_state;
    }
    MemoryChecker(const MemoryChecker& other) : _value(other.value())
    {
        // check for EXPECT_TRUE(state() != alive_state, ...) has not been done since
        // compiler can optimize out the copy ctor call that results in false positive failure
        EXPECT_TRUE(other.state() == alive_state, "wrong effect from MemoryChecker(const MemoryChecker&): attempt to "
                                                  "construct an object from non-existing object");
        // set constructed state and increment counter for living object
        inc_alive_objects();
        _state = alive_state;
    }
    MemoryChecker&
    operator=(MemoryChecker&& other)
    {
        // check if we do not assign over uninitialized memory
        EXPECT_TRUE(state() == alive_state, "wrong effect from MemoryChecker::operator=(MemoryChecker&& other): "
                                            "attempt to assign to non-existing object");
        EXPECT_TRUE(other.state() == alive_state, "wrong effect from MemoryChecker::operator=(MemoryChecker&& other): "
                                                  "attempt to assign from non-existing object");
        // just assign new value, counter is the same, state is the same
        _value = other.value();

        return *this;
    }
    MemoryChecker&
    operator=(const MemoryChecker& other)
    {
        // check if we do not assign over uninitialized memory
        EXPECT_TRUE(state() == alive_state, "wrong effect from MemoryChecker::operator=(const MemoryChecker& other): "
                                            "attempt to assign to non-existing object");
        EXPECT_TRUE(other.state() == alive_state, "wrong effect from MemoryChecker::operator=(const MemoryChecker& "
                                                  "other): attempt to assign from non-existing object");
        // just assign new value, counter is the same, state is the same
        _value = other.value();

        return *this;
    }
    ~MemoryChecker()
    {
        // check if we do not double destruct the object
        EXPECT_TRUE(state() == alive_state,
                    "wrong effect from ~MemoryChecker(): attempt to destroy non-existing object");
        // set destructed state and decrement counter for living object
        static_cast<volatile ::std::size_t&>(_state) = dead_state;
        dec_alive_objects();
    }

    // getters
    ::std::int32_t
    value() const
    {
        return _value;
    }
    ::std::size_t
    state() const
    {
        return _state;
    }
    static ::std::size_t
    alive_objects()
    {
        return alive_object_counter.load();
    }

  private:
    // setters
    void
    inc_alive_objects()
    {
        alive_object_counter.fetch_add(1);
    }
    void
    dec_alive_objects()
    {
        alive_object_counter.fetch_sub(1);
    }
};

::std::atomic<::std::size_t> MemoryChecker::alive_object_counter{0};

::std::ostream&
operator<<(::std::ostream& os, const MemoryChecker& val)
{
    return (os << val.value());
}
bool
operator==(const MemoryChecker& v1, const MemoryChecker& v2)
{
    return v1.value() == v2.value();
}
bool
operator<(const MemoryChecker& v1, const MemoryChecker& v2)
{
    return v1.value() < v2.value();
}

// Predicates for algorithms
template <typename DataType>
struct is_equal_to
{
    is_equal_to(const DataType& expected) : m_expected(expected) {}
    bool
    operator()(const DataType& actual) const
    {
        return actual == m_expected;
    }

  private:
    DataType m_expected;
};

// Low-quality hash function, returns value between 0 and (1<<bits)-1
// Warning: low-order bits are quite predictable.
inline size_t
HashBits(size_t i, size_t bits)
{
    size_t mask = bits >= 8 * sizeof(size_t) ? ~size_t(0) : (size_t(1) << bits) - 1;
    return (424157 * i ^ 0x24aFa) & mask;
}

// Stateful unary op
template <typename T, typename U>
class Complement
{
    std::int32_t val;

  public:
    Complement(T v) : val(v) {}
    U
    operator()(const T& x) const
    {
        return U(val - x);
    }
};

// Tag used to prevent accidental use of converting constructor, even if use is explicit.
struct OddTag
{
};

class Sum;

// Type with limited set of operations.  Not default-constructible.
// Only available operator is "==".
// Typically used as value type in tests.
class Number
{
    std::int32_t value;
    friend class Add;
    friend class Sum;
    friend class IsMultiple;
    friend class Congruent;
    friend Sum
    operator+(const Sum& x, const Sum& y);

  public:
    Number(std::int32_t val, OddTag) : value(val) {}
    friend bool
    operator==(const Number& x, const Number& y)
    {
        return x.value == y.value;
    }
    friend ::std::ostream&
    operator<<(::std::ostream& o, const Number& d)
    {
        return o << d.value;
    }
};

// Stateful predicate for Number.  Not default-constructible.
class IsMultiple
{
    long modulus;

  public:
    // True if x is multiple of modulus
    bool
    operator()(Number x) const
    {
        return x.value % modulus == 0;
    }
    IsMultiple(long modulus_, OddTag) : modulus(modulus_) {}
};

// Stateful equivalence-class predicate for Number.  Not default-constructible.
class Congruent
{
    long modulus;

  public:
    // True if x and y have same remainder for the given modulus.
    // Note: this is not quite the same as "equivalent modulo modulus" when x and y have different
    // sign, but nonetheless AreCongruent is still an equivalence relationship, which is all
    // we need for testing.
    bool
    operator()(Number x, Number y) const
    {
        return x.value % modulus == y.value % modulus;
    }
    Congruent(long modulus_, OddTag) : modulus(modulus_) {}
};

// Stateful reduction operation for Number
class Add
{
    long bias;

  public:
    explicit Add(OddTag) : bias(1) {}
    Number
    operator()(Number x, const Number& y)
    {
        return Number(x.value + y.value + (bias - 1), OddTag());
    }
};

// Class similar to Number, but has default constructor and +.
class Sum : public Number
{
  public:
    Sum() : Number(0, OddTag()) {}
    Sum(long x, OddTag) : Number(x, OddTag()) {}
    friend Sum
    operator+(const Sum& x, const Sum& y)
    {
        return Sum(x.value + y.value, OddTag());
    }
};

// Type with limited set of operations, which includes an associative but not commutative operation.
// Not default-constructible.
// Typically used as value type in tests involving "GENERALIZED_NONCOMMUTATIVE_SUM".
class MonoidElement
{
    size_t a, b;

  public:
    MonoidElement(size_t a_, size_t b_, OddTag) : a(a_), b(b_) {}
    friend bool
    operator==(const MonoidElement& x, const MonoidElement& y)
    {
        return x.a == y.a && x.b == y.b;
    }
    friend ::std::ostream&
    operator<<(::std::ostream& o, const MonoidElement& x)
    {
        return o << "[" << x.a << ".." << x.b << ")";
    }
    friend class AssocOp;
};

// Stateful associative op for MonoidElement
// It's not really a monoid since the operation is not allowed for any two elements.
// But it's good enough for testing.
class AssocOp
{
    unsigned c;

  public:
    explicit AssocOp(OddTag) : c(5) {}
    MonoidElement
    operator()(const MonoidElement& x, const MonoidElement& y)
    {
        unsigned d = 5;
        EXPECT_EQ(d, c, "state lost");
        EXPECT_EQ(x.b, y.a, "commuted?");

        return MonoidElement(x.a, y.b, OddTag());
    }
};

// Multiplication of matrix is an associative but not commutative operation
// Typically used as value type in tests involving "GENERALIZED_NONCOMMUTATIVE_SUM".
template <typename T>
struct Matrix2x2
{
    T a00, a01, a10, a11;
    Matrix2x2() : a00(1), a01(0), a10(0), a11(1) {}
    Matrix2x2(T x, T y) : a00(0), a01(x), a10(x), a11(y) {}
};

template <typename T>
bool
operator==(const Matrix2x2<T>& left, const Matrix2x2<T>& right)
{
    return left.a00 == right.a00 && left.a01 == right.a01 && left.a10 == right.a10 && left.a11 == right.a11;
}

template <typename T>
struct multiply_matrix
{
    Matrix2x2<T>
    operator()(const Matrix2x2<T>& left, const Matrix2x2<T>& right) const
    {
        Matrix2x2<T> result;
        result.a00 = left.a00 * right.a00 + left.a01 * right.a10;
        result.a01 = left.a00 * right.a01 + left.a01 * right.a11;
        result.a10 = left.a10 * right.a00 + left.a11 * right.a10;
        result.a11 = left.a10 * right.a01 + left.a11 * right.a11;

        return result;
    }
};

template <typename F>
struct NonConstAdapter
{
    F my_f;
    NonConstAdapter(const F& f) : my_f(f) {}

    template <typename... Types>
    auto
    operator()(Types&&... args) -> decltype(::std::declval<F>().
                                            operator()(::std::forward<Types>(args)...))
    {
        return my_f(::std::forward<Types>(args)...);
    }
};

template <typename F>
NonConstAdapter<F>
non_const(const F& f)
{
    return NonConstAdapter<F>(f);
}

// Wrapper for types. It's need for counting of constructing and destructing objects
template <typename T>
class Wrapper
{
  public:
    Wrapper()
        : my_field(::std::make_shared<T>())
    {
        ++my_count;
    }
    Wrapper(const T& input)
        : my_field(::std::make_shared<T>(input))
    {
        ++my_count;
    }
    Wrapper(const Wrapper& input)
        : my_field(input.my_field)
    {
        ++my_count;
    }
    Wrapper(Wrapper&& input)
        : my_field(::std::move(input.my_field))
    {
        ++move_count;
    }
    Wrapper&
    operator=(const Wrapper& input)
    {
        my_field = input.my_field;
        return *this;
    }
    Wrapper&
    operator=(Wrapper&& input)
    {
        my_field = ::std::move(input.my_field);
        ++move_count;
        return *this;
    }
    bool
    operator==(const Wrapper& input) const
    {
        return my_field == input.my_field;
    }
    bool
    operator<(const Wrapper& input) const
    {
        return *my_field < *input.my_field;
    }
    bool
    operator>(const Wrapper& input) const
    {
        return *my_field > *input.my_field;
    }
    friend ::std::ostream&
    operator<<(::std::ostream& stream, const Wrapper& input)
    {
        return stream << *(input.my_field);
    }
    ~Wrapper()
    {
        --my_count;
        if (move_count > 0)
        {
            --move_count;
        }
    }
    T*
    get_my_field() const
    {
        return my_field.get();
    };
    static size_t
    Count()
    {
        return my_count;
    }
    static size_t
    MoveCount()
    {
        return move_count;
    }
    static void
    SetCount(const size_t& n)
    {
        my_count = n;
    }
    static void
    SetMoveCount(const size_t& n)
    {
        move_count = n;
    }

  private:
    static ::std::atomic<size_t> my_count;
    static ::std::atomic<size_t> move_count;
    ::std::shared_ptr<T> my_field;
};

template <typename T>
::std::atomic<size_t> Wrapper<T>::my_count = {0};

template <typename T>
::std::atomic<size_t> Wrapper<T>::move_count = {0};

template <typename InputIterator, typename T, typename BinaryOperation, typename UnaryOperation>
T
transform_reduce_serial(InputIterator first, InputIterator last, T init, BinaryOperation binary_op,
                        UnaryOperation unary_op) noexcept
{
    for (; first != last; ++first)
    {
        init = binary_op(init, unary_op(*first));
    }
    return init;
}

int
done(int is_done = 1)
{
    if (is_done)
    {
#if _PSTL_TEST_SUCCESSFUL_KEYWORD
        ::std::cout << "done\n";
#else
        ::std::cout << "passed\n";
#endif
        return 0;
    }
    else
    {
        ::std::cout << "Skipped\n";
        return _SKIP_RETURN_CODE;
    }
}

// test_algo_basic_* functions are used to execute
// f on a very basic sequence of elements of type T.

// Should be used with unary predicate
template <typename T, typename F>
static void
test_algo_basic_single(F&& f)
{
    size_t N = 10;
    Sequence<T> in(N, [](size_t v) -> T { return T(v); });
    invoke_on_all_host_policies()(::std::forward<F>(f), in.begin());
}

// Should be used with binary predicate
template <typename T, typename F>
static void
test_algo_basic_double(F&& f)
{
    size_t N = 10;
    Sequence<T> in(N, [](size_t v) -> T { return T(v); });
    Sequence<T> out(N, [](size_t v) -> T { return T(v); });
    invoke_on_all_host_policies()(::std::forward<F>(f), in.begin(), out.begin());
}

template <typename Policy, typename F>
static void
invoke_if(Policy&&, F f)
{
    f();
}

template <typename T, typename = bool>
struct can_use_default_less_operator : ::std::false_type
{
};

template <typename T>
struct can_use_default_less_operator<T, decltype(::std::declval<T>() < ::std::declval<T>())> : ::std::true_type
{
};

// An arbitrary binary predicate to simulate a predicate the user providing
// a custom predicate.
template <typename _Tp>
struct UserBinaryPredicate
{
    bool
    operator()(const _Tp& __x, const _Tp& __y) const
    {
        using KeyT = ::std::decay_t<_Tp>;
        return __y != KeyT(1);
    }
};

template <typename _Tp>
struct MaxFunctor
{
    _Tp
    operator()(const _Tp& __x, const _Tp& __y) const
    {
        return (__x < __y) ? __y : __x;
    }
};

// TODO: Investigate why we cannot call ::std::abs on complex
// types with the CUDA backend.
template <typename _Tp>
struct MaxFunctor<::std::complex<_Tp>>
{
    auto
    complex_abs(const ::std::complex<_Tp>& __x) const
    {
        return ::std::sqrt(__x.real() * __x.real() + __x.imag() * __x.imag());
    }
    ::std::complex<_Tp>
    operator()(const ::std::complex<_Tp>& __x, const ::std::complex<_Tp>& __y) const
    {
        return (complex_abs(__x) < complex_abs(__y)) ? __y : __x;
    }
};

template <typename _Tp>
struct MaxAbsFunctor;

// A modification of the functor we use in reduce_by_segment
template <typename _Tp>
struct MaxAbsFunctor<::std::complex<_Tp>>
{
    auto
    complex_abs(const ::std::complex<_Tp>& __x) const
    {
        return ::std::sqrt(__x.real() * __x.real() + __x.imag() * __x.imag());
    }
    ::std::complex<_Tp>
    operator()(const ::std::complex<_Tp>& __x, const ::std::complex<_Tp>& __y) const
    {
        return (complex_abs(__x) < complex_abs(__y)) ? complex_abs(__y) : complex_abs(__x);
    }
};

} /* namespace TestUtils */

#endif // _UTILS_H
