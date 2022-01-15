// -*- C++ -*-
//===-- utils.h -----------------------------------------------------------===//
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
#include <memory>
#include <sstream>
#include <vector>

#include "iterator_utils.h"

#define _SKIP_RETURN_CODE 77

// Test data ranges other than those that start at the beginning of an input.
const int max_n = 100000;
const int inout1_offset = 3;
const int inout2_offset = 5;
const int inout3_offset = 7;
const int inout4_offset = 9;

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

template <typename T>
class Sequence;

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

template <typename Iterator, typename F>
void
fill_data(Iterator first, Iterator last, F f)
{
    typedef typename ::std::iterator_traits<Iterator>::value_type T;
    for (::std::size_t i = 0; first != last; ++first, ++i)
    {
        *first = T(f(i));
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

// Sequence<T> is a container of a sequence of T with lots of kinds of iterators.
// Prefixes on begin/end mean:
//      c = "const"
//      f = "forward"
// No prefix indicates non-const random-access iterator.
template <typename T>
class Sequence
{
    ::std::vector<T> m_storage;

  public:
    typedef typename ::std::vector<T>::iterator iterator;
    typedef typename ::std::vector<T>::const_iterator const_iterator;
    typedef ForwardIterator<iterator, ::std::forward_iterator_tag> forward_iterator;
    typedef ForwardIterator<const_iterator, ::std::forward_iterator_tag> const_forward_iterator;

    typedef BidirectionalIterator<iterator, ::std::bidirectional_iterator_tag> bidirectional_iterator;
    typedef BidirectionalIterator<const_iterator, ::std::bidirectional_iterator_tag> const_bidirectional_iterator;

    typedef T value_type;
    explicit Sequence(size_t size) : m_storage(size) {}

    // Construct sequence [f(0), f(1), ... f(size-1)]
    // f can rely on its invocations being sequential from 0 to size-1.
    template <typename Func>
    Sequence(size_t size, Func f)
    {
        m_storage.reserve(size);
        // Use push_back because T might not have a default constructor
        for (size_t k = 0; k < size; ++k)
            m_storage.push_back(T(f(k)));
    }
    Sequence(const ::std::initializer_list<T>& data) : m_storage(data) {}

    const_iterator
    begin() const
    {
        return m_storage.begin();
    }
    const_iterator
    end() const
    {
        return m_storage.end();
    }
    iterator
    begin()
    {
        return m_storage.begin();
    }
    iterator
    end()
    {
        return m_storage.end();
    }
    const_iterator
    cbegin() const
    {
        return m_storage.cbegin();
    }
    const_iterator
    cend() const
    {
        return m_storage.cend();
    }
    forward_iterator
    fbegin()
    {
        return forward_iterator(m_storage.begin());
    }
    forward_iterator
    fend()
    {
        return forward_iterator(m_storage.end());
    }
    const_forward_iterator
    cfbegin() const
    {
        return const_forward_iterator(m_storage.cbegin());
    }
    const_forward_iterator
    cfend() const
    {
        return const_forward_iterator(m_storage.cend());
    }
    const_forward_iterator
    fbegin() const
    {
        return const_forward_iterator(m_storage.cbegin());
    }
    const_forward_iterator
    fend() const
    {
        return const_forward_iterator(m_storage.cend());
    }

    const_bidirectional_iterator
    cbibegin() const
    {
        return const_bidirectional_iterator(m_storage.cbegin());
    }
    const_bidirectional_iterator
    cbiend() const
    {
        return const_bidirectional_iterator(m_storage.cend());
    }

    bidirectional_iterator
    bibegin()
    {
        return bidirectional_iterator(m_storage.begin());
    }
    bidirectional_iterator
    biend()
    {
        return bidirectional_iterator(m_storage.end());
    }

    ::std::size_t
    size() const
    {
        return m_storage.size();
    }
    const T*
    data() const
    {
        return m_storage.data();
    }
    typename ::std::vector<T>::reference operator[](size_t j) { return m_storage[j]; }
    typename ::std::vector<T>::const_reference operator[](size_t j) const { return m_storage[j]; }

    // Fill with given value
    void
    fill(const T& value)
    {
        for (size_t i = 0; i < m_storage.size(); i++)
            m_storage[i] = value;
    }

    void
    print() const;

    template <typename Func>
    void
    fill(Func f)
    {
        fill_data(m_storage.begin(), m_storage.end(), f);
    }
};

template <typename T>
void
Sequence<T>::print() const
{
    ::std::cout << "size = " << size() << ": { ";
    ::std::copy(begin(), end(), ::std::ostream_iterator<T>(::std::cout, " "));
    ::std::cout << " } " << ::std::endl;
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

// Invoke op(policy,rest...) for each non-hetero policy.
struct invoke_on_all_host_policies
{
    template <typename Op, typename... T>
    void
    operator()(Op op, T&&... rest)
    {
        using namespace oneapi::dpl::execution;

#if !TEST_ONLY_HETERO_POLICIES
        // Try static execution policies
        invoke_on_all_iterator_types()(seq, op, ::std::forward<T>(rest)...);
        invoke_on_all_iterator_types()(unseq, op, ::std::forward<T>(rest)...);
        invoke_on_all_iterator_types()(par, op, ::std::forward<T>(rest)...);
        invoke_on_all_iterator_types()(par_unseq, op, ::std::forward<T>(rest)...);

#endif
    }
};

template <::std::size_t CallNumber = 0>
struct invoke_on_all_policies
{
    template <typename Op, typename... T>
    void
    operator()(Op op, T&&... rest)
    {

        invoke_on_all_host_policies()(op, ::std::forward<T>(rest)...);
#if TEST_DPCPP_BACKEND_PRESENT
        invoke_on_all_hetero_policies<CallNumber>()(op, ::std::forward<T>(rest)...);
#endif
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
    {
        my_field = ::std::shared_ptr<T>(new T());
        ++my_count;
    }
    Wrapper(const T& input)
    {
        my_field = ::std::shared_ptr<T>(new T(input));
        ++my_count;
    }
    Wrapper(const Wrapper& input)
    {
        my_field = input.my_field;
        ++my_count;
    }
    Wrapper(Wrapper&& input)
    {
        my_field = input.my_field;
        input.my_field = nullptr;
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
        my_field = input.my_field;
        input.my_field = nullptr;
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

// Used with algorithms that have two input sequences and one output sequences
template <typename T, typename TestName>
void
test_algo_three_sequences()
{
    for (size_t n = 2; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> inout1(max_n + inout1_offset);
        Sequence<T> inout2(max_n + inout2_offset);
        Sequence<T> inout3(max_n + inout3_offset);

        // create iterators
        auto inout1_offset_first = std::begin(inout1) + inout1_offset;
        auto inout2_offset_first = std::begin(inout2) + inout2_offset;
        auto inout3_offset_first = std::begin(inout3) + inout3_offset;

        invoke_on_all_host_policies()(TestName(), inout1_offset_first, inout1_offset_first + n, inout2_offset_first,
                                      inout2_offset_first + n, inout3_offset_first, inout3_offset_first + n, n);
    }
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

} /* namespace TestUtils */
