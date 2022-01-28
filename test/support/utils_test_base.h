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
#ifndef UTILS_TEST_BASE
#define UTILS_TEST_BASE

#include <memory>

#include "utils_const.h"
#include "utils_sequence.h"
#include "utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "sycl_alloc_utils.h"
#endif // TEST_DPCPP_BACKEND_PRESENT

namespace TestUtils
{
////////////////////////////////////////////////////////////////////////////////
//
enum class UDTKind
{
    eKeys = 0,
    eVals,
    eRes
};

template <typename TEnum>
auto
enum_val_to_index(TEnum enumVal)
    -> decltype(static_cast<typename ::std::underlying_type<TEnum>::type>(enumVal))
{
    return static_cast<typename ::std::underlying_type<TEnum>::type>(enumVal);
}

template <typename TestValueType>
struct test_base_data_visitor;

////////////////////////////////////////////////////////////////////////////////
//
template <typename TestValueType>
struct test_base_data
{
    virtual bool visit(test_base_data_visitor<TestValueType>* visitor) = 0;
};

////////////////////////////////////////////////////////////////////////////////
///
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType>
struct test_base_data_usm : test_base_data<TestValueType>
{
    struct Data
    {
        using TSourceData = usm_data_transfer_base<TestValueType>;
        using TSourceDataPtr = ::std::unique_ptr<TSourceData>;

        sycl::usm::alloc alloc_type;
        TSourceDataPtr   src_data_usm;
        ::std::size_t    offset = 0;

        template<typename _Size>
        Data(sycl::usm::alloc __alloc_type, sycl::queue __q, _Size __sz, ::std::size_t __offset)
            : alloc_type(__alloc_type)
            , src_data_usm(create_usm_data_transfer<TestValueType>(__alloc_type, __q, __sz + __offset))
            , offset(__offset)
        {
        }

        template <typename Pred>
        void process_usm_data_transfer(Pred& pred)
        {
            auto& obj = *src_data_usm.get();

            switch (alloc_type)
            {
                case sycl::usm::alloc::shared :
                    {
                        auto& usm_data_transfer_obj = reinterpret_cast<usm_data_transfer<sycl::usm::alloc::shared, TestValueType>&>(obj);
                        pred(usm_data_transfer_obj);
                    }
                    return;

                case sycl::usm::alloc::device :
                    {
                        auto& usm_data_transfer_obj = reinterpret_cast<usm_data_transfer<sycl::usm::alloc::device, TestValueType>&>(obj);
                        pred(usm_data_transfer_obj);
                    }
                    return;

                case sycl::usm::alloc::host:
                case sycl::usm::alloc::unknown:
                    break;
            }

            assert(false);
            throw ::std::runtime_error("Ivalid alloc type");
        }
    };
    ::std::vector<Data> data;

    struct InitParam
    {
        ::std::size_t size   = 0;
        ::std::size_t offset = 0;
    };

    test_base_data_usm(sycl::usm::alloc alloc_type, sycl::queue __q, ::std::initializer_list<InitParam> init)
    {
        for (auto& initParam : init)
            data.emplace_back(alloc_type, __q, initParam.size, initParam.offset);
    }

    struct PredGetStartFrom
    {
        ::std::size_t index = 0;
        TestValueType* from_ptr = nullptr;

        template <sycl::usm::alloc alloc_type>
        void operator()(usm_data_transfer<alloc_type, TestValueType>& obj)
        {
            from_ptr = obj.get_data();
        }
    };

    TestValueType* get_start_from(::std::size_t index)
    {
        auto& data_item = data.at(index);

        PredGetStartFrom pred;
        data_item.process_usm_data_transfer(pred);

        return pred.from_ptr + data_item.offset;
    }

    // test_base_data

    virtual bool visit(test_base_data_visitor<TestValueType>* visitor) override;
};
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
////////////////////////////////////////////////////////////////////////////////
template <typename TestValueType>
struct test_base_data_buffer : test_base_data<TestValueType>
{
    struct Data
    {
        using TSourceData = sycl::buffer<TestValueType, 1>;

        TSourceData   src_data_buf;
        ::std::size_t offset = 0;

        template<typename _Size>
        Data(_Size __sz, ::std::size_t __offset)
            : src_data_buf(sycl::range<1>(__sz + __offset))
            , offset(__offset)
        {
        }
    };
    ::std::vector<Data> data;

    test_base_data_buffer(::std::initializer_list<Data> init)
        : data(init)
    {
    }

    sycl::buffer<TestValueType, 1>& get_buffer(::std::size_t index)
    {
        return data.at(index).src_data_buf;
    }

    auto get_start_from(::std::size_t index)
        -> decltype(oneapi::dpl::begin(data.at(index).src_data_buf) + data.at(index).offset)
    {
        return oneapi::dpl::begin(data.at(index).src_data_buf) + data.at(index).offset;
    }

    // test_base_data

    virtual bool visit(test_base_data_visitor<TestValueType>* visitor) override;
};
#endif // TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
template <typename TestValueType>
struct test_base_data_sequence : test_base_data<TestValueType>
{
    struct Data
    {
        using TSourceData = Sequence<TestValueType>;

        TSourceData   src_data_seq;
        ::std::size_t offset = 0;

        Data(::std::size_t size, ::std::size_t __offset)
            : src_data_seq(size)
            , offset(__offset)
        {
        }
    };
    ::std::vector<Data> data;

    test_base_data_sequence(::std::initializer_list<Data> init)
        : data(init)
    {
    }

    auto get_start_from(::std::size_t index)
        -> decltype(data.at(index).src_data_seq.begin() + data.at(index).offset)
    {
        return data.at(index).src_data_seq.begin() + data.at(index).offset;
    }

    // test_base_data

    virtual bool visit(test_base_data_visitor<TestValueType>* visitor) override;
};

////////////////////////////////////////////////////////////////////////////////
template <typename TestValueType>
struct test_base_data_visitor
{
#if TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_usm     <TestValueType>& obj) = 0;
    virtual bool on_visit(::std::size_t nIndex, test_base_data_buffer  <TestValueType>& obj) = 0;
#endif // TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_sequence<TestValueType>& obj) = 0;
};

////////////////////////////////////////////////////////////////////////////////
template <typename TestValueType, typename Iterator>
struct test_base_data_visitor_impl : test_base_data_visitor<TestValueType>
{
    test_base_data_visitor_impl(UDTKind kind, Iterator it_from, Iterator it_to)
        : __kind(kind), __it_from(it_from), __it_to(it_to)
    {
    }

    const UDTKind  __kind;
    const Iterator __it_from;
    const Iterator __it_to;
};

////////////////////////////////////////////////////////////////////////////////
template <typename TestValueType, typename Iterator>
struct test_base_data_visitor_retrieve : test_base_data_visitor_impl<TestValueType, Iterator>
{
    using Base = test_base_data_visitor_impl<TestValueType, Iterator>;

    test_base_data_visitor_retrieve(UDTKind kind, Iterator it_from, Iterator it_to)
        : Base(kind, it_from, it_to)
    {
    }

#if TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_usm     <TestValueType>& obj) override;
    virtual bool on_visit(::std::size_t nIndex, test_base_data_buffer  <TestValueType>& obj) override;
#endif // TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_sequence<TestValueType>& obj) override;
};

////////////////////////////////////////////////////////////////////////////////
template <typename TestValueType, typename Iterator>
struct test_base_data_visitor_update : test_base_data_visitor_impl<TestValueType, Iterator>
{
    using Base = test_base_data_visitor_impl<TestValueType, Iterator>;

    test_base_data_visitor_update(UDTKind kind, Iterator it_from, Iterator it_to)
        : Base(kind, it_from, it_to)
    {
    }

#if TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_usm     <TestValueType>& obj) override;
    virtual bool on_visit(::std::size_t nIndex, test_base_data_buffer  <TestValueType>& obj) override;
#endif // TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_sequence<TestValueType>& obj) override;
};

////////////////////////////////////////////////////////////////////////////////
///
template <typename TestValueType>
struct test_base
{
    test_base_data<TestValueType>& base_data_ref;

    test_base(test_base_data<TestValueType>& _base_data_ref)
        : base_data_ref(_base_data_ref)
    {
    }

    template <UDTKind kind, typename Size>
    class TestDataTransfer
    {
    public:

        using HostData = std::vector<TestValueType>;
        using Iterator = typename HostData::iterator;

        TestDataTransfer(test_base& _test_base, Size _count)
            : __test_base(_test_base)
            , __host_buffer(_count)
            , __count(_count)
        {
        }

        TestValueType* get()
        {
            return __host_buffer.data();
        }

        void retrieve_data()
        {
            test_base_data_visitor_retrieve<TestValueType, Iterator> visitor_retrieve(
                kind, __host_buffer.begin(), __host_buffer.end());

            if (!__test_base.base_data_ref.visit(&visitor_retrieve))
                assert(false);
        }

        void update_data()
        {
            test_base_data_visitor_update<TestValueType, Iterator> visitor_update(
                kind, __host_buffer.begin(), __host_buffer.end());

            if (!__test_base.base_data_ref.visit(&visitor_update))
                assert(false);
        }

        void update_data(Size count)
        {
            assert(count <= __count);

            test_base_data_visitor_update<TestValueType, Iterator> visitor_update(
                kind, __host_buffer.begin(), __host_buffer.begin() + count);

            if (!__test_base.base_data_ref.visit(&visitor_update))
                assert(false);
        }

    protected:

        test_base& __test_base;
        HostData   __host_buffer;
        const Size __count = 0;
    };
};

// 1) define class as
//      template <typename TestValueType>
//      struct TestClassName : TestUtils::test_base<TestValueType>
// 2) define class as
//      struct TestClassName
#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST(TestClassName)                                                  \
    template <typename TestValueType>                                               \
    struct TestClassName : TestUtils::test_base<TestValueType>
#else
#define DEFINE_TEST(TestClassName)                                                  \
    struct TestClassName
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST_1(TestClassName, TemplateParams)                                \
    template <typename TestValueType, typename TemplateParams>                      \
    struct TestClassName : TestUtils::test_base<TestValueType>
#else
#define DEFINE_TEST_1(TestClassName, TemplateParams)                                \
    template <typename TemplateParams>                                              \
    struct TestClassName
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST_CONSTRUCTOR(TestClassName)                                                                    \
    TestClassName(test_base_data<TestValueType>& _test_base_data)                                                 \
        : TestUtils::test_base<TestValueType>(_test_base_data)                                                    \
    {                                                                                                             \
    }                                                                                                             \
                                                                                                                  \
    template <UDTKind kind, typename Size>                                                                        \
    using TestDataTransfer = typename TestUtils::test_base<TestValueType>::template TestDataTransfer<kind, Size>; \
                                                                                                                  \
    using UsedValueType = TestValueType;
#else
#define DEFINE_TEST_CONSTRUCTOR(TestClassName)
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename T, typename TestName, typename TestBaseData>
typename ::std::enable_if<::std::is_base_of<test_base<T>, TestName>::value, TestName>::type
create_test_obj(TestBaseData& data)
{
    return TestName(data);
}

template <typename T, typename TestName, typename TestBaseData>
typename ::std::enable_if<!::std::is_base_of<test_base<T>, TestName>::value, TestName>::type
create_test_obj(TestBaseData&)
{
    return TestName();
}

//--------------------------------------------------------------------------------------------------------------------//
// Used with algorithms that have two input sequences and one output sequences
template <typename T, typename TestName>
//typename ::std::enable_if<::std::is_base_of<test_base<T>, TestName>::value, void>::type
void
test_algo_three_sequences()
{
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        using TestBaseData = test_base_data_sequence<T>;

        TestBaseData test_base_data({ { (::std::size_t)max_n, (::std::size_t)inout1_offset },
                                      { (::std::size_t)max_n, (::std::size_t)inout2_offset },
                                      { (::std::size_t)max_n, (::std::size_t)inout3_offset } });

        // create iterators
        auto inout1_offset_first = test_base_data.get_start_from(0);
        auto inout2_offset_first = test_base_data.get_start_from(1);
        auto inout3_offset_first = test_base_data.get_start_from(2);

        invoke_on_all_host_policies()(create_test_obj<T, TestName>(test_base_data),
                                      inout1_offset_first, inout1_offset_first + n,
                                      inout2_offset_first, inout2_offset_first + n,
                                      inout3_offset_first, inout3_offset_first + n,
                                      n);
    }
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestName>
typename ::std::enable_if<
    ::std::is_base_of<test_base<typename TestName::UsedValueType>, TestName>::value,
    void>::type
test_algo_three_sequences()
{
    test_algo_three_sequences<typename TestName::UsedValueType, TestName>();
}

}; // namespace TestUtils

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType>
bool
TestUtils::test_base_data_usm<TestValueType>::visit(test_base_data_visitor<TestValueType>* visitor)
{
    for (::std::size_t nIndex = 0; nIndex < data.size(); ++nIndex)
    {
        if (visitor->on_visit(nIndex, *this))
            return true;
    }

    return false;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
bool
TestUtils::test_base_data_buffer<TestValueType>::visit(test_base_data_visitor<TestValueType>* visitor)
{
    for (::std::size_t nIndex = 0; nIndex < data.size(); ++nIndex)
    {
        if (visitor->on_visit(nIndex, *this))
            return true;
    }

    return false;
}

#endif //  TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
bool
TestUtils::test_base_data_sequence<TestValueType>::visit(test_base_data_visitor<TestValueType>* visitor)
{
    for (::std::size_t nIndex = 0; nIndex < data.size(); ++nIndex)
    {
        if (visitor->on_visit(nIndex, *this))
            return true;
    }

    return false;
}

#if TEST_DPCPP_BACKEND_PRESENT
//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_retrieve<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_buffer<TestValueType>& obj)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    auto& data = obj.data.at(nIndex);
    auto acc = data.src_data_buf.template get_access<sycl::access::mode::read_write>();

    auto __index = data.offset;
    for (auto __it = Base::__it_from; __it != Base::__it_to; ++__it, ++__index)
    {
        *__it = acc[__index];
    }

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType, typename Iterator>
struct RetrieveUsmData
{
    Iterator      it_from;
    Iterator      it_to;
    ::std::size_t offset = 0;

    RetrieveUsmData(Iterator itFrom, Iterator itTo, ::std::size_t __offset)
        : it_from(itFrom)
        , it_to(itTo)
        , offset(__offset)
    {
    }

    template <sycl::usm::alloc alloc_type>
    void operator()(TestUtils::usm_data_transfer<alloc_type, TestValueType>& obj)
    {
        const auto count = it_to - it_from;
        obj.retrieve_data(it_from, offset, count);
    }
};
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_retrieve<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_usm<TestValueType>& obj)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    auto& data_item = obj.data.at(nIndex);

    RetrieveUsmData<TestValueType, Iterator> pred(Base::__it_from, Base::__it_to, data_item.offset);
    data_item.process_usm_data_transfer(pred);

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_retrieve<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_sequence<TestValueType>& /*obj*/)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    // No additional actions required here

    return true;
}

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_update<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_buffer<TestValueType>& obj)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    auto& data = obj.data.at(nIndex);

    auto acc = data.src_data_buf.template get_access<sycl::access::mode::read_write>();

    auto __index = data.offset;
    for (auto __it = Base::__it_from; __it != Base::__it_to; ++__it, ++__index)
    {
        acc[__index] = *__it;
    }

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType, typename Iterator>
struct UpdateUsmData
{
    Iterator      it_from;
    Iterator      it_to;
    ::std::size_t offset = 0;

    UpdateUsmData(Iterator itFrom, Iterator itTo, ::std::size_t __offset)
        : it_from(itFrom)
        , it_to(itTo)
        , offset(__offset)
    {
    }

    template <sycl::usm::alloc alloc_type>
    void operator()(TestUtils::usm_data_transfer<alloc_type, TestValueType>& obj)
    {
        const auto count = it_to - it_from;
        obj.update_data(it_from, offset, count);
    }
};
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_update<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_usm<TestValueType>& obj)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    auto& data_item = obj.data.at(nIndex);

    UpdateUsmData<TestValueType, Iterator> pred(Base::__it_from, Base::__it_to, data_item.offset);
    data_item.process_usm_data_transfer(pred);

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_update<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_sequence<TestValueType>& /*obj*/)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    // No additional actions required here

    return true;
}

//--------------------------------------------------------------------------------------------------------------------//

#endif // UTILS_TEST_BASE
