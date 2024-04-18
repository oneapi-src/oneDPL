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

#ifndef _UNINITIALIZED_CUSTOM_DATA_STRUCT_H
#define _UNINITIALIZED_CUSTOM_DATA_STRUCT_H

struct StructTriviallyCopyConstructible
{
    StructTriviallyCopyConstructible() = default;
    StructTriviallyCopyConstructible(StructTriviallyCopyConstructible&&) = delete;
    StructTriviallyCopyConstructible(const StructTriviallyCopyConstructible&) = default;

    StructTriviallyCopyConstructible&
    operator=(const StructTriviallyCopyConstructible&) = delete;
    StructTriviallyCopyConstructible&
    operator=(StructTriviallyCopyConstructible&&) = delete;

    bool
    operator==(const StructTriviallyCopyConstructible& other) const
    {
        return true;
    }
    bool
    operator!=(const StructTriviallyCopyConstructible& other) const
    {
        return false;
    }
};

struct StructTriviallyMoveConstructible
{
    StructTriviallyMoveConstructible() = default;
    StructTriviallyMoveConstructible(StructTriviallyMoveConstructible&&) = default;
    StructTriviallyMoveConstructible(const StructTriviallyMoveConstructible&) = delete;

    StructTriviallyMoveConstructible&
    operator=(const StructTriviallyMoveConstructible&) = delete;
    StructTriviallyMoveConstructible&
    operator=(StructTriviallyMoveConstructible&&) = delete;

    bool
    operator==(const StructTriviallyMoveConstructible& other) const
    {
        return true;
    }
    bool
    operator!=(const StructTriviallyMoveConstructible& other) const
    {
        return false;
    }
};

#endif // _UNINITIALIZED_CUSTOM_DATA_STRUCT_H
