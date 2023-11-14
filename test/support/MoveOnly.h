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

#ifndef _MOVEONLY_H
#define _MOVEONLY_H

#include <utility>

struct MoveOnly
{
    int data_;

    MoveOnly(const MoveOnly&);
    MoveOnly&
    operator=(const MoveOnly&);

    MoveOnly(int data = 1) : data_(data) {}
    MoveOnly(MoveOnly&& x) : data_(x.data_) { x.data_ = 0; }
    MoveOnly&
    operator=(MoveOnly&& x)
    {
        data_ = x.data_;
        x.data_ = 0;
        return *this;
    }

    int
    get() const
    {
        return data_;
    }

    bool
    operator==(const MoveOnly& x) const
    {
        return data_ == x.data_;
    }
    bool
    operator<(const MoveOnly& x) const
    {
        return data_ < x.data_;
    }
    MoveOnly
    operator+(const MoveOnly& x) const
    {
        return MoveOnly{data_ + x.data_};
    }
    MoveOnly
    operator*(const MoveOnly& x) const
    {
        return MoveOnly{data_ * x.data_};
    }

    void
    swap(MoveOnly& other)
    {
        std::swap(data_, other.data_);
    }
};

#endif  // _MOVEONLY_H
