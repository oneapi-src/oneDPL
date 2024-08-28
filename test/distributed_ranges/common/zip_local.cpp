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

#include "xp_tests.hpp"

template <typename... Rs> auto test_zip(Rs &&...rs) {
  return xp::views::zip(std::forward<Rs>(rs)...);
}

// Fixture
class ZipLocal : public ::testing::Test {
protected:
  void SetUp() override {
    int val = 100;
    for (std::size_t i = 0; i < ops.size(); i++) {
      auto &op = ops[i];
      auto &mop = mops[i];
      op.resize(10);
      mop.resize(10);
      stdrng::iota(op, val);
      stdrng::iota(mop, val);
      val += 100;
    }
  }

  std::array<std::vector<int>, 3> ops;
  std::array<std::vector<int>, 3> mops;
};

// Try 1, 2, 3 to check pair/tuple issues
TEST_F(ZipLocal, Op1) { EXPECT_EQ(stdrng::views::zip(ops[0]), test_zip(ops[0])); }

TEST_F(ZipLocal, Op2) {
  EXPECT_EQ(stdrng::views::zip(ops[0], ops[1]), test_zip(ops[0], ops[1]));
}

TEST_F(ZipLocal, Op3) {
  EXPECT_EQ(stdrng::views::zip(ops[0], ops[1], ops[2]),
            test_zip(ops[0], ops[1], ops[2]));
}

TEST_F(ZipLocal, Distance) {
  EXPECT_EQ(stdrng::distance(stdrng::views::zip(ops[0], ops[1], ops[2])),
            stdrng::distance(test_zip(ops[0], ops[1], ops[2])));
}

TEST_F(ZipLocal, Size) {
  auto z = test_zip(ops[0], ops[1]);
  EXPECT_EQ(stdrng::size(ops[0]), stdrng::size(z));
}

TEST_F(ZipLocal, Begin) {
  auto z = test_zip(ops[0], ops[1]);
  auto r = stdrng::views::zip(ops[0], ops[1]);
  EXPECT_EQ(*r.begin(), *z.begin());
}

TEST_F(ZipLocal, Empty) {
  EXPECT_FALSE(test_zip(ops[0]).empty());
  EXPECT_TRUE(test_zip(stdrng::views::take(ops[0], 0)).empty());
}

TEST_F(ZipLocal, End) {
  auto z = test_zip(ops[0], ops[1]);
  auto r = stdrng::views::zip(ops[0], ops[1]);
  EXPECT_EQ(r.end() - r.begin(), z.end() - z.begin());
}

TEST_F(ZipLocal, CpoBeginEnd) {
  auto z = test_zip(ops[0], ops[1]);
  auto r = stdrng::views::zip(ops[0], ops[1]);
  EXPECT_EQ(stdrng::end(r) - stdrng::begin(r), stdrng::end(z) - stdrng::begin(z));
}

TEST_F(ZipLocal, All) {
  auto z = test_zip(stdrng::views::all(ops[0]));
  auto r = stdrng::views::zip(stdrng::views::all(ops[0]));
  EXPECT_EQ(r, z);
}

TEST_F(ZipLocal, Iota) {
  auto z = test_zip(ops[0], xp::views::iota(10));
  auto r = stdrng::views::zip(ops[0], stdrng::views::iota(10));
  EXPECT_EQ(r, z);
}

TEST_F(ZipLocal, IterPlusPlus) {
  auto z = test_zip(ops[0], ops[1]);
  auto r = stdrng::views::zip(ops[0], ops[1]);
  auto z_iter = z.begin();
  z_iter++;
  auto r_iter = r.begin();
  r_iter++;
  EXPECT_EQ(*r_iter, *z_iter);
}

TEST_F(ZipLocal, IterEquals) {
  auto z = test_zip(ops[0], ops[1]);
  EXPECT_TRUE(z.begin() == z.begin());
  EXPECT_FALSE(z.begin() == z.begin() + 1);
}

TEST_F(ZipLocal, For) {
  auto z = test_zip(ops[0], ops[1]);

  auto i = 0;
  for (auto it = z.begin(); it != z.end(); it++) {
    // values are same
    EXPECT_EQ(this->ops[0][i], std::get<0>(*it));
    EXPECT_EQ(this->ops[1][i], std::get<1>(*it));

    // addresses are same
    EXPECT_EQ(&(this->ops[0][i]), &std::get<0>(*it));
    EXPECT_EQ(&(this->ops[1][i]), &std::get<1>(*it));
    i++;
  };
  EXPECT_EQ(ops[0].size(), i);
}

TEST_F(ZipLocal, RangeFor) {
  auto z = test_zip(ops[0], ops[1]);

  auto i = 0;
  for (auto v : z) {
    // values are same
    EXPECT_EQ(this->ops[0][i], std::get<0>(v));
    EXPECT_EQ(this->ops[1][i], std::get<1>(v));

    // addresses are same
    EXPECT_EQ(&(this->ops[0][i]), &std::get<0>(v));
    EXPECT_EQ(&(this->ops[1][i]), &std::get<1>(v));
    i++;
  };
  EXPECT_EQ(ops[0].size(), i);
}

TEST_F(ZipLocal, ForEach) {
  auto z = test_zip(ops[0], ops[1]);

  auto i = 0;
  auto check = [this, &i](auto v) {
    // values are same
    EXPECT_EQ(this->ops[0][i], std::get<0>(v));
    EXPECT_EQ(this->ops[1][i], std::get<1>(v));

    // addresses are same
    EXPECT_EQ(&(this->ops[0][i]), &std::get<0>(v));
    EXPECT_EQ(&(this->ops[1][i]), &std::get<1>(v));
    i++;
  };
  stdrng::for_each(z, check);
  EXPECT_EQ(ops[0].size(), i);
}

TEST_F(ZipLocal, Mutate) {
  auto r = stdrng::views::zip(ops[0], ops[1]);
  auto x = test_zip(mops[0], mops[1]);

  std::get<0>(r[0]) = 99;
  std::get<0>(x[0]) = 99;
  EXPECT_EQ(ops[0], mops[0]);
  EXPECT_EQ(ops[1], mops[1]);
  static_assert(stdrng::random_access_range<decltype(x)>);
}

TEST_F(ZipLocal, MutateForEach) {
  auto r = stdrng::views::zip(ops[0], ops[1]);
  auto x = test_zip(mops[0], mops[1]);
  auto n2 = [](auto &&v) {
    std::get<0>(v) += 1;
    std::get<1>(v) = -std::get<1>(v);
  };

  stdrng::for_each(r, n2);
  stdrng::for_each(x, n2);
  EXPECT_EQ(ops[0], mops[0]);
  EXPECT_EQ(ops[1], mops[1]);

  static_assert(stdrng::random_access_range<decltype(x)>);
}
