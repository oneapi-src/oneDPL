// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class ExclusiveScan : public testing::Test {
public:
};

// segfaults with ISHMEM
TYPED_TEST_SUITE(ExclusiveScan, AllTypesWithoutIshmem);

TYPED_TEST(ExclusiveScan, whole_range) {
  TypeParam dv_in(15);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(15, 0);

  xhp::exclusive_scan(dv_in, dv_out, 10, std::plus<>());
  EXPECT_EQ(10, dv_out[0]);
  EXPECT_EQ(10 + 1, dv_out[1]);
  EXPECT_EQ(10 + 1 + 2, dv_out[2]);
  EXPECT_EQ(10 + 1 + 2 + 3, dv_out[3]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4, dv_out[4]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5, dv_out[5]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6, dv_out[6]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7, dv_out[7]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[8]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[9]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10, dv_out[10]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11, dv_out[11]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12, dv_out[12]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13,
            dv_out[13]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14,
            dv_out[14]);
}

TYPED_TEST(ExclusiveScan, whole_range_small) {
  TypeParam dv_in(3);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(3, 0);

  xhp::exclusive_scan(dv_in, dv_out, 10, std::plus<>());
  EXPECT_EQ(10, dv_out[0]);
  EXPECT_EQ(10 + 1, dv_out[1]);
  EXPECT_EQ(10 + 1 + 2, dv_out[2]);
}

TYPED_TEST(ExclusiveScan, empty) {
  TypeParam dv_in(11, 1);
  TypeParam dv_out(11, 0);
  xhp::exclusive_scan(rng::begin(dv_in), rng::begin(dv_in), rng::begin(dv_out),
                      0);
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
  EXPECT_EQ(0, dv_out[3]);
  EXPECT_EQ(0, dv_out[4]);
  EXPECT_EQ(0, dv_out[5]);
  EXPECT_EQ(0, dv_out[6]);
  EXPECT_EQ(0, dv_out[7]);
  EXPECT_EQ(0, dv_out[8]);
  EXPECT_EQ(0, dv_out[9]);
  EXPECT_EQ(0, dv_out[10]);
}

TYPED_TEST(ExclusiveScan, one_element) {
  TypeParam dv_in(11, 1);
  TypeParam dv_out(11, 0);
  xhp::exclusive_scan(rng::begin(dv_in), ++rng::begin(dv_in),
                      rng::begin(dv_out), 0);
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
  EXPECT_EQ(0, dv_out[3]);
  EXPECT_EQ(0, dv_out[4]);
  EXPECT_EQ(0, dv_out[5]);
  EXPECT_EQ(0, dv_out[6]);
  EXPECT_EQ(0, dv_out[7]);
  EXPECT_EQ(0, dv_out[8]);
  EXPECT_EQ(0, dv_out[9]);
  EXPECT_EQ(0, dv_out[10]);
}

TYPED_TEST(ExclusiveScan, multiply) {
  TypeParam dv_in(13);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(13, 0);

  xhp::exclusive_scan(dv_in, dv_out, 1, std::multiplies<>());

  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1, dv_out[1]);
  EXPECT_EQ(1 * 2, dv_out[2]);
  EXPECT_EQ(1 * 2 * 3, dv_out[3]);
  EXPECT_EQ(1 * 2 * 3 * 4, dv_out[4]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5, dv_out[5]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5 * 6, dv_out[6]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5 * 6 * 7, dv_out[7]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5 * 6 * 7 * 8, dv_out[8]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9, dv_out[9]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10, dv_out[10]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11, dv_out[11]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12, dv_out[12]);
}

TYPED_TEST(ExclusiveScan, multiply_small) {
  TypeParam dv_in(3);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(3, 0);

  xhp::exclusive_scan(dv_in, dv_out, 1, std::multiplies<>());

  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1, dv_out[1]);
  EXPECT_EQ(1 * 2, dv_out[2]);
}

TYPED_TEST(ExclusiveScan, touching_first_segment) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::exclusive_scan(rng::begin(dv_in), ++(++rng::begin(dv_in)),
                      rng::begin(dv_out), 0);
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(1, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
  EXPECT_EQ(0, dv_out[3]);
  EXPECT_EQ(0, dv_out[4]);
  EXPECT_EQ(0, dv_out[5]);
  EXPECT_EQ(0, dv_out[6]);
  EXPECT_EQ(0, dv_out[7]);
  EXPECT_EQ(0, dv_out[8]);
  EXPECT_EQ(0, dv_out[9]);
  EXPECT_EQ(0, dv_out[10]);
}

TYPED_TEST(ExclusiveScan, touching_last_segment) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::exclusive_scan(--(--rng::end(dv_in)), rng::end(dv_in),
                      --(--rng::end(dv_out)), 0);
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
  EXPECT_EQ(0, dv_out[3]);
  EXPECT_EQ(0, dv_out[4]);
  EXPECT_EQ(0, dv_out[5]);
  EXPECT_EQ(0, dv_out[6]);
  EXPECT_EQ(0, dv_out[7]);
  EXPECT_EQ(0, dv_out[8]);
  EXPECT_EQ(0, dv_out[9]);
  EXPECT_EQ(10, dv_out[10]);
}

TYPED_TEST(ExclusiveScan, without_last_element) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::exclusive_scan(rng::begin(dv_in), --rng::end(dv_in), rng::begin(dv_out),
                      0);
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(1, dv_out[1]);
  EXPECT_EQ(1 + 2, dv_out[2]);
  EXPECT_EQ(1 + 2 + 3, dv_out[3]);
  EXPECT_EQ(1 + 2 + 3 + 4, dv_out[4]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5, dv_out[5]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6, dv_out[6]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7, dv_out[7]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[8]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[9]);
  EXPECT_EQ(0, dv_out[10]);
}

TYPED_TEST(ExclusiveScan, without_first_element) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::exclusive_scan(++rng::begin(dv_in), rng::end(dv_in),
                      ++rng::begin(dv_out), 0);
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(2, dv_out[2]);
  EXPECT_EQ(2 + 3, dv_out[3]);
  EXPECT_EQ(2 + 3 + 4, dv_out[4]);
  EXPECT_EQ(2 + 3 + 4 + 5, dv_out[5]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6, dv_out[6]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7, dv_out[7]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[8]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[9]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10, dv_out[10]);
}

TYPED_TEST(ExclusiveScan, without_first_and_last_elements) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::exclusive_scan(++rng::begin(dv_in), --rng::end(dv_in),
                      ++rng::begin(dv_out), 0);
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(2, dv_out[2]);
  EXPECT_EQ(2 + 3, dv_out[3]);
  EXPECT_EQ(2 + 3 + 4, dv_out[4]);
  EXPECT_EQ(2 + 3 + 4 + 5, dv_out[5]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6, dv_out[6]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7, dv_out[7]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[8]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[9]);
  EXPECT_EQ(0, dv_out[10]);
}
