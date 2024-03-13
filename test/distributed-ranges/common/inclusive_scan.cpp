// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class InclusiveScan : public testing::Test {
public:
};

// suite doesn't end with ISHMEM
TYPED_TEST_SUITE(InclusiveScan, AllTypesWithoutIshmem);

TYPED_TEST(InclusiveScan, whole_range) {
  TypeParam dv_in(15);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(15, 0);

  xhp::inclusive_scan(dv_in, dv_out, std::plus<>());
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 + 2, dv_out[1]);
  EXPECT_EQ(1 + 2 + 3, dv_out[2]);
  EXPECT_EQ(1 + 2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6, dv_out[5]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7, dv_out[6]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[7]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[8]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10, dv_out[9]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11, dv_out[10]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12, dv_out[11]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13, dv_out[12]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14,
            dv_out[13]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15,
            dv_out[14]);
}

TYPED_TEST(InclusiveScan, whole_range_small) {
  TypeParam dv_in(3);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(3, 0);

  xhp::inclusive_scan(dv_in, dv_out, std::plus<>());
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 + 2, dv_out[1]);
  EXPECT_EQ(1 + 2 + 3, dv_out[2]);
}

TYPED_TEST(InclusiveScan, whole_range_with_init_value) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::inclusive_scan(dv_in, dv_out, std::plus<>(), 10);
  EXPECT_EQ(10 + 1, dv_out[0]);
  EXPECT_EQ(10 + 1 + 2, dv_out[1]);
  EXPECT_EQ(10 + 1 + 2 + 3, dv_out[2]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6, dv_out[5]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7, dv_out[6]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[7]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[8]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10, dv_out[9]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11, dv_out[10]);
}

TYPED_TEST(InclusiveScan, whole_range_with_init_value_small) {
  TypeParam dv_in(3);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(3, 0);

  xhp::inclusive_scan(dv_in, dv_out, std::plus<>(), 10);
  EXPECT_EQ(10 + 1, dv_out[0]);
  EXPECT_EQ(10 + 1 + 2, dv_out[1]);
  EXPECT_EQ(10 + 1 + 2 + 3, dv_out[2]);
}

TYPED_TEST(InclusiveScan, empty) {
  TypeParam dv_in(3, 1);
  TypeParam dv_out(3, 0);
  xhp::inclusive_scan(rng::begin(dv_in), rng::begin(dv_in), rng::begin(dv_out));
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
}

TYPED_TEST(InclusiveScan, one_element) {
  TypeParam dv_in(3, 1);
  TypeParam dv_out(3, 0);
  xhp::inclusive_scan(rng::begin(dv_in), ++rng::begin(dv_in),
                      rng::begin(dv_out));
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
}

TYPED_TEST(InclusiveScan, multiply) {
  TypeParam dv_in(5);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(5, 0);

  xhp::inclusive_scan(dv_in, dv_out, std::multiplies<>());

  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 * 2, dv_out[1]);
  EXPECT_EQ(1 * 2 * 3, dv_out[2]);
  EXPECT_EQ(1 * 2 * 3 * 4, dv_out[3]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5, dv_out[4]);
}

TYPED_TEST(InclusiveScan, touching_first_segment) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::inclusive_scan(rng::begin(dv_in), ++(++rng::begin(dv_in)),
                      rng::begin(dv_out), std::plus<>());
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 + 2, dv_out[1]);
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

TYPED_TEST(InclusiveScan, touching_last_segment) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::inclusive_scan(--(--rng::end(dv_in)), rng::end(dv_in),
                      --(--rng::end(dv_out)));
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
  EXPECT_EQ(0, dv_out[3]);
  EXPECT_EQ(0, dv_out[4]);
  EXPECT_EQ(0, dv_out[5]);
  EXPECT_EQ(0, dv_out[6]);
  EXPECT_EQ(0, dv_out[7]);
  EXPECT_EQ(0, dv_out[8]);
  EXPECT_EQ(10, dv_out[9]);
  EXPECT_EQ(10 + 11, dv_out[10]);
}

TYPED_TEST(InclusiveScan, without_last_element) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::inclusive_scan(rng::begin(dv_in), --rng::end(dv_in), rng::begin(dv_out));
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 + 2, dv_out[1]);
  EXPECT_EQ(1 + 2 + 3, dv_out[2]);
  EXPECT_EQ(1 + 2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6, dv_out[5]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7, dv_out[6]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[7]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[8]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10, dv_out[9]);
  EXPECT_EQ(0, dv_out[10]);
}

TYPED_TEST(InclusiveScan, without_first_element) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::inclusive_scan(++rng::begin(dv_in), rng::end(dv_in),
                      ++rng::begin(dv_out));
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(2, dv_out[1]);
  EXPECT_EQ(2 + 3, dv_out[2]);
  EXPECT_EQ(2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6, dv_out[5]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7, dv_out[6]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[7]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[8]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10, dv_out[9]);
}

TYPED_TEST(InclusiveScan, without_first_and_last_elements) {
  TypeParam dv_in(11);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(11, 0);

  xhp::inclusive_scan(++rng::begin(dv_in), --rng::end(dv_in),
                      ++rng::begin(dv_out));
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(2, dv_out[1]);
  EXPECT_EQ(2 + 3, dv_out[2]);
  EXPECT_EQ(2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6, dv_out[5]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7, dv_out[6]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8, dv_out[7]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, dv_out[8]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10, dv_out[9]);
  EXPECT_EQ(0, dv_out[10]);
}
