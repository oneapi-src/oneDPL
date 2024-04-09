// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Take : public testing::Test {
public:
};

TYPED_TEST_SUITE(Take, AllTypes);

TYPED_TEST(Take, isCompliant) {
  TypeParam dv(10);
  static_assert(compliant_view<decltype(xhp::views::take(dv, 6))>);
}

TYPED_TEST(Take, mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(ops, rng::views::take(ops.vec, 6),
                                xhp::views::take(ops.dist_vec, 6)));
}

template <class TypeParam>
void localAndDrTakeResultsAreSameTest(std::size_t takeSize) {
  Ops1<TypeParam> ops(10);
  auto dist = xhp::views::take(ops.dist_vec, takeSize);
  auto local = rng::views::take(ops.vec, takeSize);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, lessThanSize) {
  localAndDrTakeResultsAreSameTest<TypeParam>(6);
}

TYPED_TEST(Take, sameSize) { localAndDrTakeResultsAreSameTest<TypeParam>(10); }

TYPED_TEST(Take, moreSize) { localAndDrTakeResultsAreSameTest<TypeParam>(12); }

TYPED_TEST(Take, zero) { localAndDrTakeResultsAreSameTest<TypeParam>(0); }

TYPED_TEST(Take, one) { localAndDrTakeResultsAreSameTest<TypeParam>(1); }

TYPED_TEST(Take, emptyInput_zeroSize) {
  TypeParam dv(0);
  auto dist = xhp::views::take(dv, 0);
  EXPECT_TRUE(rng::empty(dist));
}

TYPED_TEST(Take, emptyInput_nonZeroSize) {
  TypeParam dv(0);
  auto dist = xhp::views::take(dv, 1);
  EXPECT_TRUE(rng::empty(dist));
}

TYPED_TEST(Take, large) {
  TypeParam dv(123456, 77);

  auto take_result = xhp::views::take(dv, 54321);

  EXPECT_EQ(*(--take_result.end()), 77);
  fence();
  *(--take_result.end()) = 5;
  fence();
  EXPECT_EQ(dv[54320], 5);
  EXPECT_EQ(dv[54321], 77);
  EXPECT_EQ(rng::size(take_result), 54321);
}

TYPED_TEST(Take, takeOfOneElementHasOneSegmentAndSameRank) {
  TypeParam dv(10, 77);
  auto take_view_result = xhp::views::take(dv, 1);

  auto take_view_segments = dr::ranges::segments(take_view_result);
  auto dv_segments = dr::ranges::segments(dv);

  EXPECT_TRUE(check_segments(take_view_result));
  EXPECT_EQ(rng::size(take_view_segments), 1);
  EXPECT_EQ(dr::ranges::rank(take_view_segments[0]),
            dr::ranges::rank(dv_segments[0]));
}

TYPED_TEST(Take, takeOfFirstSegementHasOneSegmentAndSameRank) {
  TypeParam dv(10, 77);

  const auto first_seg_size = dr::ranges::segments(dv)[0].size();
  auto take_view_result = xhp::views::take(dv, first_seg_size);
  auto take_view_segments = dr::ranges::segments(take_view_result);
  EXPECT_EQ(rng::size(take_view_segments), 1);
  EXPECT_EQ(dr::ranges::rank(take_view_segments[0]),
            dr::ranges::rank(dr::ranges::segments(dv)[0]));
}

template <class TypeParam>
void takeHasSameSegments(std::size_t dv_size, std::size_t take_size) {
  TypeParam dv(dv_size, 77);

  auto dv_segments = dr::ranges::segments(dv);
  auto take_view_result = xhp::views::take(dv, take_size);
  auto take_view_segments = dr::ranges::segments(take_view_result);

  EXPECT_EQ(rng::size(dv_segments), rng::size(take_view_segments));
  for (std::size_t i = 0; i < rng::size(dv_segments); ++i)
    EXPECT_EQ(dr::ranges::rank(dv_segments[i]),
              dr::ranges::rank(take_view_segments[i]));
}

TYPED_TEST(Take, takeOfAllButOneSizeHasAllSegmentsWithSameRanks) {
  takeHasSameSegments<TypeParam>(EVENLY_DIVIDABLE_SIZE,
                                 EVENLY_DIVIDABLE_SIZE - 1);
}

TYPED_TEST(Take, takeOfMoreSizeHasSameNumberOfSegmentsAndSameRanks) {
  takeHasSameSegments<TypeParam>(EVENLY_DIVIDABLE_SIZE,
                                 EVENLY_DIVIDABLE_SIZE * 2);
}
