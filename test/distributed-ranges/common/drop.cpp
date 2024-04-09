// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Drop : public testing::Test {
public:
};

TYPED_TEST_SUITE(Drop, AllTypes);

TYPED_TEST(Drop, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::drop(ops.vec, 2);
  auto dist = xhp::views::drop(ops.dist_vec, 2);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Drop, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(ops, rng::views::drop(ops.vec, 2),
                                xhp::views::drop(ops.dist_vec, 2)));
}

template <class TypeParam>
void localAndDrDropResultsAreSameTest(std::size_t dropSize) {
  Ops1<TypeParam> ops(10);
  auto dist = xhp::views::drop(ops.dist_vec, dropSize);
  auto local = rng::views::drop(ops.vec, dropSize);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Drop, lessThanSize) {
  localAndDrDropResultsAreSameTest<TypeParam>(6);
}

TYPED_TEST(Drop, sameSize) { localAndDrDropResultsAreSameTest<TypeParam>(10); }

TYPED_TEST(Drop, moreSize) { localAndDrDropResultsAreSameTest<TypeParam>(12); }

TYPED_TEST(Drop, zero) { localAndDrDropResultsAreSameTest<TypeParam>(0); }

TYPED_TEST(Drop, one) { localAndDrDropResultsAreSameTest<TypeParam>(1); }

TYPED_TEST(Drop, emptyInput_zeroSize) {
  TypeParam dv(0);
  auto dist = xhp::views::drop(dv, 0);
  EXPECT_TRUE(rng::empty(dist));
}

TYPED_TEST(Drop, emptyInput_nonZeroSize) {
  TypeParam dv(0);
  auto dist = xhp::views::drop(dv, 1);
  EXPECT_TRUE(rng::empty(dist));
}

TYPED_TEST(Drop, large) {
  TypeParam dv(123456, 77);

  auto drop_result = xhp::views::drop(dv, 54321);

  EXPECT_EQ(*(--drop_result.end()), 77);
  fence();
  *(drop_result.begin()) = 5;
  fence();
  EXPECT_EQ(dv[54321], 5);
  EXPECT_EQ(dv[54322], 77);
  EXPECT_EQ(rng::size(drop_result), 123456 - 54321);
}

TYPED_TEST(Drop, largeDropOfAllButOneHasSameSegmentAndRank) {
  TypeParam dv(123456, 77);

  auto drop_view_result = xhp::views::drop(dv, 123456 - 1);

  auto drop_view_segments = dr::ranges::segments(drop_view_result);
  auto dv_segments = dr::ranges::segments(dv);
  auto last_segment_index = dv_segments.size() - 1;

  EXPECT_TRUE(check_segments(drop_view_result));
  EXPECT_EQ(rng::size(drop_view_segments), 1);
  EXPECT_EQ(dr::ranges::rank(drop_view_segments[0]),
            dr::ranges::rank(dv_segments[last_segment_index]));
}

TYPED_TEST(Drop, dropOfAllElementsButOneHasOneSegmentAndSameRank) {
  TypeParam dv(10, 77);
  auto drop_view_result = xhp::views::drop(dv, 9);

  auto drop_view_segments = dr::ranges::segments(drop_view_result);
  auto dv_segments = dr::ranges::segments(dv);
  auto last_segment_index = dv_segments.size() - 1;

  EXPECT_TRUE(check_segments(drop_view_result));
  EXPECT_EQ(rng::size(drop_view_segments), 1);
  EXPECT_EQ(dr::ranges::rank(drop_view_segments[0]),
            dr::ranges::rank(dv_segments[last_segment_index]));
}

TYPED_TEST(Drop, dropOfFirstSegementHasSameSegmentsSize) {
  TypeParam dv(10, 77);

  const auto first_seg_size = dr::ranges::segments(dv)[0].size();
  auto drop_view_result = xhp::views::drop(dv, first_seg_size);
  auto drop_view_segments = dr::ranges::segments(drop_view_result);
  EXPECT_EQ(rng::size(drop_view_segments), dr::ranges::segments(dv).size() - 1);
}

TYPED_TEST(Drop, dropOfOneElementHasAllSegmentsWithSameRanks) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE, 77);

  auto dv_segments = dr::ranges::segments(dv);
  auto drop_view_result = xhp::views::drop(dv, 1);
  auto drop_view_segments = dr::ranges::segments(drop_view_result);

  EXPECT_EQ(rng::size(dv_segments), rng::size(drop_view_segments));
  for (std::size_t i = 0; i < rng::size(dv_segments); ++i)
    EXPECT_EQ(dr::ranges::rank(dv_segments[i]),
              dr::ranges::rank(drop_view_segments[i]));
}
