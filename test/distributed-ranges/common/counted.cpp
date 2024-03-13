// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Counted : public testing::Test {
public:
};

TYPED_TEST_SUITE(Counted, AllTypes);

TYPED_TEST(Counted, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::counted(ops.vec.begin() + 1, 2);
  auto dist = xhp::views::counted(ops.dist_vec.begin() + 1, 2);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Counted, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(
      check_mutate_view(ops, rng::views::counted(ops.vec.begin() + 2, 3),
                        xhp::views::counted(ops.dist_vec.begin() + 2, 3)));
}

template <class TypeParam>
void localAndDrCountedResultsAreSameTest(std::size_t countedSize) {
  Ops1<TypeParam> ops(10);
  auto dist = xhp::views::counted(ops.dist_vec.begin() + 2, countedSize);
  auto local = rng::views::counted(ops.vec.begin() + 2, countedSize);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Counted, lessThanSize) {
  localAndDrCountedResultsAreSameTest<TypeParam>(6);
}

TYPED_TEST(Counted, sameSize) {
  localAndDrCountedResultsAreSameTest<TypeParam>(8);
}

TYPED_TEST(Counted, zero) { localAndDrCountedResultsAreSameTest<TypeParam>(0); }

TYPED_TEST(Counted, one) { localAndDrCountedResultsAreSameTest<TypeParam>(1); }

TYPED_TEST(Counted, emptyInput_zeroSize) {
  TypeParam dv(0);
  auto dist = xhp::views::counted(dv.begin(), 0);
  EXPECT_TRUE(rng::empty(dist));
}

TYPED_TEST(Counted, large) {
  TypeParam dv(123456, 77);

  auto counted_result = xhp::views::counted(dv.begin() + 2, 54321);

  EXPECT_EQ(*(--counted_result.end()), 77);
  fence();
  *(--counted_result.end()) = 5;
  fence();
  EXPECT_EQ(dv[54322], 5);
  EXPECT_EQ(dv[54323], 77);
  EXPECT_EQ(rng::size(counted_result), 54321);
}

TYPED_TEST(Counted, countedOfOneElementHasOneSegmentAndSameRank) {
  TypeParam dv(10, 77);
  auto counted_view_result = xhp::views::counted(dv.end() - 1, 1);

  auto counted_view_segments = dr::ranges::segments(counted_view_result);
  auto dv_segments = dr::ranges::segments(dv);
  auto last_segment_index = dv_segments.size() - 1;

  EXPECT_TRUE(check_segments(counted_view_result));
  EXPECT_EQ(rng::size(counted_view_segments), 1);
  EXPECT_EQ(dr::ranges::rank(counted_view_segments[0]),
            dr::ranges::rank(dv_segments[last_segment_index]));
}

TYPED_TEST(Counted, countedOfFirstSegementHasOneSegmentAndSameRank) {
  TypeParam dv(123456, 77);

  const auto first_seg_size = dr::ranges::segments(dv)[0].size();
  std::size_t bias = 2;
  // test assumes there are not too many ranks
  assert(dr::ranges::segments(dv)[0].size() > bias);
  auto counted_view_result =
      xhp::views::counted(dv.begin() + bias, first_seg_size - bias);
  auto counted_view_segments = dr::ranges::segments(counted_view_result);
  EXPECT_EQ(rng::size(counted_view_segments), 1);
  EXPECT_EQ(dr::ranges::rank(counted_view_segments[0]),
            dr::ranges::rank(dr::ranges::segments(dv)[0]));
}

TYPED_TEST(Counted, countedOfAllButOneSizeHasAllSegmentsWithSameRanks) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE, 77);

  auto dv_segments = dr::ranges::segments(dv);
  std::size_t bias = 1;
  // test assumes there are not too many ranks
  assert(dv_segments[0].size() > bias);
  auto counted_view_result =
      xhp::views::counted(dv.begin() + bias, EVENLY_DIVIDABLE_SIZE - bias);
  auto counted_view_segments = dr::ranges::segments(counted_view_result);

  EXPECT_EQ(rng::size(dv_segments), rng::size(counted_view_segments));
  for (std::size_t i = 0; i < rng::size(dv_segments); ++i)
    EXPECT_EQ(dr::ranges::rank(dv_segments[i]),
              dr::ranges::rank(counted_view_segments[i]));
}
