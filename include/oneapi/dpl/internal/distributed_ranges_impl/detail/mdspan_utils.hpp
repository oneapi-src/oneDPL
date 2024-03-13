// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>

namespace dr::__detail {

template <std::size_t Rank> auto dims(md::dextents<std::size_t, Rank> extents) {
  if constexpr (Rank == 1) {
    return std::tuple(extents.extent(0));
  } else if constexpr (Rank == 2) {
    return std::tuple(extents.extent(0), extents.extent(1));
  } else if constexpr (Rank == 3) {
    return std::tuple(extents.extent(0), extents.extent(1), extents.extent(2));
  } else {
    assert(false);
  }
}

template <typename Index> auto shape_to_strides(const Index &shape) {
  const std::size_t rank = rng::size(shape);
  Index strides;
  strides[rank - 1] = 1;
  for (std::size_t i = 1; i < rank; i++) {
    strides[rank - i - 1] = strides[rank - i] * shape[rank - i];
  }
  return strides;
}

template <typename Index>
auto linear_to_index(std::size_t linear, const Index &shape) {
  Index index, strides(shape_to_strides(shape));

  for (std::size_t i = 0; i < rng::size(shape); i++) {
    index[i] = linear / strides[i];
    linear = linear % strides[i];
  }

  return index;
}

template <typename Mdspan>
concept mdspan_like = requires(Mdspan &mdspan) {
  mdspan.rank();
  mdspan.extents();
};

template <typename Mdarray>
concept mdarray_like = requires(Mdarray &mdarray) { mdarray.to_mdspan(); };

template <std::size_t Rank> using dr_extents = std::array<std::size_t, Rank>;
template <std::size_t Rank> using md_extents = md::dextents<std::size_t, Rank>;

//
// Mdspan accessor using an iterator
//
template <std::random_access_iterator Iter> class mdspan_iter_accessor {
public:
  using data_handle_type = Iter;
  using reference = std::iter_reference_t<Iter>;
  using offset_policy = mdspan_iter_accessor;

  constexpr mdspan_iter_accessor() noexcept = default;
  constexpr auto access(Iter iter, std::size_t index) const {
    return iter[index];
  }

  constexpr auto offset(Iter iter, std::size_t index) const noexcept {
    return iter + index;
  }
};

template <typename M, std::size_t Rank, std::size_t... indexes>
auto make_submdspan_impl(M mdspan, const dr_extents<Rank> &starts,
                         const dr_extents<Rank> &ends,
                         std::index_sequence<indexes...>) {
  return md::submdspan(mdspan, std::tuple(starts[indexes], ends[indexes])...);
}

// Mdspan accepts slices, but that is hard to work with because it
// requires parameter packs. Work with starts/size vectors internally
// and use slices at the interface
template <std::size_t Rank>
auto make_submdspan(auto mdspan, const std::array<std::size_t, Rank> &starts,
                    const std::array<std::size_t, Rank> &ends) {
  return make_submdspan_impl(mdspan, starts, ends,
                             std::make_index_sequence<Rank>{});
}

template <std::size_t Rank, typename Op>
void mdspan_foreach(md_extents<Rank> extents, Op op,
                    dr_extents<Rank> index = dr_extents<Rank>(),
                    std::size_t rank = 0) {
  for (index[rank] = 0; index[rank] < extents.extent(rank); index[rank]++) {
    if (rank == Rank - 1) {
      op(index);
    } else {
      mdspan_foreach(extents, op, index, rank + 1);
    }
  }
}

// Pack mdspan into contiguous container
template <mdspan_like Src>
auto mdspan_copy(Src src, std::forward_iterator auto dst) {
  __detail::event event;

  constexpr std::size_t rank = std::remove_cvref_t<Src>::rank();
  if (rank >= 2 && rank <= 3 && mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    constexpr std::size_t rank = std::remove_cvref_t<Src>::rank();
    if constexpr (rank == 2) {
      event = dr::__detail::parallel_for(
          dr::mhp::sycl_queue(), sycl::range(src.extent(0), src.extent(1)),
          [src, dst](auto idx) {
            dst[idx[0] * src.extent(1) + idx[1]] = src(idx);
          });
    } else if constexpr (rank == 3) {
      event = dr::__detail::parallel_for(
          dr::mhp::sycl_queue(),
          sycl::range(src.extent(0), src.extent(1), src.extent(2)),
          [src, dst](auto idx) {
            dst[idx[0] * src.extent(1) * src.extent(2) +
                idx[1] * src.extent(2) + idx[2]] = src(idx);
          });
    } else {
      assert(false);
    }
#endif
  } else {
    auto pack = [src, &dst](auto index) { *dst++ = src(index); };
    mdspan_foreach<src.rank(), decltype(pack)>(src.extents(), pack);
  }

  return event;
}

// unpack contiguous container into mdspan
template <mdspan_like Dst>
auto mdspan_copy(std::forward_iterator auto src, Dst dst) {
  __detail::event event;

  constexpr std::size_t rank = std::remove_cvref_t<Dst>::rank();
  if (rank >= 2 && rank <= 3 && mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    if constexpr (rank == 2) {
      event = dr::__detail::parallel_for(
          dr::mhp::sycl_queue(), sycl::range(dst.extent(0), dst.extent(1)),
          [src, dst](auto idx) {
            dst(idx) = src[idx[0] * dst.extent(1) + idx[1]];
          });
    } else if constexpr (rank == 3) {
      event = dr::__detail::parallel_for(
          dr::mhp::sycl_queue(),
          sycl::range(dst.extent(0), dst.extent(1), dst.extent(2)),
          [src, dst](auto idx) {
            dst(idx) = src[idx[0] * dst.extent(1) * dst.extent(2) +
                           idx[1] * dst.extent(2) + idx[2]];
          });
    } else {
      assert(false);
    }
#endif
  } else {
    auto unpack = [&src, dst](auto index) { dst(index) = *src++; };
    mdspan_foreach<dst.rank(), decltype(unpack)>(dst.extents(), unpack);
  }

  return event;
}

// copy mdspan to mdspan
auto mdspan_copy(mdspan_like auto src, mdspan_like auto dst) {
  __detail::event event;

  assert(src.extents() == dst.extents());

  constexpr std::size_t rank = std::remove_cvref_t<decltype(src)>::rank();
  if (rank >= 2 && rank <= 3 && mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    dr::drlog.debug("mdspan_copy using sycl\n");
    if constexpr (rank == 2) {
      event = dr::__detail::parallel_for(
          dr::mhp::sycl_queue(), sycl::range(dst.extent(0), dst.extent(1)),
          [src, dst](auto idx) { dst(idx) = src(idx); });
    } else if constexpr (rank == 3) {
      event = dr::__detail::parallel_for(
          dr::mhp::sycl_queue(),
          sycl::range(dst.extent(0), dst.extent(1), dst.extent(2)),
          [src, dst](auto idx) { dst(idx) = src(idx); });
    } else {
      assert(false);
    }
#endif
  } else {

    auto copy = [src, dst](auto index) { dst(index) = src(index); };
    mdspan_foreach<src.rank(), decltype(copy)>(src.extents(), copy);
  }

  return event;
}

// For operator(), rearrange indices according to template arguments.
//
// For mdtranspose<mdspan3d, 2, 0, 1> a(b);
//
// a(1, 2, 3) references b(3, 1, 2)
//
template <typename Mdspan, std::size_t... Is>
class mdtranspose : public Mdspan {
private:
  static constexpr std::size_t rank_ = Mdspan::rank();

public:
  // Inherit constructors from base class
  mdtranspose(Mdspan &mdspan) : Mdspan(mdspan) {}

  // rearrange indices according to template arguments
  template <std::integral... Indexes>
  auto &operator()(Indexes... indexes) const {
    std::tuple index(indexes...);
    return Mdspan::operator()(std::get<Is>(index)...);
  }
  auto &operator()(std::array<std::size_t, rank_> index) const {
    return Mdspan::operator()(index[Is]...);
  }

  auto extents() const {
    // To get the extents, we must invert the index mapping
    std::array<std::size_t, rank_> from_transposed({Is...});
    std::array<std::size_t, rank_> extents_t;
    for (std::size_t i = 0; i < rank_; i++) {
      extents_t[from_transposed[i]] = Mdspan::extent(i);
    }

    return md_extents<rank_>(extents_t);
  }
  auto extent(std::size_t d) const { return extents().extent(d); }
};

} // namespace dr::__detail

template <dr::__detail::mdspan_like Mdspan>
struct fmt::formatter<Mdspan, char> : public formatter<string_view> {
  template <typename FmtContext>
  auto format(Mdspan mdspan, FmtContext &ctx) const {
    std::array<std::size_t, mdspan.rank()> index;
    rng::fill(index, 0);
    format_mdspan(ctx, mdspan, index, 0);
    return ctx.out();
  }

  void format_mdspan(auto &ctx, auto mdspan, auto &index,
                     std::size_t dim) const {
    for (std::size_t i = 0; i < mdspan.extent(dim); i++) {
      index[dim] = i;
      if (dim == mdspan.rank() - 1) {
        if (i == 0) {
          format_to(ctx.out(), "{}: ", index);
        }
        format_to(ctx.out(), "{:4} ", mdspan(index));
      } else {
        format_mdspan(ctx, mdspan, index, dim + 1);
      }
    }
    format_to(ctx.out(), "\n");
  }
};

namespace MDSPAN_NAMESPACE {

template <dr::__detail::mdspan_like M1, dr::__detail::mdspan_like M2>
bool operator==(const M1 &m1, const M2 &m2) {
  constexpr std::size_t rank1 = M1::rank(), rank2 = M2::rank();
  static_assert(rank1 == rank2);
  if (dr::__detail::dims<rank1>(m1.extents()) !=
      dr::__detail::dims<rank1>(m2.extents())) {
    return false;
  }

  // See mdspan_foreach for a way to generalize this to all ranks
  if constexpr (M1::rank() == 1) {
    for (std::size_t i = 0; i < m1.extent(0); i++) {
      if (m1(i) != m2(i)) {
        return false;
      }
    }
  } else if constexpr (M1::rank() == 2) {
    for (std::size_t i = 0; i < m1.extent(0); i++) {
      for (std::size_t j = 0; j < m1.extent(1); j++) {
        if (m1(i, j) != m2(i, j)) {
          return false;
        }
      }
    }
  } else if constexpr (M1::rank() == 3) {
    for (std::size_t i = 0; i < m1.extent(0); i++) {
      for (std::size_t j = 0; j < m1.extent(1); j++) {
        for (std::size_t k = 0; k < m1.extent(2); k++) {
          if (m1(i, j, k) != m2(i, j, k)) {
            return false;
          }
        }
      }
    }
  } else {
    assert(false);
  }

  return true;
}

template <dr::__detail::mdspan_like M>
inline std::ostream &operator<<(std::ostream &os, const M &m) {
  if constexpr (dr::__detail::mdarray_like<M>) {
    os << fmt::format("\n{}", m.to_mdspan());
  } else {
    os << fmt::format("\n{}", m);
  }
  return os;
}

} // namespace MDSPAN_NAMESPACE

namespace dr {

template <typename R>
concept distributed_mdspan_range =
    distributed_range<R> && requires(R &r) { r.mdspan(); };

} // namespace dr
