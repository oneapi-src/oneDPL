


public:
    // Engine types
    //

private:
    // methods to manipulate counters
    using counter_type = ::std::array<result_type, n>;

    ::std::uint64_t get_counter_internal() const { // need to check
        ::std::uint64_t ret = 0;
        for (size_t i = 0; i < period_counter_count; ++i) {
            ret |= ::std::uint64_t(state_[i]) << (word_size * i);
        }
        return ret;
    }
    void set_counter_internal(state& s, counter_type newctr) { // need to check
        static_assert(word_size * period_counter_count <= ::std::numeric_limits<counter_type>::digits);
        for (size_t i = 0; i < period_counter_count; ++i)
            s[i] = (newctr >> (word_size * i)) & in_mask;
    }
    void increase_counter_internal() { // need to check
        state_[0] = (state_[0] + 1) & in_mask;
        for (size_t i = 1; i < period_counter_count; ++i) {
            if (state_[i - 1]) {
                [[likely]] return;
            }
            state_[i] = (state_[i] + 1) & in_mask;
        }
    }

    void seed_internal(::std::initializer_list<result_type> seed) {
        auto start = seed.begin();
        auto end = seed.end();
        size_t i = 0;
        for (i = 0; i < word_count; i++) {
            state_[i] = 0; // all counters are set to zero
            // WARNING: do we need to do it if counters were set before?
        }
        for (; i < state_size; i++) { // keys are set as seed
            state_[i] = (start == end) ? 0 : (*start++) & in_mask;
        }
        ridxref() = 0;
    }



    void generate() {
        if constexpr (n == 2) {
            result_type R0 = (*state_++) & in_mask;
            result_type L0 = (*state_++) & in_mask;
            result_type K0 = (*state_++) & in_mask;
            for (size_t i = 0; i < round_count; ++i) {
                auto [hi, lo] = detail::mulhilo<word_size>(R0, multipliers[0]);
                R0 = hi ^ K0 ^ L0;
                L0 = lo;
                K0 = (K0 + round_consts[0]) & in_mask;
            }
            *output++ = R0;
            *output++ = L0;
        }
        else if constexpr (n == 4) {
                result_type R0 = (*state_++) & in_mask;
                result_type L0 = (*state_++) & in_mask;
                result_type R1 = (*state_++) & in_mask;
                result_type L1 = (*state_++) & in_mask;
                result_type K0 = (*state_++) & in_mask;
                result_type K1 = (*state_++) & in_mask;
                for (size_t i = 0; i < round_count; ++i) {
                    auto [hi0, lo0] = detail::mulhilo<word_size>(R0, multipliers[0]);
                    auto [hi1, lo1] = detail::mulhilo<word_size>(R1, multipliers[1]);
                    R0 = hi1 ^ L0 ^ K0;
                    L0 = lo1;
                    R1 = hi0 ^ L1 ^ K1;
                    L1 = lo0;
                    K0 = (K0 + round_consts[0]) & in_mask;
                    K1 = (K1 + round_consts[1]) & in_mask;
                }
                *output++ = R0;
                *output++ = L0;
                *output++ = R1;
                *output++ = L1;        
        }
        else if constexpr (n == 8) {
            ;// permute 3
        }

        // mul
        //auto [hi, lo] = detail::mulhilo<word_size>(R0, multipliers[0]);
    }

{
    // template <::std::ranges::input_range InRange, ::std::weakly_incrementable Output>
    // requires ::std::ranges::sized_range<InRange> &&
    //     ::std::integral<std::iter_value_t<::std::ranges::range_value_t<InRange>>> &&
    //     ::std::integral<std::iter_value_t<Output>> &&
    //     ::std::indirectly_writable<Output, ::std::iter_value_t<Output>>
    //         Output generate(InRange&& inrange, Output output) {
    //     for (auto initer : inrange) {
    //         if constexpr (n == 2) {
                                    //state_elm
    //             result_type R0 = (*initer++) & in_mask;
    //             result_type L0 = (*initer++) & in_mask;
    //             result_type K0 = (*initer++) & in_mask;
    //             for (size_t i = 0; i < round_count; ++i) {
    //                 auto [hi, lo] = detail::mulhilo<word_size>(R0, multipliers[0]);
    //                 R0 = hi ^ K0 ^ L0;
    //                 L0 = lo;
    //                 K0 = (K0 + round_consts[0]) & in_mask;
    //             }
    //             *output++ = R0;
    //             *output++ = L0;
    //         }
    //         else if constexpr (n == 4) {
    //             result_type R0 = (*initer++) & in_mask;
    //             result_type L0 = (*initer++) & in_mask;
    //             result_type R1 = (*initer++) & in_mask;
    //             result_type L1 = (*initer++) & in_mask;
    //             result_type K0 = (*initer++) & in_mask;
    //             result_type K1 = (*initer++) & in_mask;
    //             for (size_t i = 0; i < round_count; ++i) {
    //                 auto [hi0, lo0] = detail::mulhilo<word_size>(R0, multipliers[0]);
    //                 auto [hi1, lo1] = detail::mulhilo<word_size>(R1, multipliers[1]);
    //                 R0 = hi1 ^ L0 ^ K0;
    //                 L0 = lo1;
    //                 R1 = hi0 ^ L1 ^ K1;
    //                 L1 = lo0;
    //                 K0 = (K0 + round_consts[0]) & in_mask;
    //                 K1 = (K1 + round_consts[1]) & in_mask;
    //             }
    //             *output++ = R0;
    //             *output++ = L0;
    //             *output++ = R1;
    //             *output++ = L1;
    //         }
    //         // No more cases.  See the static_assert(n==2 || n==4) at the top of the class
    //     }
    //     return output;
    // }
}

    // template <::std::output_iterator<const result_type&> Output,
    //           ::std::sized_sentinel_for<Output> Sentinel>
    result_type* operator()(result_type* out) {
        // auto len = sizeof(result_type);

        // //std::cout << "len = " << len << std::endl;

        // // Deliver any saved results
        // auto ri = ridxref();
        // if (ri && len) {
        //     while (ri < word_count && len) {
        //         *out++ = results_[ri++];
        //         --len;
        //     }
        //     if (ri == word_count)
        //         ri = 0;
        // }

        // // Call the bulk generator
        // auto nprf = len / word_count;

        // //std::cout << "nprf = " << nprf << std::endl;
        // // lazily construct the input range.  No need
        // // to allocate and fill a big chunk of memory
        // auto c0 = get_counter_internal();
        // state state_tmp;
        // // out = generate(
        // //     ::std::ranges::views::iota(c0, c0 + nprf) | ::std::ranges::views::transform([&](auto ctr) {
        // //         state_tmp = state_;
        // //         set_counter_internal(state_tmp, ctr);
        // //         return ::std::ranges::begin(state_tmp);
        // //     }),
        // //     out);
        // len -= nprf * word_count;
        // set_counter_internal(state_, c0 + nprf);

        // // Restock the results array
        // if (ri == 0 && len) {
        //     // (*this)(std::begin(state_), std::begin(results_));
        //    // generate(::std::ranges::single_view(state_.data()), results_.data());
        //     increase_counter_internal();
        // }

        // // Finish off any stragglers.
        // while (len--)
        //     *out++ = results_[ri++];
        // ridxref() = ri;
        return out;
    }