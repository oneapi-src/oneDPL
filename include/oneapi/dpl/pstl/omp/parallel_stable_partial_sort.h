namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <typename _RandomAccessIterator, typename _Compare>
struct _MinKOp
{
    std::vector<_RandomAccessIterator>& __items;
    _Compare __comp;

    _MinKOp(std::vector<_RandomAccessIterator>& __items_, _Compare __comp_) : __items(__items_), __comp(__comp_) {}

    auto
    __it_comp()
    {
        return [this](const auto& l, const auto& r) { return __comp(*l, *r); };
    }

    void
    __keep_smallest_k_items(_RandomAccessIterator __item)
    {
        // Put the new item on the heap and re-establish the heap invariant.
        __items.push_back(__item);
        std::push_heap(__items.begin(), __items.end(), __it_comp());

        // Pop the largest item off the heap.
        std::pop_heap(__items.begin(), __items.end(), __it_comp());
        __items.pop_back();
    };

    void
    __merge(std::vector<_RandomAccessIterator>& __other, std::size_t __k)
    {
        if (__items.capacity() < __k)
        {
            __items.reserve(__k);
        }

        for (auto __it = std::begin(__other); __it != std::end(__other); ++__it)
        {
            if (__items.size() < __k)
            {
                // Continue growing the items list until we have at least k items by
                // putting the new item on the heap and re-establishing the heap
                // invariant.
                __items.push_back(*__it);
                std::push_heap(__items.begin(), __items.end(), __it_comp());
            }
            else
            {
                __keep_smallest_k_items(*__it);
            }
        }
    }

    void
    __initialize(_RandomAccessIterator __first, _RandomAccessIterator __last, std::size_t __k)
    {
        // If the 'k' value is larger than the chunk size, don't allocate more space. This will
        // cause errors in the reducer as it tries to de-reference iterators that were never
        // assigned.
        __items.resize(std::min(__k, static_cast<std::size_t>(std::distance(__first, __last))));

        auto __item_it = __first;
        auto __tracking_it = std::begin(__items);
        while (__item_it != __last && __tracking_it != std::end(__items))
        {
            *__tracking_it = __item_it;
            ++__item_it;
            ++__tracking_it;
        }
        std::make_heap(__items.begin(), __items.end(), __it_comp());
        for (; __item_it != __last; ++__item_it)
        {
            __keep_smallest_k_items(__item_it);
        }
    }

    static auto
    __reduce(std::vector<_RandomAccessIterator>& __v1, std::vector<_RandomAccessIterator>& __v2, std::size_t __k,
             _Compare __comp) -> std::vector<_RandomAccessIterator>
    {
        if (__v1.empty())
        {
            return __v2;
        }

        if (__v2.empty())
        {
            return __v1;
        }

        if (__v1.size() >= __v2.size())
        {
            _MinKOp<_RandomAccessIterator, _Compare> __op(__v1, __comp);
            __op.__merge(__v2, __k);
            return __v1;
        }

        _MinKOp<_RandomAccessIterator, _Compare> __op(__v2, __comp);
        __op.__merge(__v1, __k);
        return __v2;
    }
};

template <typename _RandomAccessIterator, typename _Compare>
auto
__find_min_k(_RandomAccessIterator __first, _RandomAccessIterator __last, std::size_t __k, _Compare __comp)
    -> std::vector<_RandomAccessIterator>
{
    std::vector<_RandomAccessIterator> __items;
    _MinKOp<_RandomAccessIterator, _Compare> op(__items, __comp);

    op.__initialize(__first, __last, __k);
    return __items;
}

template <typename _RandomAccessIterator, typename _Compare>
auto
__parallel_find_pivot(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, std::size_t __nsort)
    -> _RandomAccessIterator
{
    using _Value = std::vector<_RandomAccessIterator>;
    using _Op = _MinKOp<_RandomAccessIterator, _Compare>;

    std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
    __chunk_partitioner(__first, __last, __n_chunks, __chunk_size, __first_chunk_size,
                        std::max(__nsort, __default_chunk_size));
    /*
     * This function creates a vector of iterators to the container being operated
     * on. It splits that container into fixed size chunks, just like other
     * functions in this backend. For each chunk it finds the smallest k items.
     * The chunks are run through a reducer which keeps the smallest k items from
     * each chunk. Finally, the largest item from the merged chunks is returned as
     * the pivot.
     *
     * The reducer will always produce a chunk merge so that the longest k items
     * list propagates out. So even if some of the chunks are less than __nsort
     * elements, at least one chunk will be and this chunk will  end up producing
     * a correctly sized smallest k items list.
     *
     * Note that the case where __nsort == distance(__first, __last) is handled by
     * performing a complete sort of the container, so we don't have to handle
     * that here.
     */

    auto __reduce_chunk = [&](std::uint32_t __chunk)
    {
        auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
        auto __index = __chunk == 0 ? 0 : (__chunk * __chunk_size) + (__first_chunk_size - __chunk_size);
        auto __begin = std::next(__first, __index);
        auto __end = std::next(__begin, __this_chunk_size);

        return __find_min_k(__begin, __end, __nsort, __comp);
    };

    auto __reduce_value = [&](auto& __v1, auto& __v2) { return _Op::__reduce(__v1, __v2, __nsort, __comp); };
    auto __result = __parallel_reduce_chunks(0, __n_chunks, __reduce_chunk, __reduce_value, _Value());

    // Return largest item
    return __result.front();
}

template <typename _RandomAccessIterator, typename _Compare>
void
__parallel_partition(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _RandomAccessIterator __pivot,
                     _Compare __comp, std::size_t __nsort)
{
    auto __size = static_cast<std::size_t>(std::distance(__xs, __xe));
    std::atomic_bool* __status = new std::atomic_bool[__size];

    /*
     * First, walk through the entire array and mark items that are on the
     * correct side of the pivot as true, and the others as false.
     */
    _PSTL_PRAGMA(omp taskloop shared(__status))
    for (std::size_t __index = 0U; __index < __size; ++__index)
    {
        auto __item = std::next(__xs, __index);
        if (__index < __nsort)
        {
            __status[__index].store(__comp(*__item, *__pivot));
        }
        else
        {
            __status[__index].store(__comp(*__pivot, *__item));
        }
    }

    /*
     * Second, walk through the first __nsort items of the array and move
     * any items that are not in the right place. The status array is used
     * to locate places outside the partition where values can be safely
     * swapped.
     */
    _PSTL_PRAGMA(omp taskloop shared(__status))
    for (std::size_t __index = 0U; __index < __nsort; ++__index)
    {
        // If the item is already in the right place, move along.
        if (__status[__index].load())
        {
            continue;
        }

        // Otherwise, find the an item that can be moved into this
        // spot safely.
        for (std::size_t __swap_index = __nsort; __swap_index < __size; ++__swap_index)
        {
            // Try to capture this slot by using compare and exchange. If we
            // are able to capture the slot then perform a swap and exit this
            // loop.
            if (__status[__swap_index].load() == false && __status[__swap_index].exchange(true) == false)
            {
                auto __current_item = std::next(__xs, __index);
                auto __swap_item = std::next(__xs, __swap_index);
                std::iter_swap(__current_item, __swap_item);
                break;
            }
        }
    }

    delete[] __status;
}

template <typename _RandomAccessIterator, typename _Compare>
void
__parallel_stable_sort_body(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _Compare __comp);

template <typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
void
__parallel_stable_partial_sort(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _Compare __comp,
                               _LeafSort __leaf_sort, std::size_t __nsort)
{
    auto __pivot = __parallel_find_pivot(__xs, __xe, __comp, __nsort);
    __parallel_partition(__xs, __xe, __pivot, __comp, __nsort);
    auto __part_end = std::next(__xs, __nsort);

    if (__nsort <= __default_chunk_size)
    {
        __leaf_sort(__xs, __part_end, __comp);
    }
    else
    {
        __parallel_stable_sort_body(__xs, __part_end, __comp);
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
