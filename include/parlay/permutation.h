#ifndef PARLAY_COUNTING_SORT_H_
#define PARLAY_COUNTING_SORT_H_

#include <cassert>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

#include "sequence_ops.h"
#include "uninitialized_sequence.h"

#include "../monoid.h"
#include "../parallel.h"
#include "../sequence.h"
#include "../slice.h"
#include "../utilities.h"


namespace parlay {
namespace internal {

// the following parameters can be tuned
constexpr const size_t SEQ_THRESHOLD = 8192;
constexpr const size_t BUCKET_FACTOR = 32;
constexpr const size_t LOW_BUCKET_FACTOR = 16;

constexpr const size_t BLOCK_SIZE = 32;

// local classification, at the end leftover blocks still in buffers and counts used to recover end of written buffers
template <typename InS, typename BufferIterator, typename BufferKeyIterator, typename CountIterator, typename KeyS>
void local_classification(InS In, KeyS Keys, BufferIterator bufferIn, BufferKeyIterator bufferKeys, 
                    CountIterator counts, size_t num_buckets) {
    using s_size_t = typename std::iterator_traits<BufferIterator>::value_type;
    size_t n = In.size();
    // use local counts to avoid false sharing
    auto local_counts = sequence<s_size_t>(num_buckets);
    size_t filled_idx = 0;
    for (size_t j = 0; j < n; j++) {
        size_t k = Keys[j];
        assert(k < num_buckets);
        local_counts[k]++;

        auto buff_idx = local_counts[k] % BLOCK_SIZE;
        if (buff_idx == 0) {
            // write buffer to input TODO: is there a builtin for this?
            for (size_t i = BLOCK_SIZE * k; i < BLOCK_SIZE * (k + 1); i++) {
                InS[filled_idx] = bufferIn[i];
                KeyS[filled_idx] = bufferKeys[i];
                filled_idx++;
            }
        }
        // write to buffer
        bufferIn[buff_idx] = InS[j];
        bufferKeys[buff_idx] = k;
    }

    for (size_t i = 0; i < num_buckets; i++) counts[i] = local_counts[i];
}



// IPS4o blocked permutation and cleanup
// returns counts, and a flag
// If skip_if_in_one and returned flag is true, then the Input was alread
// sorted, and it has not been moved to the output.
//
// Values are transferred from In to Out as per the type of assignment_tag.
// E.g. If assignment_tag is parlay::copy_assign_tag, values are copied,
// if it is parlay::uninitialized_move_tag, they are moved assuming that
// Out is uninitialized, etc.
template <typename assignment_tag, typename s_size_t, typename InIterator, typename OutIterator, typename KeyIterator>
std::pair<sequence<size_t>, bool> ips4o_permute(slice<InIterator, InIterator> In,
                                              slice<OutIterator, OutIterator> Out,
                                              slice<KeyIterator, KeyIterator> Keys,
                                              size_t num_buckets,
                                              float parallelism = 1.0,
                                              bool skip_if_in_one = false) {
    using T = typename slice<InIterator, InIterator>::value_type;
    size_t n = In.size();
    size_t num_threads = num_workers();
    bool is_nested = parallelism < .5;

    // pick number of blocks for sufficient parallelism but to make sure
    // cost on counts is not to high (i.e. bucket upper).
    // size_t par_lower = 1 + static_cast<size_t>(round(num_threads * parallelism * 9));
    // size_t size_lower = 1;  // + n * sizeof(T) / 2000000;
    // size_t bucket_upper =
    //     1 + n * sizeof(T) / (4 * num_buckets * sizeof(s_size_t));
    // size_t num_blocks = (std::min)(bucket_upper, (std::max)(par_lower, size_lower));
    // size_t num_blocks = 1 + n * sizeof(T) / std::max<size_t>(num_buckets * 500, 5000);
    size_t num_blocks = 1 + n * sizeof(T) / (num_buckets * 5000);
    
    // if insufficient parallelism, sort sequentially **TODO not updated 
    if (n < SEQ_THRESHOLD || num_blocks == 1 || num_threads == 1) {
        return std::make_pair(
        seq_count_sort<assignment_tag>(In, Out, Keys, num_buckets),
        false);
    }

    size_t stripe_size = ((n - 1) / num_threads) + 1;
    auto bufferIn = sequence<size_t>::uninitialized(num_threads * BLOCK_SIZE * num_buckets);
    auto bufferKeys = sequence<size_t>::uninitialized(num_threads * BLOCK_SIZE * num_buckets);
    auto counts = sequence<size_t>::uninitialized(num_threads * num_buckets);


    // Local Classification
    // each thread gets own set of buffer blocks
    // 1 buffer block for each stripe
    parallel_for(0, num_threads,
               [&](size_t i) {
                 size_t start = (std::min)(i * stripe_size, n);
                 size_t end = (std::min)(start + stripe_size, n);
                 local_classification(In.cut(start, end), make_slice(Keys).cut(start, end),
                            bufferIn + i * num_buckets * BLOCK_SIZE, bufferKeys + i * num_buckets * BLOCK_SIZE,
                            counts.begin() + i * num_buckets, num_buckets);
               },
               1, is_nested);

    // Blocked Distribute
    // :sob:
    
    // Cleanup
    // :sob:
    {
        if (kIsParallel) overflow_ = shared_->overflow;

        // Distribute buckets among threads
        const int num_buckets = num_buckets_;
        const int buckets_per_thread = (num_buckets + num_threads_ - 1) / num_threads_;
        int my_first_bucket = my_id_ * buckets_per_thread;
        int my_last_bucket = (my_id_ + 1) * buckets_per_thread;
        my_first_bucket = num_buckets < my_first_bucket ? num_buckets : my_first_bucket;
        my_last_bucket = num_buckets < my_last_bucket ? num_buckets : my_last_bucket;

        // Save excess elements at right end of stripe
        const auto in_swap_buffer = !kIsParallel
                                            ? std::pair<int, diff_t>(-1, 0)
                                            : saveMargins(my_last_bucket);
        if (kIsParallel) shared_->sync.barrier();

        // Write remaining elements
        writeMargins<kIsParallel>(my_first_bucket, my_last_bucket, overflow_bucket,
                                  in_swap_buffer.first, in_swap_buffer.second);
    }
}

}
}
#endif  // PARLAY_COUNTING_SORT_H_
