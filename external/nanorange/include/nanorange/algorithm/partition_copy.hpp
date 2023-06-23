// nanorange/algorithm/partition_copy.hpp
//
// Copyright (c) 2018 Tristan Brindle (tcbrindle at gmail dot com)
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NANORANGE_ALGORITHM_PARTITION_COPY_HPP_INCLUDED
#define NANORANGE_ALGORITHM_PARTITION_COPY_HPP_INCLUDED

#include <nanorange/detail/algorithm/result_types.hpp>
#include <nanorange/ranges.hpp>

NANO_BEGIN_NAMESPACE

template <typename I, typename O1, typename O2>
using partition_copy_result = in_out_out_result<I, O1, O2>;

namespace detail {

struct partition_copy_fn {
private:
    template <typename I, typename S, typename O1, typename O2,
              typename Pred, typename Proj>
    static constexpr partition_copy_result<I, O1, O2>
    impl(I first, S last, O1 out_true, O2 out_false, Pred& pred, Proj& proj)
    {
        while (first != last) {
            auto&& val = *first;
            if (nano::invoke(pred, nano::invoke(proj, val))) {
                *out_true = std::forward<decltype(val)>(val);
                ++out_true;
            } else {
                *out_false = std::forward<decltype(val)>(val);
                ++out_false;
            }
            ++first;
        }

        return {std::move(first), std::move(out_true), std::move(out_false)};
    }

public:
    template <typename I, typename S, typename O1, typename O2,
              typename Pred, typename Proj = identity>
    constexpr std::enable_if_t<
        input_iterator<I> && sentinel_for<S, I> &&
        weakly_incrementable<O1> &&
        weakly_incrementable<O2> &&
            indirect_unary_predicate<Pred, projected<I, Proj>> &&
            indirectly_copyable<I, O1> && indirectly_copyable<I, O2>,
        partition_copy_result<I, O1, O2>>
    operator()(I first, S last, O1 out_true, O2 out_false, Pred pred,
               Proj proj = Proj{}) const
    {
        return partition_copy_fn::impl(std::move(first), std::move(last),
                                       std::move(out_true), std::move(out_false),
                                       pred, proj);
    }

    template <typename Rng, typename O1, typename O2,
            typename Pred, typename Proj = identity>
    constexpr std::enable_if_t<
        input_range<Rng> &&
        weakly_incrementable<O1> &&
        weakly_incrementable<O2> &&
            indirect_unary_predicate<Pred, projected<iterator_t<Rng>, Proj>> &&
            indirectly_copyable<iterator_t<Rng>, O1> &&
            indirectly_copyable<iterator_t<Rng>, O2>,
        partition_copy_result<borrowed_iterator_t<Rng>, O1, O2>>
    operator()(Rng&& rng, O1 out_true, O2 out_false, Pred pred,
            Proj proj = Proj{}) const
    {
        return partition_copy_fn::impl(nano::begin(rng), nano::end(rng),
                                       std::move(out_true), std::move(out_false),
                                       pred, proj);
    }
};

}

NANO_INLINE_VAR(detail::partition_copy_fn, partition_copy)

NANO_END_NAMESPACE

#endif
