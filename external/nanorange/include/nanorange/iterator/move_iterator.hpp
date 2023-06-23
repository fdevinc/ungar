// nanorange/iterator/move_iterator.hpp
//
// Copyright (c) 2018 Tristan Brindle (tcbrindle at gmail dot com)
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NANORANGE_ITERATOR_MOVE_ITERATOR_HPP_INCLUDED
#define NANORANGE_ITERATOR_MOVE_ITERATOR_HPP_INCLUDED

#include <nanorange/detail/iterator/concepts.hpp>
#include <nanorange/iterator/default_sentinel.hpp>

NANO_BEGIN_NAMESPACE

namespace move_iterator_ {

template <typename I>
class move_iterator {

    static_assert(
        input_iterator<I>,
        "Template argument to move_iterator must model InputIterator");

    template <typename I2>
    friend class move_iterator;

public:
    using iterator_type = I;
    using difference_type = iter_difference_t<I>;
    using value_type = iter_value_t<I>;
    using iterator_category = input_iterator_tag;
    using reference = iter_rvalue_reference_t<I>;

    constexpr move_iterator() = default;

    explicit constexpr move_iterator(I i) : current_(std::move(i)) {}

    template <typename U, std::enable_if_t<convertible_to<U, I>, int> = 0>
    constexpr move_iterator(const move_iterator<U>& i) : current_(i.current_)
    {}

    template <typename U>
    constexpr std::enable_if_t<convertible_to<U, I>, move_iterator&>
    operator=(const move_iterator<U>& i)
    {
        current_ = i.current_;
        return *this;
    }

    constexpr I base() const { return current_; }

    constexpr reference operator*() const { return iter_move(current_); }

    constexpr move_iterator& operator++()
    {
        ++current_;
        return *this;
    }

    template <typename II = I>
    constexpr auto operator++(int) -> std::enable_if_t<!forward_iterator<II>>
    {
        ++current_;
    }

    template <typename II = I>
    constexpr auto operator++(int)
        -> std::enable_if_t<forward_iterator<II>, move_iterator>
    {
        move_iterator tmp = *this;
        ++current_;
        return tmp;
    }

    template <typename II = I>
    constexpr auto operator--()
        -> std::enable_if_t<bidirectional_iterator<II>, move_iterator&>
    {
        --current_;
        return *this;
    }

    template <typename II = I>
    constexpr auto operator--(int)
        -> std::enable_if_t<bidirectional_iterator<II>, move_iterator>
    {
        move_iterator tmp = *this;
        --current_;
        return tmp;
    }

    template <typename II = I>
    constexpr auto operator+(difference_type n) const
        -> std::enable_if_t<random_access_iterator<II>, move_iterator>
    {
        return move_iterator(current_ + n);
    }

    template <typename II = I>
    constexpr auto operator+=(difference_type n)
        -> std::enable_if_t<random_access_iterator<II>, move_iterator&>
    {
        current_ += n;
        return *this;
    }

    template <typename II = I>
    constexpr auto operator-(difference_type n) const
        -> std::enable_if_t<random_access_iterator<II>, move_iterator>
    {
        return move_iterator(current_ - n);
    }

    template <typename II = I>
    constexpr auto operator-=(difference_type n)
        -> std::enable_if_t<random_access_iterator<II>, move_iterator&>
    {
        current_ -= n;
        return *this;
    }

    template <typename II = I>
    constexpr auto operator[](difference_type n) const
        -> std::enable_if_t<random_access_iterator<II>, reference>
    //   -> decltype(auto)
    {
        return iter_move(current_ + n);
    }

    friend constexpr iter_rvalue_reference_t<I>
    iter_move(const move_iterator& i) noexcept(
        noexcept(ranges::iter_move(i.current_)))
    {
        return ranges::iter_move(i.current_);
    }

    template <typename I2>
    friend constexpr auto
    iter_swap(const move_iterator& x, const move_iterator<I2>& y) noexcept(
        noexcept(ranges::iter_swap(x.current_, y.current_)))
        -> std::enable_if_t<indirectly_swappable<I2, I>>
    {
        ranges::iter_swap(x.current_, y.current_);
    }

private:
    I current_{};
};

template <typename I1, typename I2>
constexpr auto operator==(const move_iterator<I1>& x,
                          const move_iterator<I2>& y)
    -> std::enable_if_t<equality_comparable_with<I1, I2>, bool>
{
    return x.base() == y.base();
}

template <typename I1, typename I2>
constexpr auto operator!=(const move_iterator<I1>& x,
                          const move_iterator<I2>& y)
    -> std::enable_if_t<equality_comparable_with<I1, I2>, bool>
{
    return !(x == y);
}

template <typename I1, typename I2>
constexpr auto operator<(const move_iterator<I1>& x, const move_iterator<I2>& y)
    -> std::enable_if_t<totally_ordered_with<I1, I2>, bool>
{
    return x.base() < y.base();
}

template <typename I1, typename I2>
constexpr auto operator<=(const move_iterator<I1>& x,
                          const move_iterator<I2>& y)
    -> std::enable_if_t<totally_ordered_with<I1, I2>, bool>
{
    return !(y < x);
}

template <typename I1, typename I2>
constexpr auto operator>(const move_iterator<I1>& x, const move_iterator<I2>& y)
    -> std::enable_if_t<totally_ordered_with<I1, I2>, bool>
{
    return y < x;
}

template <typename I1, typename I2>
constexpr auto operator>=(const move_iterator<I1>& x,
                          const move_iterator<I2>& y)
    -> std::enable_if_t<totally_ordered_with<I1, I2>, bool>
{
    return !(x < y);
}

template <typename I1, typename I2>
constexpr auto operator-(const move_iterator<I1>& x, const move_iterator<I2>& y)
    -> std::enable_if_t<sized_sentinel_for<I1, I2>, iter_difference_t<I2>>
{
    return x.base() - y.base();
}

template <typename I>
constexpr auto operator+(iter_difference_t<I> n, const move_iterator<I>& x)
    -> std::enable_if_t<random_access_iterator<I>, move_iterator<I>>
{
    return x + n;
}

} // namespace move_iterator_

using move_iterator_::move_iterator;

template <typename I>
constexpr auto make_move_iterator(I i)
    -> std::enable_if_t<input_iterator<I>, move_iterator<I>>
{
    return move_iterator<I>(std::move(i));
}

template <typename S>
class move_sentinel {
    static_assert(semiregular<S>,
                  "Template argument to move_sentinel must model Semiregular");

public:
    constexpr move_sentinel() = default;

    constexpr explicit move_sentinel(S s) : last_(std::move(s)) {}

    template <typename U, std::enable_if_t<convertible_to<U, S>, int> = 0>
    constexpr move_sentinel(const move_sentinel<U>& s) : last_(s.base())
    {}

    template <typename U>
    constexpr auto operator=(const move_sentinel<U>& s)
        -> std::enable_if_t<convertible_to<U, S>, move_sentinel&>
    {
        last_ = s.base();
        return *this;
    }

    constexpr S base() const { return last_; }

private:
    S last_{};
};

template <typename I, typename S>
constexpr auto operator==(const move_iterator<I>& i, const move_sentinel<S>& s)
    -> std::enable_if_t<sentinel_for<S, I>, bool>
{
    return i.base() == s.base();
}

template <typename I, typename S>
constexpr auto operator==(const move_sentinel<S>& s, const move_iterator<I>& i)
    -> std::enable_if_t<sentinel_for<S, I>, bool>
{
    return i.base() == s.base();
}

template <typename I, typename S>
constexpr auto operator!=(const move_iterator<I>& i, const move_sentinel<S>& s)
    -> std::enable_if_t<sentinel_for<S, I>, bool>
{
    return !(i == s);
}

template <typename I, typename S>
constexpr auto operator!=(const move_sentinel<S>& s, const move_iterator<I>& i)
    -> std::enable_if_t<sentinel_for<S, I>, bool>
{
    return !(i == s);
}

template <typename I, typename S>
constexpr auto operator-(const move_sentinel<S>& s, const move_iterator<I>& i)
    -> std::enable_if_t<sized_sentinel_for<S, I>, iter_difference_t<I>>
{
    return s.base() - i.base();
}

template <typename I, typename S>
constexpr auto operator-(const move_iterator<I>& i, const move_sentinel<S>& s)
    -> std::enable_if_t<sized_sentinel_for<S, I>, iter_difference_t<I>>
{
    return i.base() - s.base();
}

template <typename S>
constexpr auto make_move_sentinel(S s)
    -> std::enable_if_t<semiregular<S>, move_sentinel<S>>
{
    return move_sentinel<S>(std::move(s));
}

NANO_END_NAMESPACE

namespace std {

template <typename I>
struct iterator_traits<::nano::move_iterator_::move_iterator<I>> {
    using value_type =
        typename ::nano::move_iterator_::move_iterator<I>::value_type;
    using reference =
        typename ::nano::move_iterator_::move_iterator<I>::reference;
    using pointer = value_type*;
    using difference_type =
        typename ::nano::move_iterator_::move_iterator<I>::difference_type;
    using iterator_category =
        typename ::nano::move_iterator_::move_iterator<I>::iterator_category;
};

} // namespace std

#endif
