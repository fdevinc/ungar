// nanorange/detail/ranges/primitives.hpp
//
// Copyright (c) 2018 Tristan Brindle (tcbrindle at gmail dot com)
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NANORANGE_DETAIL_RANGES_PRIMITIVES_HPP_INCLUDED
#define NANORANGE_DETAIL_RANGES_PRIMITIVES_HPP_INCLUDED

#include <nanorange/detail/ranges/basic_range_types.hpp>
#include <nanorange/detail/ranges/begin_end.hpp>

NANO_BEGIN_NAMESPACE

// [range.prim.size]

template <typename>
inline constexpr bool disable_sized_range = false;

namespace detail {
namespace size_ {

template <typename T>
void size(T&&) = delete;

// For some reason MSVC doesn't mind poison pills,
// as long as there are two
template <typename T>
void size(T&) = delete;

struct fn {
private:
    template <typename T, std::size_t N>
    static constexpr std::size_t impl(const T(&&)[N], priority_tag<3>) noexcept
    {
        return N;
    }

    template <typename T, std::size_t N>
    static constexpr std::size_t impl(const T (&)[N], priority_tag<3>) noexcept
    {
        return N;
    }

    template <typename T,
              typename I = decltype(decay_copy(std::declval<T>().size()))>
    static constexpr auto impl(T&& t, priority_tag<2>) noexcept(
        noexcept(decay_copy(std::forward<T>(t).size())))
        -> std::enable_if_t<
            integral<I> && !disable_sized_range<remove_cvref_t<T>>, I>
    {
        return decay_copy(std::forward<T>(t).size());
    }

    template <typename T,
              typename I = decltype(decay_copy(size(std::declval<T>())))>
    static constexpr auto impl(T&& t, priority_tag<1>) noexcept(
        noexcept(decay_copy(size(std::forward<T>(t)))))
        -> std::enable_if_t<
            integral<I> && !disable_sized_range<remove_cvref_t<T>>, I>
    {
        return decay_copy(size(std::forward<T>(t)));
    }

    template <typename T,
              typename I = decltype(ranges::begin(std::declval<T>())),
              typename S = decltype(ranges::end(std::declval<T>())),
              typename D = decltype(decay_copy(std::declval<S>() -
                                               std::declval<I>()))>
    static constexpr auto impl(T&& t, priority_tag<0>) noexcept(
        noexcept(decay_copy(ranges::end(t) - ranges::begin(t))))
        -> std::enable_if_t<
            !std::is_array<remove_cvref_t<T>>::value && // MSVC sillyness?
                sized_sentinel_for<S, I> && forward_iterator<I>,
            D>
    {
        return decay_copy(ranges::end(t) - ranges::begin(t));
    }

public:
    template <typename T>
    constexpr auto operator()(T&& t) const
        noexcept(noexcept(fn::impl(std::forward<T>(t), priority_tag<3>{})))
            -> decltype(fn::impl(std::forward<T>(t), priority_tag<3>{}))
    {
        return fn::impl(std::forward<T>(t), priority_tag<3>{});
    }
};

} // namespace size_
} // namespace detail

NANO_INLINE_VAR(detail::size_::fn, size)

// [range.prim.ssize]

namespace detail {
namespace ssize_ {

struct fn {
private:
    template <typename T>
    using ssize_return_t =
        std::conditional_t<sizeof(range_difference_t<T>) <
                               sizeof(std::ptrdiff_t),
                           std::ptrdiff_t, range_difference_t<T>>;

    template <typename T>
    static constexpr auto
    impl(T&& t) noexcept(noexcept(ranges::size(std::forward<T>(t))))
        -> decltype(ranges::size(std::forward<T>(t)), ssize_return_t<T>())
    {
        return static_cast<ssize_return_t<T>>(ranges::size(std::forward<T>(t)));
    }

public:
    template <typename T>
    constexpr auto operator()(T&& t) const
        noexcept(noexcept(fn::impl(std::forward<T>(t))))
            -> decltype(fn::impl(std::forward<T>(t)))
    {
        return fn::impl(std::forward<T>(t));
    }
};

} // namespace ssize_
} // namespace detail

NANO_INLINE_VAR(detail::ssize_::fn, ssize)

// [range.prim.empty]

namespace detail {
namespace empty_ {

struct fn {
private:
    template <typename T>
    static constexpr auto
    impl(T&& t,
         priority_tag<2>) noexcept(noexcept((bool(std::forward<T>(t).empty()))))
        -> decltype((bool(std::forward<T>(t).empty())))
    {
        return bool((std::forward<T>(t).empty()));
    }

    template <typename T>
    static constexpr auto impl(T&& t, priority_tag<1>) noexcept(
        noexcept(ranges::size(std::forward<T>(t)) == 0))
        -> decltype(ranges::size(std::forward<T>(t)) == 0)
    {
        return ranges::size(std::forward<T>(t)) == 0;
    }

    template <typename T,
              typename I = decltype(ranges::begin(std::declval<T>()))>
    static constexpr auto
    impl(T&& t,
         priority_tag<0>) noexcept(noexcept(ranges::begin(t) == ranges::end(t)))
        -> std::enable_if_t<forward_iterator<I>,
                            decltype(ranges::begin(t) == ranges::end(t))>
    {
        return ranges::begin(t) == ranges::end(t);
    }

public:
    template <typename T>
    constexpr auto operator()(T&& t) const
        noexcept(noexcept(fn::impl(std::forward<T>(t), priority_tag<2>{})))
            -> decltype(fn::impl(std::forward<T>(t), priority_tag<2>{}))
    {
        return fn::impl(std::forward<T>(t), priority_tag<2>{});
    }
};

} // namespace empty_
} // namespace detail

NANO_INLINE_VAR(detail::empty_::fn, empty)

namespace detail {

template <typename, typename = void>
inline constexpr bool is_object_pointer_v = false;

template <typename P>
inline constexpr bool is_object_pointer_v<P,
    std::enable_if_t<std::is_pointer_v<P> &&
                     std::is_object_v<iter_value_t<P>>>> = true;

namespace data_ {

struct fn {
private:
    template <typename T, typename D = decltype(decay_copy(std::declval<T&>().data()))>
    static constexpr auto
    impl(T& t, priority_tag<1>) noexcept(noexcept(decay_copy(t.data())))
        -> std::enable_if_t<is_object_pointer_v<D>, D>
    {
        return t.data();
    }

    template <typename T>
    static constexpr auto
    impl(T&& t,
         priority_tag<0>) noexcept(noexcept(ranges::begin(std::forward<T>(t))))
        -> std::enable_if_t<
            is_object_pointer_v<decltype(ranges::begin(std::forward<T>(t)))>,
            decltype(ranges::begin(std::forward<T>(t)))>
    {
        return ranges::begin(std::forward<T>(t));
    }

public:
    template <typename T>
    constexpr auto operator()(T&& t) const
        noexcept(noexcept(fn::impl(std::forward<T>(t), priority_tag<1>{})))
            -> decltype(fn::impl(std::forward<T>(t), priority_tag<1>{}))
    {
        return fn::impl(std::forward<T>(t), priority_tag<1>{});
    }
};

} // namespace data_
} // namespace detail

NANO_INLINE_VAR(detail::data_::fn, data)

namespace detail {
namespace cdata_ {

struct fn {
private:
    template <typename T, typename U = std::remove_reference_t<T>,
	      std::enable_if_t<std::is_lvalue_reference_v<T>, int> = 0>
    static constexpr auto impl(T&& t)
        noexcept(noexcept(ranges::data(static_cast<const U&>(t))))
	-> decltype(ranges::data(static_cast<const U&>(t)))
    {
	return ranges::data(static_cast<const U&>(t));
    }

    template <typename T,
	      std::enable_if_t<!std::is_lvalue_reference_v<T>, int> = 0>
    static constexpr auto impl(T&& t)
        noexcept(noexcept(ranges::data(static_cast<const T&&>(t))))
	-> decltype(ranges::data(static_cast<const T&&>(t)))
    {
	return ranges::data(static_cast<const T&&>(t));
    }

public:
    template <typename T>
    constexpr auto operator()(T&& t) const
        noexcept(noexcept(fn::impl(std::forward<T>(t))))
	-> decltype(fn::impl(std::forward<T>(t)))
    {
	return fn::impl(std::forward<T>(t));
    }
};

} // namespace cdata_
} // namespace detail

NANO_INLINE_VAR(detail::cdata_::fn, cdata)

NANO_END_NAMESPACE

#endif
