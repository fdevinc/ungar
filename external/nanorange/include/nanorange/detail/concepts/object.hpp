// nanorange/detail/concepts/object.hpp
//
// Copyright (c) 2018 Tristan Brindle (tcbrindle at gmail dot com)
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NANORANGE_DETAIL_CONCEPTS_OBJECT_HPP_INCLUDED
#define NANORANGE_DETAIL_CONCEPTS_OBJECT_HPP_INCLUDED

#include <nanorange/detail/concepts/comparison.hpp>
#include <nanorange/detail/concepts/core.hpp>
#include <nanorange/detail/functional/invoke.hpp>

NANO_BEGIN_NAMESPACE

// [concept.movable]
namespace detail {

struct movable_concept {
    template <typename T>
    static auto test(long) -> std::false_type;

    template <typename T>
    static auto test(int) -> std::enable_if_t<
        std::is_object_v<T> && move_constructible<T> &&
        assignable_from<T&, T> && swappable<T>,
        std::true_type>;
};
}

template <typename T>
NANO_CONCEPT movable = decltype(detail::movable_concept::test<T>(0))::value;

// [concept.copyable]
namespace detail {

struct copyable_concept {
    template <typename>
    static auto test(long) -> std::false_type;

    template <typename T>
    static auto test(int) -> std::enable_if_t<
        copy_constructible<T> && movable<T> && assignable_from<T&, T&> &&
        assignable_from<T&, const T&> && assignable_from<T&, const T>,
        std::true_type>;

};

}

template <typename T>
NANO_CONCEPT copyable = decltype(detail::copyable_concept::test<T>(0))::value;

// [concept.semiregular]
template <typename T>
NANO_CONCEPT semiregular = copyable<T> && default_initializable<T>;

// [concept.regular]
template <typename T>
NANO_CONCEPT regular = semiregular<T> && equality_comparable<T>;

// [concept.invocable]
namespace detail {

struct invocable_concept {
    // FIXME (Clang): https://bugs.llvm.org/show_bug.cgi?id=21446
#if (defined(__clang_major__) && (defined(__apple_build_version__) ||__clang_major__ < 7))
    template <typename F, typename... Args>
    auto requires_(F&& f, Args&&... args) -> invoke_result_t<F, Args...>;
#else
    template <typename F, typename... Args>
    auto requires_(F&& f, Args&&... args) -> decltype(
        nano::invoke(std::forward<F>(f), std::forward<Args>(args)...)
    );
#endif
};

} // namespace detail

template <typename F, typename... Args>
NANO_CONCEPT invocable = detail::requires_<detail::invocable_concept, F, Args...>;

// [concept.regularinvocable]
template <typename F, typename... Args>
NANO_CONCEPT regular_invocable = invocable<F, Args...>;

// [concept.predicate]
namespace detail {

struct predicate_concept {
    template <typename, typename...>
    static auto test(long) -> std::false_type;

    template <typename F, typename... Args>
    static auto test(int) -> std::enable_if_t<
        regular_invocable<F, Args...> &&
        boolean_testable<invoke_result_t<F, Args...>>,
        std::true_type>;

};

}

template <typename F, typename... Args>
NANO_CONCEPT predicate = decltype(detail::predicate_concept::test<F, Args...>(0))::value;

// [concept.relation]
template <typename R, typename T, typename U>
NANO_CONCEPT relation =
    predicate<R, T, T> && predicate<R, U, U> &&
    predicate<R, T, U> && predicate<R, U, T>;

// [concept.equiv]
template <typename R, typename T, typename U>
NANO_CONCEPT equivalence_relation = relation<R, T, U>;

// [concept.strictweakorder]
template <typename R, typename T, typename U>
NANO_CONCEPT strict_weak_order = relation<R, T, U>;

NANO_END_NAMESPACE

#endif
