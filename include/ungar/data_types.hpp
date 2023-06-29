/******************************************************************************
 *
 * @file ungar/data_types.hpp
 * @author Flavio De Vincenti (flavio.devincenti@inf.ethz.ch)
 *
 * @section LICENSE
 * -----------------------------------------------------------------------
 *
 * Copyright 2023 Flavio De Vincenti
 *
 * -----------------------------------------------------------------------
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS
 * IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 ******************************************************************************/

#ifndef _UNGAR__DATA_TYPES_HPP_
#define _UNGAR__DATA_TYPES_HPP_

#include "nanorange/ranges.hpp"
#include "nanorange/views.hpp"

#include "Eigen/Geometry"
#include "Eigen/Sparse"

#include "boost/hana.hpp"
#include "boost/hana/ext/std/array.hpp"
#include "boost/hana/ext/std/tuple.hpp"

#ifndef UNGAR_CODEGEN_FOLDER
#define UNGAR_CODEGEN_FOLDER (std::filesystem::temp_directory_path() / "ungar_codegen").c_str()
#endif

namespace std {

template <std::size_t N, typename... _Ts>
struct tuple_element<N, boost::hana::tuple<_Ts...>> {
    using type = typename decltype(+boost::hana::tuple_t<_Ts...>[boost::hana::size_c<N>])::type;
};

template <typename... _Ts>
struct tuple_size<boost::hana::tuple<_Ts...>>
    : public integral_constant<std::size_t, sizeof...(_Ts)> {};

}  // namespace std

namespace boost {
namespace hana {

template <std::size_t N, typename... _Ts>
constexpr decltype(auto) get(hana::tuple<_Ts...>& tuple) {
    return tuple[hana::size_c<N>];
}

template <std::size_t N, typename... _Ts>
constexpr decltype(auto) get(const hana::tuple<_Ts...>& tuple) {
    return tuple[hana::size_c<N>];
}

template <std::size_t N, typename... _Ts>
constexpr decltype(auto) get(hana::tuple<_Ts...>&& tuple) {
    return static_cast<hana::tuple<_Ts...>&&>(tuple)[hana::size_c<N>];
}

template <std::size_t N, typename... _Ts>
constexpr decltype(auto) get(const hana::tuple<_Ts...>&& tuple) {
    return static_cast<const hana::tuple<_Ts...>&&>(tuple)[hana::size_c<N>];
}

}  // namespace hana
}  // namespace boost

namespace Ungar {

namespace hana = boost::hana;

using namespace std::literals;
using namespace hana::literals;

using real_t = double;

using index_t     = Eigen::Index;
using time_step_t = index_t;

template <typename _T, template <typename...> typename _TemplateType>
struct is_specialization_of : std::false_type {};
template <template <typename...> typename _TemplateType, typename... _Args>
struct is_specialization_of<_TemplateType<_Args...>, _TemplateType> : std::true_type {};
template <typename _T, template <typename...> typename _TemplateType>
constexpr bool is_specialization_of_v = is_specialization_of<_T, _TemplateType>::value;

template <class _T>
struct is_std_array : std::false_type {};
template <class _T, std::size_t _N>
struct is_std_array<std::array<_T, _N>> : std::true_type {};
template <class _T>
constexpr bool is_std_array_v = is_std_array<_T>::value;

template <class _T>
struct std_array_size;
template <class _T, std::size_t _N>
struct std_array_size<std::array<_T, _N>> : std::integral_constant<size_t, _N> {};
template <class _T>
constexpr bool std_array_size_v = std_array_size<_T>::value;

template <class _T>
struct std_array_empty;
template <class _T, std::size_t _N>
struct std_array_empty<std::array<_T, _N>> : std::integral_constant<bool, !_N> {};
template <class _T>
constexpr bool std_array_empty_v = std_array_empty<_T>::value;

template <class _T>
struct is_std_optional : std::false_type {};
template <class _T>
struct is_std_optional<std::optional<_T>> : std::true_type {};
template <class _T>
constexpr bool is_std_optional_v = is_std_optional<_T>::value;

template <class _T>
struct is_hana_integral_constant_tag : std::false_type {};
template <class _T>
struct is_hana_integral_constant_tag<hana::integral_constant_tag<_T>> : std::true_type {};
template <class _T>
constexpr bool is_hana_integral_constant_tag_v = is_hana_integral_constant_tag<_T>::value;

template <class _T>
struct is_hana_integral_constant : is_hana_integral_constant_tag<hana::tag_of_t<_T>> {};
template <class _T>
constexpr bool is_hana_integral_constant_v = is_hana_integral_constant<_T>::value;

template <class _T>
struct is_hana_tuple : std::is_same<hana::tag_of_t<_T>, hana::tuple_tag> {};
template <class _T>
constexpr bool is_hana_tuple_v = is_hana_tuple<_T>::value;

template <class _T>
struct is_hana_set : std::is_same<hana::tag_of_t<_T>, hana::set_tag> {};
template <class _T>
constexpr bool is_hana_set_v = is_hana_set<_T>::value;

template <class _T>
struct remove_cvref {
    typedef std::remove_cv_t<std::remove_reference_t<_T>> type;
};
template <class _T>
using remove_cvref_t = typename remove_cvref<_T>::type;

template <typename _T1, typename _T2, typename... _Args>
struct same
    : std::integral_constant<bool,
                             std::is_same_v<_T1, _T2> && (std::is_same_v<_T1, _Args> && ...)> {};
template <typename _T1, typename _T2, typename... _Args>
constexpr bool same_v = same<_T1, _T2, _Args...>::value;

template <typename _ContiguousRangeOf, typename _T>
struct contiguous_range_of
    : std::conjunction<std::bool_constant<nano::ranges::contiguous_range<_ContiguousRangeOf>>,
                       std::is_same<nano::ranges::range_value_t<_ContiguousRangeOf>, _T>> {};
template <typename _ContiguousRangeOf, typename _T>
constexpr bool contiguous_range_of_v = contiguous_range_of<_ContiguousRangeOf, _T>::value;

template <typename... _BoolConstants>
struct conjunction : std::bool_constant<(_BoolConstants::value && ...)> {};

template <typename _Matrix>
struct is_dense_matrix_expression
    : std::is_base_of<Eigen::MatrixBase<remove_cvref_t<_Matrix>>, remove_cvref_t<_Matrix>> {};
template <typename _Matrix>
constexpr bool is_dense_matrix_expression_v = is_dense_matrix_expression<_Matrix>::value;

namespace Internal {

template <typename _Vector>
constexpr auto IsDenseVectorExpression() {
    if constexpr (is_dense_matrix_expression_v<_Vector>) {
        return std::bool_constant<remove_cvref_t<_Vector>::ColsAtCompileTime == 1>{};
    } else {
        return std::bool_constant<false>{};
    }
}

}  // namespace Internal

template <typename _Vector>
struct is_dense_vector_expression
    : std::conjunction<is_dense_matrix_expression<_Vector>,
                       decltype(Internal::IsDenseVectorExpression<_Vector>())> {};
template <typename _Vector>
constexpr bool is_dense_vector_expression_v = is_dense_vector_expression<_Vector>::value;

template <typename _Matrix>
struct is_sparse_matrix_expression
    : std::is_base_of<Eigen::SparseMatrixBase<remove_cvref_t<_Matrix>>, remove_cvref_t<_Matrix>> {};
template <typename _Matrix>
constexpr bool is_sparse_matrix_expression_v = is_sparse_matrix_expression<_Matrix>::value;

template <class _T>
struct type_identity {
    using type = _T;
};
template <class _T>
using type_identity_t = typename type_identity<_T>::type;

#define _UNGAR_MAKE_EIGEN_TYPEDEFS_IMPL(Type, TypeSuffix, SIZE, SizeSuffix)                     \
    using Matrix##SizeSuffix##TypeSuffix    = Eigen::Matrix<Type, SIZE, SIZE>;                  \
    using Vector##SizeSuffix##TypeSuffix    = Eigen::Matrix<Type, SIZE, 1>;                     \
    using RowVector##SizeSuffix##TypeSuffix = Eigen::Matrix<Type, 1, SIZE>;                     \
                                                                                                \
    using RefToMatrix##SizeSuffix##TypeSuffix    = Eigen::Ref<Eigen::Matrix<Type, SIZE, SIZE>>; \
    using RefToVector##SizeSuffix##TypeSuffix    = Eigen::Ref<Eigen::Matrix<Type, SIZE, 1>>;    \
    using RefToRowVector##SizeSuffix##TypeSuffix = Eigen::Ref<Eigen::Matrix<Type, 1, SIZE>>;    \
    using RefToConstMatrix##SizeSuffix##TypeSuffix =                                            \
        Eigen::Ref<const Eigen::Matrix<Type, SIZE, SIZE>>;                                      \
    using RefToConstVector##SizeSuffix##TypeSuffix =                                            \
        Eigen::Ref<const Eigen::Matrix<Type, SIZE, 1>>;                                         \
    using RefToConstRowVector##SizeSuffix##TypeSuffix =                                         \
        Eigen::Ref<const Eigen::Matrix<Type, 1, SIZE>>;                                         \
                                                                                                \
    using MapToMatrix##SizeSuffix##TypeSuffix    = Eigen::Map<Eigen::Matrix<Type, SIZE, SIZE>>; \
    using MapToVector##SizeSuffix##TypeSuffix    = Eigen::Map<Eigen::Matrix<Type, SIZE, 1>>;    \
    using MapToRowVector##SizeSuffix##TypeSuffix = Eigen::Map<Eigen::Matrix<Type, 1, SIZE>>;    \
    using MapToConstMatrix##SizeSuffix##TypeSuffix =                                            \
        Eigen::Map<const Eigen::Matrix<Type, SIZE, SIZE>>;                                      \
    using MapToConstVector##SizeSuffix##TypeSuffix =                                            \
        Eigen::Map<const Eigen::Matrix<Type, SIZE, 1>>;                                         \
    using MapToConstRowVector##SizeSuffix##TypeSuffix =                                         \
        Eigen::Map<const Eigen::Matrix<Type, 1, SIZE>>
#define UNGAR_MAKE_EIGEN_TYPEDEFS(Type, TypeSuffix)          \
    _UNGAR_MAKE_EIGEN_TYPEDEFS_IMPL(Type, TypeSuffix, 2, 2); \
    _UNGAR_MAKE_EIGEN_TYPEDEFS_IMPL(Type, TypeSuffix, 3, 3); \
    _UNGAR_MAKE_EIGEN_TYPEDEFS_IMPL(Type, TypeSuffix, 4, 4); \
    _UNGAR_MAKE_EIGEN_TYPEDEFS_IMPL(Type, TypeSuffix, Eigen::Dynamic, X)

UNGAR_MAKE_EIGEN_TYPEDEFS(real_t, r);
#undef UNGAR_MAKE_EIGEN_TYPEDEFS
#undef _UNGAR_MAKE_EIGEN_TYPEDEFS_IMPL

#define UNGAR_MAKE_EIGEN_TYPEDEFS(SIZE, SizeSuffix)                                            \
    template <typename _Scalar>                                                                \
    using Matrix##SizeSuffix = Eigen::Matrix<_Scalar, SIZE, SIZE>;                             \
    template <typename _Scalar>                                                                \
    using Vector##SizeSuffix = Eigen::Matrix<_Scalar, SIZE, 1>;                                \
    template <typename _Scalar>                                                                \
    using RowVector##SizeSuffix = Eigen::Matrix<_Scalar, 1, SIZE>;                             \
                                                                                               \
    template <typename _Scalar>                                                                \
    using RefToMatrix##SizeSuffix = Eigen::Ref<Eigen::Matrix<_Scalar, SIZE, SIZE>>;            \
    template <typename _Scalar>                                                                \
    using RefToVector##SizeSuffix = Eigen::Ref<Eigen::Matrix<_Scalar, SIZE, 1>>;               \
    template <typename _Scalar>                                                                \
    using RefToRowVector##SizeSuffix = Eigen::Ref<Eigen::Matrix<_Scalar, 1, SIZE>>;            \
    template <typename _Scalar>                                                                \
    using RefToConstMatrix##SizeSuffix = Eigen::Ref<const Eigen::Matrix<_Scalar, SIZE, SIZE>>; \
    template <typename _Scalar>                                                                \
    using RefToConstVector##SizeSuffix = Eigen::Ref<const Eigen::Matrix<_Scalar, SIZE, 1>>;    \
    template <typename _Scalar>                                                                \
    using RefToConstRowVector##SizeSuffix = Eigen::Ref<const Eigen::Matrix<_Scalar, 1, SIZE>>; \
                                                                                               \
    template <typename _Scalar>                                                                \
    using MapToMatrix##SizeSuffix = Eigen::Map<Eigen::Matrix<_Scalar, SIZE, SIZE>>;            \
    template <typename _Scalar>                                                                \
    using MapToVector##SizeSuffix = Eigen::Map<Eigen::Matrix<_Scalar, SIZE, 1>>;               \
    template <typename _Scalar>                                                                \
    using MapToRowVector##SizeSuffix = Eigen::Map<Eigen::Matrix<_Scalar, 1, SIZE>>;            \
    template <typename _Scalar>                                                                \
    using MapToConstMatrix##SizeSuffix = Eigen::Map<const Eigen::Matrix<_Scalar, SIZE, SIZE>>; \
    template <typename _Scalar>                                                                \
    using MapToConstVector##SizeSuffix = Eigen::Map<const Eigen::Matrix<_Scalar, SIZE, 1>>;    \
    template <typename _Scalar>                                                                \
    using MapToConstRowVector##SizeSuffix = Eigen::Map<const Eigen::Matrix<_Scalar, 1, SIZE>>

UNGAR_MAKE_EIGEN_TYPEDEFS(2, 2);
UNGAR_MAKE_EIGEN_TYPEDEFS(3, 3);
UNGAR_MAKE_EIGEN_TYPEDEFS(4, 4);
UNGAR_MAKE_EIGEN_TYPEDEFS(Eigen::Dynamic, X);
#undef UNGAR_MAKE_EIGEN_TYPEDEFS

#define UNGAR_MAKE_EIGEN_TYPEDEFS(SIZE)                                   \
    template <typename _Scalar>                                           \
    using Matrix##SIZE##X = Eigen::Matrix<_Scalar, SIZE, Eigen::Dynamic>; \
    template <typename _Scalar>                                           \
    using Matrix##X##SIZE = Eigen::Matrix<_Scalar, Eigen::Dynamic, SIZE>

UNGAR_MAKE_EIGEN_TYPEDEFS(2);
UNGAR_MAKE_EIGEN_TYPEDEFS(3);
UNGAR_MAKE_EIGEN_TYPEDEFS(4);
#undef UNGAR_MAKE_EIGEN_TYPEDEFS

template <typename _Scalar, int SIZE>
using Vector = Eigen::Matrix<_Scalar, SIZE, 1>;
template <typename _Scalar, int SIZE>
using RowVector = Eigen::Matrix<_Scalar, 1, SIZE>;
template <typename _Scalar, int SIZE>
using RefToVector = Eigen::Ref<Eigen::Matrix<_Scalar, SIZE, 1>>;
template <typename _Scalar, int SIZE>
using RefToRowVector = Eigen::Ref<Eigen::Matrix<_Scalar, 1, SIZE>>;
template <typename _Scalar, int SIZE>
using RefToConstVector = Eigen::Ref<const Eigen::Matrix<_Scalar, SIZE, 1>>;
template <typename _Scalar, int SIZE>
using RefToConstRowVector = Eigen::Ref<const Eigen::Matrix<_Scalar, 1, SIZE>>;
template <typename _Scalar>
using SparseMatrix = Eigen::SparseMatrix<_Scalar>;

template <typename _Scalar>
using Quaternion  = Eigen::Quaternion<_Scalar>;
using Quaternionr = Eigen::Quaternion<real_t>;
template <typename _Scalar>
using MapToQuaternion  = Eigen::Map<Eigen::Quaternion<_Scalar>>;
using MapToQuaternionr = Eigen::Map<Eigen::Quaternion<real_t>>;
template <typename _Scalar>
using MapToConstQuaternion  = Eigen::Map<const Eigen::Quaternion<_Scalar>>;
using MapToConstQuaternionr = Eigen::Map<const Eigen::Quaternion<real_t>>;

template <typename _Scalar>
using AngleAxis  = Eigen::AngleAxis<_Scalar>;
using AngleAxisr = Eigen::AngleAxis<real_t>;

template <typename _Scalar>
using Rotation2D  = Eigen::Rotation2D<_Scalar>;
using Rotation2Dr = Eigen::Rotation2D<real_t>;

inline namespace Literals {

constexpr index_t operator"" _idx(const unsigned long long index) {
    return static_cast<index_t>(index);
}

constexpr time_step_t operator"" _step(const unsigned long long timeStep) {
    return static_cast<time_step_t>(timeStep);
}

}  // namespace Literals

template <typename _Scalar,
          typename _StorageIndex = typename Eigen::SparseMatrix<_Scalar>::StorageIndex>
class MutableTriplet final : public Eigen::Triplet<_Scalar, _StorageIndex> {
  public:
    using Eigen::Triplet<_Scalar, _StorageIndex>::Triplet;

    _StorageIndex& row() {
        return m_row;
    }

    _StorageIndex& col() {
        return m_col;
    }

    _Scalar& value() {
        return m_value;
    }

  private:
    using BaseType = Eigen::Triplet<_Scalar, _StorageIndex>;
    using BaseType::m_col;
    using BaseType::m_row;
    using BaseType::m_value;
};

template <bool VALUE, typename... _Args>
inline constexpr bool dependent_bool_value = VALUE;

template <typename... _Args>
inline constexpr bool dependent_false = dependent_bool_value<false, _Args...>;

template <typename _T, typename... _Args>
using dependent_type = _T;

template <index_t VALUE>
using idx_ = hana::integral_constant<index_t, VALUE>;
template <index_t VALUE>
inline constexpr idx_<VALUE> idx_c{};

constexpr auto enumerate(const index_t n) {
    return nano::views::iota(0_idx, n);
}

template <typename _T>
constexpr auto cast_to = nano::views::transform([](auto&& el) -> decltype(auto) {
    return static_cast<_T>(std::forward<decltype(el)>(el));
});

template <typename _T, bool _B>
using add_const_if = std::conditional<_B, std::add_const_t<_T>, _T>;
template <typename _T, bool _B>
using add_const_if_t = std::conditional_t<_B, std::add_const_t<_T>, _T>;

template <typename _T>
using is_const_ref =
    std::conjunction<std::is_reference<_T>, std::is_const<std::remove_reference_t<_T>>>;
template <typename _T>
constexpr bool is_const_ref_v = is_const_ref<_T>::value;

namespace Internal {

template <class T>
constexpr T& ReferenceWrapperHelper(T& t) noexcept {
    return t;
}
template <class T>
void ReferenceWrapperHelper(T&&) = delete;

}  // namespace Internal

template <typename _T>
class reference_wrapper {
  public:
    using type = _T;

    // Construct/copy/destroy.
    template <
        class U,
        class = decltype(Internal::ReferenceWrapperHelper<_T>(std::declval<U>()),
                         std::enable_if_t<!std::is_same_v<reference_wrapper, remove_cvref_t<U>>>())>
    constexpr reference_wrapper(U&& u) noexcept(
        noexcept(Internal::ReferenceWrapperHelper<_T>(std::forward<U>(u))))
        : _ptr(std::addressof(Internal::ReferenceWrapperHelper<_T>(std::forward<U>(u)))) {
    }

    reference_wrapper(const reference_wrapper&) noexcept = default;

    // Assignment.
    reference_wrapper& operator=(const reference_wrapper& x) noexcept = default;

    // Access.
    constexpr operator _T&() const noexcept {
        return *_ptr;
    }
    constexpr _T& get() const noexcept {
        return *_ptr;
    }

    template <class... ArgTypes>
    constexpr std::invoke_result_t<_T&, ArgTypes...> operator()(ArgTypes&&... args) const
        noexcept(std::is_nothrow_invocable_v<_T&, ArgTypes...>) {
        return std::invoke(get(), std::forward<ArgTypes>(args)...);
    }

  private:
    _T* _ptr;
};

// Deduction guides.
template <typename _T>
reference_wrapper(_T&) -> reference_wrapper<_T>;

/// Take reference to a variable.
template <typename _T>
constexpr inline reference_wrapper<_T> ref(_T& t) noexcept {
    return reference_wrapper<_T>(t);
}

/// Take const reference to a variable.
template <typename _T>
constexpr inline reference_wrapper<const _T> cref(const _T& t) noexcept {
    return reference_wrapper<const _T>(t);
}

template <typename _T>
void ref(const _T&&) = delete;

template <typename _T>
void cref(const _T&&) = delete;

/// Overload to prevent wrapping a reference_wrapper
template <typename _T>
constexpr inline reference_wrapper<_T> ref(reference_wrapper<_T> t) noexcept {
    return t;
}

/// Overload to prevent wrapping a reference_wrapper
template <typename _T>
constexpr inline reference_wrapper<const _T> cref(reference_wrapper<_T> t) noexcept {
    return {t.get()};
}

}  // namespace Ungar

#endif /* _UNGAR__DATA_TYPES_HPP_ */
