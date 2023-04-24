/******************************************************************************
 *
 * @file ungar/utils/utils.hpp
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

#ifndef _UNGAR__UTILS__UTILS_HPP_
#define _UNGAR__UTILS__UTILS_HPP_

#include <filesystem>
#include <locale>
#include <random>

#include "ungar/assert.hpp"
#include "ungar/data_types.hpp"

#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
#include "ungar/autodiff/data_types.hpp"
#endif

namespace Ungar {
namespace Concepts {

template <typename _Scalar>
concept Scalar =
#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
    std::convertible_to<_Scalar, real_t> || std::convertible_to<_Scalar, ad_scalar_t>;
#else
    std::convertible_to<_Scalar, real_t>;
#endif

}  // namespace Concepts

[[noreturn]] inline void Unreachable() {
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#ifdef __GNUC__  // GCC, Clang, ICC
    __builtin_unreachable();
#elif defined _MSC_VER  // MSVC
    __assume(false);
#endif
}

namespace Utils {

/**
 * @brief Convert a real matrix into a dynamic array of triplets.
 *
 * @tparam _CLEAR_TRIPLETS   If true, the output dynamic array of triplets is
 *                           cleared before the input matrix gets converted.
 * @param[in] matrix         Input matrix to be converted into triplets.
 * @param[out] triplets      Dynamic array of triplets generated from the
 *                           input matrix.
 * @param[in] rowOffset      Offset applied to all the row indices of the
 *                           triplets generated from the input matrix.
 * @param[in] colOffset      Offset applied to all the column indices of
 *                           the triplets generated from the input matrix.
 */
template <bool _CLEAR_TRIPLETS = false>
void MatrixToTriplets(const RefToConstMatrixXr& matrix,
                      std::vector<MutableTriplet<real_t>>& triplets,
                      const index_t rowOffset = 0_idx,
                      const index_t colOffset = 0_idx) {
    if constexpr (_CLEAR_TRIPLETS) {
        triplets.clear();
    }

    for (index_t j = 0_idx; j < matrix.cols(); ++j) {
        for (index_t i = 0_idx; i < matrix.rows(); ++i) {
            triplets.emplace_back(i + rowOffset, j + colOffset, matrix(i, j));
        }
    }
}

/**
 * @brief Convert a real sparse matrix into a dynamic array of triplets.
 *
 * @tparam _LOWER_TRIANGULAR If true, only the lower triangular part
 *                           of the input matrix gets converted.
 * @tparam _CLEAR_TRIPLETS   If true, the output dynamic array of triplets is
 *                           cleared before the input matrix gets converted.
 * @param[in] sparseMatrix   Input matrix to be converted into triplets.
 * @param[out] triplets      Dynamic array of triplets generated from the
 *                           input matrix.
 * @param[in] rowOffset      Offset applied to all the row indices of the
 *                           triplets generated from the input matrix.
 * @param[in] colOffset      Offset applied to all the column indices of
 *                           the triplets generated from the input matrix.
 */
template <bool _LOWER_TRIANGULAR = false, bool _CLEAR_TRIPLETS = false>
void SparseMatrixToTriplets(const SparseMatrix<real_t>& sparseMatrix,
                            std::vector<MutableTriplet<real_t>>& triplets,
                            const index_t rowOffset = 0_idx,
                            const index_t colOffset = 0_idx) {
    if constexpr (_CLEAR_TRIPLETS) {
        triplets.clear();
    }

    for (int i = 0; i < sparseMatrix.outerSize(); ++i) {
        for (SparseMatrix<real_t>::InnerIterator it(sparseMatrix, i); it; ++it) {
            if constexpr (_LOWER_TRIANGULAR) {
                if (it.row() < it.col()) {
                    continue;
                }
            }
            triplets.emplace_back(it.row() + rowOffset, it.col() + colOffset, it.value());
        }
    }
}

namespace Internal {
namespace Concepts {

// clang-format off
template <typename _SparseMatrix>
concept HasNonZeros = requires (const _SparseMatrix sparseMatrix) {
    sparseMatrix.nonZeros();
};
// clang-format on

}  // namespace Concepts
}  // namespace Internal

namespace Internal {

template <typename... _SparseMatrices>
class VerticallyStackSparseMatricesImpl {
  public:
    using ScalarType = std::common_type_t<typename std::remove_cvref_t<_SparseMatrices>::Scalar...>;

    VerticallyStackSparseMatricesImpl(_SparseMatrices&&... sparseMatrices)
        : _sparseMatrices{std::forward<_SparseMatrices>(sparseMatrices)...} {
    }

    template <typename _StackedSparseMatrix>
    void In(Eigen::SparseMatrixBase<_StackedSparseMatrix> const& stackedSparseMatrix) {
        hana::unpack(_sparseMatrices, [&](auto&&... sparseMatrices) {
            In(std::forward<_SparseMatrices>(sparseMatrices)...,
               stackedSparseMatrix.const_cast_derived());
        });
    }

    SparseMatrix<ScalarType> ToSparse() {
        return hana::unpack(_sparseMatrices, [&](auto&&... sparseMatrices) {
            return ToSparse(std::forward<_SparseMatrices>(sparseMatrices)...);
        });
    }

  private:
    template <typename _StackedSparseMatrix>
    void In(const Eigen::SparseMatrixBase<std::remove_cvref_t<_SparseMatrices>>&... sparseMatrices,
            Eigen::SparseMatrixBase<_StackedSparseMatrix> const& stackedSparseMatrix) {
        UNGAR_ASSERT([](const auto& m0, const auto&... ms) {
            return ((m0.cols() == ms.cols()) && ...);
        }(sparseMatrices...));

        const index_t cols = [](const auto& m0, const auto&...) {
            return m0.cols();
        }(sparseMatrices...);

        std::vector<MutableTriplet<ScalarType>> triplets;
        if constexpr ((Internal::Concepts::HasNonZeros<_SparseMatrices> && ...)) {
            triplets.reserve((sparseMatrices.derived().nonZeros() + ...));
        }

        index_t rows = 0_idx;
        ((SparseMatrixToTriplets(sparseMatrices, triplets, rows), rows += sparseMatrices.rows()),
         ...);

        stackedSparseMatrix.const_cast_derived().resize(rows, cols);
        stackedSparseMatrix.const_cast_derived().setFromTriplets(triplets.begin(), triplets.end());
    }

    SparseMatrix<ScalarType> ToSparse(
        const Eigen::SparseMatrixBase<std::remove_cvref_t<_SparseMatrices>>&... sparseMatrices) {
        UNGAR_ASSERT([](const auto& m0, const auto&... ms) {
            return ((m0.cols() == ms.cols()) && ...);
        }(sparseMatrices...));

        const index_t cols = [](const auto& m0, const auto&...) {
            return m0.cols();
        }(sparseMatrices...);

        std::vector<MutableTriplet<ScalarType>> triplets;
        if constexpr ((Internal::Concepts::HasNonZeros<_SparseMatrices> && ...)) {
            triplets.reserve((sparseMatrices.derived().nonZeros() + ...));
        }

        index_t rows = 0_idx;
        ((SparseMatrixToTriplets(sparseMatrices, triplets, rows), rows += sparseMatrices.rows()),
         ...);

        SparseMatrix<ScalarType> stackedSparseMatrix{rows, cols};
        stackedSparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
        return stackedSparseMatrix;
    }

  public:
    std::tuple<_SparseMatrices&&...> _sparseMatrices;
};

}  // namespace Internal

template <typename... _SparseMatrices>  // clang-format off
requires (sizeof...(_SparseMatrices) > 0UL)
auto VerticallyStackSparseMatrices(_SparseMatrices&&... sparseMatrices) {  // clang-format on
    return Internal::VerticallyStackSparseMatricesImpl<_SparseMatrices...>{
        std::forward<_SparseMatrices>(sparseMatrices)...};
}

inline constexpr int Q    = VectorXr::MaxRowsAtCompileTime;
inline constexpr auto Q_c = hana::llong_c<static_cast<long long>(Q)>;

template <typename _Derived, bool _COND>
inline constexpr decltype(auto) EigenConstCastIf(const Eigen::MatrixBase<_Derived>& vector,
                                                 hana::bool_<_COND>) {
    if constexpr (_COND) {
        return const_cast<Eigen::MatrixBase<_Derived>&>(vector);
    } else {
        return (vector);
    }
}

namespace Internal {

template <int _SIZE>
inline constexpr auto DecompositionElementSize() {
    if constexpr (_SIZE == Q) {
        return 4;
    } else {
        return _SIZE;
    }
}

template <typename _Scalar, int _SIZE, bool _IS_CONST>
inline constexpr auto DecompositionElementTypeHelper() {
    if constexpr (_SIZE == 1) {
        return hana::type_c<add_const_if_t<_Scalar, _IS_CONST>&>;
    } else if constexpr (_SIZE == Q) {
        return hana::type_c<Eigen::Map<add_const_if_t<Quaternion<_Scalar>, _IS_CONST>>>;
    } else {
        return hana::type_c<Eigen::Map<add_const_if_t<Vector<_Scalar, _SIZE>, _IS_CONST>>>;
    }
}

template <typename _Scalar, int _SIZE, bool _IS_CONST>
using DecompositionElementType =
    typename decltype(+DecompositionElementTypeHelper<_Scalar, _SIZE, _IS_CONST>())::type;

template <int _INDEX, int _SIZE, bool _IS_CONST, typename _DecomposableVector>
inline constexpr decltype(auto) DecomposeImpl(
    Eigen::MatrixBase<_DecomposableVector> const& decomposableVector) {
    using ScalarType  = typename _DecomposableVector::Scalar;
    using ElementType = DecompositionElementType<ScalarType, _SIZE, _IS_CONST>;

    decltype(auto) vec = EigenConstCastIf(decomposableVector, hana::not_(hana::bool_c<_IS_CONST>));
    if constexpr (_SIZE == 1) {
        static_assert(std::same_as<ElementType, decltype(vec.derived().coeffRef(_INDEX))>);
        return vec.derived().coeffRef(_INDEX);
    } else {
        static_assert(
            std::same_as<add_const_if_t<ScalarType, _IS_CONST>*, decltype(vec.derived().data())>);
        return ElementType{vec.derived().data() + _INDEX};
    }
}

template <int... _SIZES, bool _IS_CONST, typename _DecomposableVector>  // clang-format off
requires (_DecomposableVector::RowsAtCompileTime == Eigen::Dynamic || 
          _DecomposableVector::RowsAtCompileTime == (DecompositionElementSize<_SIZES>() + ...))
inline auto Decompose(
        Eigen::MatrixBase<_DecomposableVector> const& decomposableVector,
        hana::bool_<_IS_CONST>) {  // clang-format on
    if constexpr (_DecomposableVector::RowsAtCompileTime == Eigen::Dynamic) {
        UNGAR_ASSERT(decomposableVector.size() == (DecompositionElementSize<_SIZES>() + ...));
    }

    using ScalarType = typename _DecomposableVector::Scalar;
    using ReturnType = std::tuple<DecompositionElementType<ScalarType, _SIZES, _IS_CONST>...>;

    constexpr auto SIZE_TUPLE = hana::make_tuple(hana::int_c<_SIZES>...);
    constexpr auto INDEX_TUPLE =
        hana::drop_back(hana::scan_left(SIZE_TUPLE, hana::int_c<0>, [](auto index, auto size) {
            return index + hana::int_c<DecompositionElementSize<size>()>;
        }));

    return hana::unpack(std::make_index_sequence<sizeof...(_SIZES)>(), [&](auto... is) {
        return ReturnType{
            DecomposeImpl<INDEX_TUPLE[is], SIZE_TUPLE[is], _IS_CONST>(decomposableVector)...};
    });
}

namespace Concepts {

template <typename _ComposableVector>
concept ComposableVector = Ungar::Concepts::DenseMatrixExpression<_ComposableVector> ||
                           Ungar::Concepts::Scalar<std::remove_cvref_t<_ComposableVector>>;

}  // namespace Concepts

template <Concepts::ComposableVector _ComposableVector>
struct ComposableVectorTraits {
    static constexpr index_t SIZE = std::remove_cvref_t<_ComposableVector>::RowsAtCompileTime;
    static constexpr bool SCALAR  = false;
    using ScalarType              = typename std::remove_cvref_t<_ComposableVector>::Scalar;
};

template <Concepts::ComposableVector _ComposableVector>  // clang-format off
requires (!Ungar::Concepts::DenseMatrixExpression<_ComposableVector>)
struct ComposableVectorTraits<_ComposableVector> {  // clang-format on
    static constexpr index_t SIZE = 1_idx;
    static constexpr bool SCALAR  = true;
    using ScalarType              = std::remove_cvref_t<_ComposableVector>;
};

index_t ComposableVectorSize(Concepts::ComposableVector auto&& composableVector) {
    if constexpr (Ungar::Concepts::DenseMatrixExpression<decltype(composableVector)>) {
        return composableVector.size();
    } else {
        return 1_idx;
    }
}

template <Concepts::ComposableVector... _ComposableVectors>
class ComposeImpl {
  public:
    static constexpr bool ALL_FIXED_SIZES =
        ((ComposableVectorTraits<_ComposableVectors>::SIZE != Eigen::Dynamic) && ...);
    static constexpr auto COMPOSED_VECTOR_SIZE =
        ALL_FIXED_SIZES ? (ComposableVectorTraits<_ComposableVectors>::SIZE + ...) : -1_idx;

    using ScalarType =
        std::common_type_t<typename ComposableVectorTraits<_ComposableVectors>::ScalarType...>;

    template <typename... _Args>
    ComposeImpl(_Args&&... args) : _composableVectors{std::forward<_Args>(args)...} {
    }

    template <typename _ComposedVector>  // clang-format off
    requires (!ALL_FIXED_SIZES)
    void In(Eigen::MatrixBase<_ComposedVector> const& composedVector) {  // clang-format on
        const index_t composedVectorSize = hana::unpack(
            _composableVectors, [](auto&&... vs) { return (ComposableVectorSize(vs) + ...); });

        if constexpr (ComposableVectorTraits<_ComposedVector>::SIZE != Eigen::Dynamic) {
            UNGAR_ASSERT(composedVector.size() == composedVectorSize);
        } else {
            composedVector.const_cast_derived().resize(composedVectorSize);
        }
        const auto sizeTuple  = hana::unpack(_composableVectors, [](auto&&... vs) {
            return hana::make_tuple(ComposableVectorSize(vs)...);
        });
        const auto indexTuple = hana::drop_back(
            hana::scan_left(sizeTuple, 0_idx, [](auto index, auto size) { return index + size; }));

        std::apply(  // clang-format off
            [&] (auto&&... args) {  // clang-format on
                return hana::unpack(
                    std::make_index_sequence<sizeof...(_ComposableVectors)>(),
                    [&](const auto... is) constexpr {
                        auto impl = hana::overload(
                            [&](auto i, auto&& scalar) requires Ungar::Concepts::Scalar<
                                std::remove_cvref_t<decltype(scalar)>> {
                                composedVector.const_cast_derived()[indexTuple[i]] = scalar;
                            },
                            [&](auto i, auto&& subvector) {
                                composedVector.const_cast_derived().segment(
                                    indexTuple[i], sizeTuple[i]) = subvector;
                            });

                        (impl(is, args), ...);
                    });

            },
            _composableVectors);
    }

    template <typename _ComposedVector>  // clang-format off
    requires ALL_FIXED_SIZES &&
             (ComposableVectorTraits<_ComposedVector>::SIZE == Eigen::Dynamic || 
              ComposableVectorTraits<_ComposedVector>::SIZE == COMPOSED_VECTOR_SIZE)
    void In(Eigen::MatrixBase<_ComposedVector> const& composedVector) {  // clang-format on
        if constexpr (ComposableVectorTraits<_ComposedVector>::SIZE == Eigen::Dynamic) {
            UNGAR_ASSERT(composedVector.size() == COMPOSED_VECTOR_SIZE);
        }
        static constexpr auto _SIZE_TUPLE =
            hana::make_tuple(hana::int_c<ComposableVectorTraits<_ComposableVectors>::SIZE>...);
        static constexpr auto _INDEX_TUPLE = hana::drop_back(hana::scan_left(
            _SIZE_TUPLE, hana::int_c<0>, [](auto index, auto size) { return index + size; }));

        std::apply(  // clang-format off
            [&]<typename... _Ts>(_Ts&&... args) {  // clang-format on
                return hana::unpack(
                    std::make_index_sequence<sizeof...(_ComposableVectors)>(),
                    [&](const auto... is) constexpr {
                        auto lazyScalar    = hana::make_lazy([&](auto i, auto&& scalar) {
                            composedVector.const_cast_derived()[_INDEX_TUPLE[i]] = scalar;
                        });
                        auto lazySubvector = hana::make_lazy([&](auto i, auto&& subvector) {
                            composedVector.const_cast_derived().template segment<_SIZE_TUPLE[i]>(
                                _INDEX_TUPLE[i]) = subvector;
                        });

                        (hana::eval(hana::if_(_SIZE_TUPLE[is] == hana::int_c<1>,
                                              lazyScalar(is, args),
                                              lazySubvector(is, args))),
                         ...);
                    });

            },
            _composableVectors);
    }

    auto ToDynamic() {
        VectorX<ScalarType> composedVector;
        In(composedVector);
        return composedVector;
    }

    auto ToDynamic() requires ALL_FIXED_SIZES {
        VectorX<ScalarType> composedVector{COMPOSED_VECTOR_SIZE};
        In(composedVector);
        return composedVector;
    }

    auto ToFixed()  // clang-format off
    requires ALL_FIXED_SIZES
    {  // clang-format on
        Vector<ScalarType, COMPOSED_VECTOR_SIZE> composedVector;
        In(composedVector);
        return composedVector;
    }

  private:
    std::tuple<_ComposableVectors&&...> _composableVectors;
};

}  // namespace Internal

template <int... SIZES>
inline auto Decompose(auto&& vector) {
    static_assert(dependent_false<decltype(vector)>,
                  "This function is not implemented for the given input type; consider wrapping it "
                  "into an Eigen::Ref (or Eigen::Map) object.");
}

template <int... SIZES, typename _Vector>  // clang-format off
requires std::same_as<_Vector, const typename std::remove_cvref_t<_Vector>::PlainMatrix&> ||
         Concepts::EigenConstTypeWrapper<std::remove_cvref_t<_Vector>>
inline auto Decompose(_Vector&& vector) {  // clang-format on
    return Internal::Decompose<SIZES...>(std::forward<_Vector>(vector), hana::true_c);
}

template <int... SIZES, typename _Vector>  // clang-format off
requires std::same_as<_Vector, typename std::remove_cvref_t<_Vector>::PlainMatrix&> ||
         Concepts::EigenMutableTypeWrapper<std::remove_cvref_t<_Vector>>
inline auto Decompose(_Vector&& vector) {  // clang-format on
    return Internal::Decompose<SIZES...>(std::forward<_Vector>(vector), hana::false_c);
}

template <Internal::Concepts::ComposableVector... _ComposableVectors>
inline Internal::ComposeImpl<_ComposableVectors&&...> Compose(
    _ComposableVectors&&... composableVectors) {
    return {std::forward<_ComposableVectors>(composableVectors)...};
}

template <Concepts::Scalar _Scalar, typename _ComposedVector>
inline auto ComposeIn(const std::vector<RefToConstVectorX<_Scalar>>& vectors,
                      Eigen::MatrixBase<_ComposedVector> const& composedVector)  // clang-format off
requires std::same_as<typename _ComposedVector::Scalar, _Scalar> {  // clang-format on
    for (size_t i = 0UL; const auto& v : vectors) {
        composedVector.const_cast_derived().segment(i, v.size()) = v;
        i += v.size();
    }
}

inline std::string ToSnakeCase(std::string_view input) {
    std::string output{};
    for (size_t i = 1UL; const char c : input) {
        if (std::isdigit(c, std::locale())) {
            output.append(1UL, c);
        } else if (std::isalpha(c, std::locale())) {
            if ((i < input.size() && std::islower(c, std::locale()) &&
                 std::isupper(input[i], std::locale())) ||
                (i + 1UL < input.size() && std::isupper(c, std::locale()) &&
                 std::isupper(input[i], std::locale()) &&
                 std::islower(input[i + 1UL], std::locale()))) {
                output.append(static_cast<char>(std::tolower(c)) + "_"s);
            } else {
                output.append(1UL, std::tolower(c));
            }
        } else {
            output.append(1UL, '_');
        }

        ++i;
    }

    return output;
}

namespace Internal {

template <typename _Scalar>
struct MutableVectorSegmenter {
  public:
    template <typename _Vector>
    MutableVectorSegmenter(Eigen::MatrixBase<_Vector> const& vector)
        : _end{vector.const_cast_derived().data()} {
    }

    MutableVectorSegmenter(_Scalar* const begin) : _end{begin} {
    }

    _Scalar& Next() {
        _Scalar* const it = _end;
        _end              = std::ranges::next(_end);
        return *it;
    }

    auto Next(const std::integral auto size) {
        _Scalar* const begin = _end;
        _end                 = std::ranges::next(_end, size);
        return Eigen::Map<VectorX<_Scalar>>{begin, static_cast<index_t>(size)};
    }

    MutableVectorSegmenter& Skip(const std::integral auto size) {
        _end = std::ranges::next(_end, size);
        return *this;
    }

    auto End() const {
        return _end;
    }

  private:
    _Scalar* _end;
};

template <class _Vector>
MutableVectorSegmenter(Eigen::MatrixBase<_Vector> const&)
    -> MutableVectorSegmenter<typename _Vector::Scalar>;

template <typename _Scalar>
struct ConstVectorSegmenter {
  public:
    template <typename _Vector>
    ConstVectorSegmenter(const Eigen::MatrixBase<_Vector>& vector) : _end{vector.derived().data()} {
    }

    ConstVectorSegmenter(const _Scalar* const begin) : _end{begin} {
    }

    const _Scalar& Next() {
        const _Scalar* const it = _end;
        _end                    = std::ranges::next(_end);
        return *it;
    }

    auto Next(const std::integral auto size) {
        const _Scalar* const begin = _end;
        _end                       = std::ranges::next(_end, size);
        return Eigen::Map<const VectorX<_Scalar>>{begin, static_cast<index_t>(size)};
    }

    ConstVectorSegmenter& Skip(const std::integral auto size) {
        _end = std::ranges::next(_end, size);
        return *this;
    }

    auto End() const {
        return _end;
    }

  private:
    const _Scalar* _end;
};

template <class _Vector>
ConstVectorSegmenter(const Eigen::MatrixBase<_Vector>&)
    -> ConstVectorSegmenter<typename _Vector::Scalar>;

}  // namespace Internal

template <typename _Vector>
inline auto VectorSegmenter(Eigen::MatrixBase<_Vector> const& vector) {
    static_assert(dependent_false<_Vector>,
                  "This function is not implemented for the given input type; consider wrapping it "
                  "into an Eigen::Ref (or Eigen::Map) object.");
}

template <typename _Vector>  // clang-format off
requires std::same_as<_Vector, const typename std::remove_cvref_t<_Vector>::PlainMatrix&> ||
         Concepts::EigenConstTypeWrapper<std::remove_cvref_t<_Vector>>
inline auto VectorSegmenter(_Vector&& vector) {  // clang-format on
    return Internal::ConstVectorSegmenter{std::forward<_Vector>(vector)};
}

template <typename _Scalar>
inline Internal::ConstVectorSegmenter<_Scalar> VectorSegmenter(const _Scalar* const begin) {
    return {begin};
}

template <typename _Vector>  // clang-format off
requires std::same_as<_Vector, typename std::remove_cvref_t<_Vector>::PlainMatrix&> ||
         Concepts::EigenMutableTypeWrapper<std::remove_cvref_t<_Vector>>
inline auto VectorSegmenter(_Vector&& vector) {  // clang-format on
    return Internal::MutableVectorSegmenter{std::forward<_Vector>(vector)};
}

template <typename _Scalar>
inline Internal::MutableVectorSegmenter<_Scalar> VectorSegmenter(_Scalar* const begin) {
    return {begin};
}

inline std::filesystem::path TemporaryDirectoryPath(const std::string& directoryNamePrefix = ""s,
                                                    const size_t maxAttempts = 1024UL) {
    const auto tmp = std::filesystem::temp_directory_path();

    std::random_device r;
    std::mt19937 randomEngine{r()};
    std::uniform_int_distribution<uint64_t> uniformDistribution{0};

    size_t i = 1UL;
    std::filesystem::path temporaryDirectoryPath;
    while (true) {
        if (i > maxAttempts) {
            throw std::runtime_error("Could not generate nonexistent temporary directory path.");
        }

        std::stringstream ss;
        ss << std::hex << uniformDistribution(randomEngine);
        temporaryDirectoryPath = tmp / (directoryNamePrefix + ss.str());

        if (!std::filesystem::is_directory(temporaryDirectoryPath)) {
            break;
        }

        ++i;
    }

    return temporaryDirectoryPath;
}

template <typename _Derived>
inline Vector3<typename _Derived::Scalar> EstimateLinearVelocity(
    const Eigen::MatrixBase<_Derived>& position,
    const Eigen::MatrixBase<_Derived>& previousPosition,
    const real_t stepSize) {
    return (position - previousPosition) / stepSize;
}

template <Concepts::Scalar _Scalar = real_t>
inline Vector3<_Scalar> EstimateBodyFrameAngularVelocity(
    const Quaternion<_Scalar>& orientation,
    const Quaternion<_Scalar>& previousOrientation,
    const real_t stepSize) {
    return 2.0 * (previousOrientation.conjugate() * orientation).vec() / stepSize;
}

template <Concepts::Scalar _Scalar = real_t>
inline Vector3<_Scalar> EstimateInertialFrameAngularVelocity(
    const Quaternion<_Scalar>& orientation,
    const Quaternion<_Scalar>& previousOrientation,
    const real_t stepSize) {
    return 2.0 * (orientation * previousOrientation.conjugate()).vec() / stepSize;
}

template <typename _Vector>
inline Quaternion<typename _Vector::Scalar> ExponentialMap(const Eigen::MatrixBase<_Vector>& v) {
    static_assert(_Vector::RowsAtCompileTime == 3);
    using ScalarType = typename _Vector::Scalar;
    Quaternion<ScalarType> q;

    if constexpr (std::convertible_to<ScalarType, real_t>) {
        const ScalarType vNorm = v.norm();
        q.vec()                = vNorm > 0.0 ? Vector3<ScalarType>{v * sin(0.5 * vNorm) / vNorm}
                              : Vector3<ScalarType>{v * 0.5};
        q.w() = vNorm > 0.0 ? cos(0.5 * vNorm) : 1.0;
    }
#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
    else if constexpr (std::convertible_to<ScalarType, ad_scalar_t>) {
        static_assert(dependent_false<ScalarType>,
                      "The 'ExponentialMap' function is not implemented for vectors with scalar "
                      "type 'ad_scalar_t'. Use 'ApproximateExponentialMap' instead.");
        /**
         * @todo Implement non-approximate version of 'ExponentialMap' for vectors with scalar
         *       type 'ad_scalar_t'.
         */
        // Here is an example implementation that does not work:
        // const ad_scalar_t vNorm = v.norm();
        // const ad_scalar_t vNormIsNotZero =
        //     CppAD::CondExpGt(vNorm, ad_scalar_t{0.0}, ad_scalar_t{1}, ad_scalar_t{0});
        // const ad_scalar_t vNormIsZero = 1.0 - vNormIsNotZero;
        // q.vec() = v * (CppAD::azmul(vNormIsNotZero, sin(0.5 * vNorm) / vNorm) +
        //                CppAD::azmul(vNormIsZero, ad_scalar_t{0.5}));
        // q.w()   = CppAD::azmul(vNormIsNotZero, cos(0.5 * vNorm)) +
        //         CppAD::azmul(vNormIsZero, ad_scalar_t{1.0});
    }
#endif
    else {
        static_assert(dependent_false<ScalarType>,
                      "The 'ExponentialMap' function is not implemented for vectors with the given "
                      "scalar type.");
    }
    return q;
}

template <typename _Vector>
inline Quaternion<typename _Vector::Scalar> ApproximateExponentialMap(
    const Eigen::MatrixBase<_Vector>& v) {
    static_assert(_Vector::RowsAtCompileTime == 3);
    using ScalarType = typename _Vector::Scalar;
    Quaternion<ScalarType> q;

    ScalarType vApproximateNorm =
        Eigen::numext::sqrt(v.squaredNorm() + Eigen::NumTraits<ScalarType>::epsilon());
    q.vec() = v * sin(0.5 * vApproximateNorm) / vApproximateNorm;
    q.w()   = cos(0.5 * vApproximateNorm);
    return q;
}

template <Concepts::Scalar _Scalar>
inline Quaternion<_Scalar> UnitQuaternionInverse(const Quaternion<_Scalar>& q) {
    return q.conjugate();
}

template <Concepts::Scalar _Scalar>
inline Quaternion<_Scalar> UnitQuaternionInverse(const Eigen::Map<const Quaternion<_Scalar>>& q) {
    return q.conjugate();
}

template <Concepts::Scalar _Scalar>
inline Quaternion<_Scalar> UnitQuaternionInverse(const Eigen::Map<Quaternion<_Scalar>>& q) {
    return q.conjugate();
}

template <Concepts::Scalar _Base, typename _Exponent>
inline auto Pow(const _Base base, const _Exponent exponent) {
    if constexpr (std::convertible_to<_Base, real_t>) {
        return std::pow(base, exponent);
    }
#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
    else if constexpr (std::convertible_to<_Base, ad_scalar_t>) {
        if constexpr (std::integral<_Exponent>) {
            return CppAD::pow(base, static_cast<int>(exponent));
        } else {
            return CppAD::pow(base, exponent);
        }
    }
#endif
    else {
        Unreachable();
    }
}

inline constexpr std::string_view DASH_LINE_SEPARATOR =
    "----------------------------------------------------------------"sv;
inline constexpr std::string_view STAR_LINE_SEPARATOR =
    "****************************************************************"sv;

template <typename _Matrix>
inline bool HasNaN(const Eigen::MatrixBase<_Matrix>& matrix) {
    return matrix.hasNaN();
}

template <typename _SparseMatrix>
inline bool HasNaN(const Eigen::SparseMatrixBase<_SparseMatrix>& sparseMatrix) {
    for (int k = 0; k < sparseMatrix.outerSize(); ++k) {
        for (typename _SparseMatrix::InnerIterator it{sparseMatrix.derived(), k}; it; ++it) {
            if (!(it.value() == it.value())) {
                return true;
            }
        }
    }
    return false;
}

inline auto ElementaryXRotationMatrix(const Concepts::Scalar auto angle) {
    using ScalarType   = std::remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle);
    const ScalarType s = sin(angle);
    // clang-format off
    return Matrix3<ScalarType>{{ScalarType{1.0},  ScalarType{0.0},  ScalarType{0.0}},
                               {ScalarType{0.0},  c,                -s},
                               {ScalarType{0.0},  s,                c}};  // clang-format on
}

inline auto ElementaryYRotationMatrix(const Concepts::Scalar auto angle) {
    using ScalarType   = std::remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle);
    const ScalarType s = sin(angle);
    // clang-format off
    return Matrix3<ScalarType>{{c,                ScalarType{0.0},  s},
                               {ScalarType{0.0},  ScalarType{1.0},  ScalarType{0.0}},
                               {-s,               ScalarType{0.0},  c}};  // clang-format on
}

inline auto ElementaryZRotationMatrix(const Concepts::Scalar auto angle) {
    using ScalarType   = std::remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle);
    const ScalarType s = sin(angle);
    // clang-format off
    return Matrix3<ScalarType>{{c,                -s,               ScalarType{0.0}},
                               {s,                c,                ScalarType{0.0}},
                               {ScalarType{0.0},  ScalarType{0.0},  ScalarType{1.0}}};  // clang-format on
}

inline auto ElementaryXQuaternion(const Concepts::Scalar auto angle) {
    using ScalarType   = std::remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle / ScalarType{2.0});
    const ScalarType s = sin(angle / ScalarType{2.0});
    return Quaternion<ScalarType>{c, s, ScalarType{0.0}, ScalarType{0.0}};
}

inline auto ElementaryYQuaternion(const Concepts::Scalar auto angle) {
    using ScalarType   = std::remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle / ScalarType{2.0});
    const ScalarType s = sin(angle / ScalarType{2.0});
    return Quaternion<ScalarType>{c, ScalarType{0.0}, s, ScalarType{0.0}};
}

inline auto ElementaryZQuaternion(const Concepts::Scalar auto angle) {
    using ScalarType   = std::remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle / ScalarType{2.0});
    const ScalarType s = sin(angle / ScalarType{2.0});
    return Quaternion<ScalarType>{c, ScalarType{0.0}, ScalarType{0.0}, s};
}

/// @brief A yaw-pitch-roll angles vector 'ypr' is defined so that
///        the yaw, pitch and roll angles correspond to 'ypr.z()',
///        'ypr.y()' and 'ypr.x()', respectively.
template <typename _Vector>  // clang-format off
requires (_Vector::RowsAtCompileTime == 3)
inline Matrix3<typename _Vector::Scalar> RotationMatrixFromYawPitchRoll(const Eigen::MatrixBase<_Vector>& ypr) {  // clang-format on
    return {ElementaryZRotationMatrix(ypr.z()) * ElementaryYRotationMatrix(ypr.y()) *
            ElementaryXRotationMatrix(ypr.x())};
}

/// @brief A yaw-pitch-roll angles vector 'ypr' is defined so that
///        the yaw, pitch and roll angles correspond to 'ypr.z()',
///        'ypr.y()' and 'ypr.x()', respectively.
template <typename _Vector>  // clang-format off
requires (_Vector::RowsAtCompileTime == 3)
inline Quaternion<typename _Vector::Scalar> QuaternionFromYawPitchRoll(const Eigen::MatrixBase<_Vector>& ypr) {  // clang-format on
    return {ElementaryZQuaternion(ypr.z()) * ElementaryYQuaternion(ypr.y()) *
            ElementaryXQuaternion(ypr.x())};
}

template <typename _Quaternion>
inline Vector3<typename _Quaternion::Scalar> QuaternionToYawPitchRoll(
    const Eigen::QuaternionBase<_Quaternion>& q) {
    return q.toRotationMatrix().eulerAngles(2_idx, 1_idx, 0_idx).reverse();
}

#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
template <typename _Quaternion>  // clang-format off
requires std::convertible_to<typename _Quaternion::Scalar, ad_scalar_t>
inline Vector3<typename _Quaternion::Scalar> QuaternionToYawPitchRoll(
    const Eigen::QuaternionBase<_Quaternion>& q) {  // clang-format on
    const Matrix3<typename _Quaternion::Scalar> R = q.toRotationMatrix();
    return {
        CppAD::atan2(R(2_idx, 1_idx), R(2_idx, 2_idx)),
        CppAD::atan2(-R(2_idx, 0_idx),
                     CppAD::sqrt(Utils::Pow(R(2_idx, 1_idx), 2) + Utils::Pow(R(2_idx, 2_idx), 2))),
        CppAD::atan2(R(1_idx, 0_idx), R(0_idx, 0_idx))};
}
#endif

template <Concepts::Scalar _Scalar>
inline _Scalar Min(const _Scalar& a, const std::type_identity_t<_Scalar>& b) {
    if constexpr (std::convertible_to<_Scalar, real_t>) {
        return std::min(a, b);
    }
#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
    else if constexpr (std::convertible_to<_Scalar, ad_scalar_t>) {
        return CppAD::CondExpGt(a, b, b, a);
    }
#endif
    else {
        Unreachable();
    }
}

template <Concepts::Scalar _Scalar>
inline _Scalar SmoothMin(const _Scalar& a,
                         const std::type_identity_t<_Scalar>& b,
                         const std::type_identity_t<_Scalar>& alpha = _Scalar{8.0}) {
    return (a * exp(-alpha * a) + b * exp(-alpha * b)) / (exp(-alpha * a) + exp(-alpha * b));
}

template <bool _LOWER_TRIANGULAR = false, typename _Matrix>  // clang-format off
requires (Concepts::DenseMatrixExpression<_Matrix> || Concepts::SparseMatrixExpression<_Matrix>) && (!_Matrix::IsRowMajor)
[[nodiscard]] bool CompareMatrices(const _Matrix& testMatrix,
                                   std::string_view testEntriesLabel,
                                   const MatrixXr& groundTruthMatrix,
                                   std::string_view groundTruthEntriesLabel) {  // clang-format on
    bool success = true;

    const real_t relativeTolerance = 1e-2;
    const real_t absoluteTolerance = 1e-3;
    const real_t epsilon           = 1e-12;

    int mismatchCounter     = 0;
    const int maxMismatches = 20;
    for (index_t j = 0_idx; j < groundTruthMatrix.cols(); ++j) {
        for (index_t i = 0_idx; i < groundTruthMatrix.rows(); ++i) {
            if constexpr (_LOWER_TRIANGULAR) {
                if (j > i) {
                    continue;
                }
            }

            const real_t absoluteError = std::abs(groundTruthMatrix(i, j) - testMatrix.coeff(i, j));
            const real_t relativeError =
                2.0 * absoluteError /
                (epsilon + std::abs(groundTruthMatrix(i, j)) + std::abs(testMatrix.coeff(i, j)));

            if ((relativeError > relativeTolerance && absoluteError > absoluteTolerance) ||
                std::isnan(testMatrix.coeff(i, j)) != std::isnan(groundTruthMatrix(i, j))) {
#ifdef UNGAR_CONFIG_ENABLE_LOGGING
                if (success) {
                    UNGAR_LOG(debug, "Some mismatches were found.");
                    UNGAR_LOG(debug, "The following mismatches were found.");
                    UNGAR_LOG(debug,
                              "{:<10}{:<10}{:<20}{:<20}{:<20}",
                              "Row",
                              "Column",
                              testEntriesLabel,
                              groundTruthEntriesLabel,
                              "Error");
                    success = false;
                }

                UNGAR_LOG(debug,
                          "{:<10}{:<10}{:<20.4f}{:<20.4f}{:<10.4f} ({:.4}%)",
                          i,
                          j,
                          testMatrix.coeff(i, j),
                          groundTruthMatrix(i, j),
                          absoluteError,
                          relativeError * 100.0);
                ++mismatchCounter;

                if (mismatchCounter > maxMismatches) {
                    const auto etc = "..."sv;
                    UNGAR_LOG(
                        debug, "{:<10}{:<10}{:<20}{:<20}{:<10}  {}", etc, etc, etc, etc, etc, etc);

                    UNGAR_LOG(debug, "More than {} mismatches were found.", maxMismatches);
                    return success;
                }
#else
                return false;
#endif
            }
        }
    }

#ifdef UNGAR_CONFIG_ENABLE_LOGGING
    if (success) {
        UNGAR_LOG(debug, "No mismatches were found.");
    }
#endif

    return success;
}

}  // namespace Utils

using Utils::Q;
using Utils::Q_c;

}  // namespace Ungar

#endif /* _UNGAR__UTILS__UTILS_HPP_ */
