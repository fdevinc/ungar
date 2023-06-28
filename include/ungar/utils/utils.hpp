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

#include <cmath>
#include <filesystem>
#include <random>

#include "ungar/assert.hpp"
#include "ungar/data_types.hpp"

#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
#include "ungar/autodiff/data_types.hpp"
#endif

namespace Ungar {

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

static auto hasNonZeros =
    hana::is_valid([](auto&& sparseMatrix) -> decltype((void)sparseMatrix.nonZeros()) {});

template <typename... _SparseMatrices>
class VerticallyStackSparseMatricesImpl {
  public:
    using ScalarType = std::common_type_t<typename remove_cvref_t<_SparseMatrices>::Scalar...>;

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
    void In(const Eigen::SparseMatrixBase<remove_cvref_t<_SparseMatrices>>&... sparseMatrices,
            Eigen::SparseMatrixBase<_StackedSparseMatrix> const& stackedSparseMatrix) {
        UNGAR_ASSERT([](const auto& m0, const auto&... ms) {
            return ((m0.cols() == ms.cols()) && ...);
        }(sparseMatrices...));

        const index_t cols = [](const auto& m0, const auto&...) {
            return m0.cols();
        }(sparseMatrices...);

        std::vector<MutableTriplet<ScalarType>> triplets;
        if constexpr ((decltype(hasNonZeros(sparseMatrices)){} && ...)) {
            triplets.reserve((sparseMatrices.derived().nonZeros() + ...));
        }

        index_t rows = 0_idx;
        ((SparseMatrixToTriplets(sparseMatrices, triplets, rows), rows += sparseMatrices.rows()),
         ...);

        stackedSparseMatrix.const_cast_derived().resize(rows, cols);
        stackedSparseMatrix.const_cast_derived().setFromTriplets(triplets.begin(), triplets.end());
    }

    SparseMatrix<ScalarType> ToSparse(
        const Eigen::SparseMatrixBase<remove_cvref_t<_SparseMatrices>>&... sparseMatrices) {
        UNGAR_ASSERT([](const auto& m0, const auto&... ms) {
            return ((m0.cols() == ms.cols()) && ...);
        }(sparseMatrices...));

        const index_t cols = [](const auto& m0, const auto&...) {
            return m0.cols();
        }(sparseMatrices...);

        std::vector<MutableTriplet<ScalarType>> triplets;
        if constexpr ((decltype(hasNonZeros(sparseMatrices)){} && ...)) {
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

inline std::string ToSnakeCase(std::string_view input) {
    std::string output{};
    size_t i = 1UL;
    for (const char c : input) {
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

template <typename _Scalar = real_t>
inline Vector3<_Scalar> EstimateBodyFrameAngularVelocity(
    const Quaternion<_Scalar>& orientation,
    const Quaternion<_Scalar>& previousOrientation,
    const real_t stepSize) {
    return 2.0 * (previousOrientation.conjugate() * orientation).vec() / stepSize;
}

template <typename _Scalar = real_t>
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

    if constexpr (std::is_convertible_v<ScalarType, real_t>) {
        const ScalarType vNorm = v.norm();
        q.vec()                = vNorm > 0.0 ? Vector3<ScalarType>{v * sin(0.5 * vNorm) / vNorm}
                              : Vector3<ScalarType>{v * 0.5};
        q.w() = vNorm > 0.0 ? cos(0.5 * vNorm) : 1.0;
    }
#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
    else if constexpr (std::is_convertible_v<ScalarType, ad_scalar_t>) {
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
inline typename _Vector::Scalar ApproximateNorm(const Eigen::MatrixBase<_Vector>& v) {
    using ScalarType = typename _Vector::Scalar;

    return Eigen::numext::sqrt(v.squaredNorm() + Eigen::NumTraits<ScalarType>::epsilon());
}

template <typename _Vector>
inline Quaternion<typename _Vector::Scalar> ApproximateExponentialMap(
    const Eigen::MatrixBase<_Vector>& v) {
    static_assert(_Vector::RowsAtCompileTime == 3);
    using ScalarType = typename _Vector::Scalar;
    Quaternion<ScalarType> q;

    const ScalarType vApproximateNorm = ApproximateNorm(v);
    q.vec()                           = v * sin(0.5 * vApproximateNorm) / vApproximateNorm;
    q.w()                             = cos(0.5 * vApproximateNorm);
    return q;
}

template <typename _Scalar>
inline Quaternion<_Scalar> UnitQuaternionInverse(const Quaternion<_Scalar>& q) {
    return q.conjugate();
}

template <typename _Scalar>
inline Quaternion<_Scalar> UnitQuaternionInverse(const Eigen::Map<const Quaternion<_Scalar>>& q) {
    return q.conjugate();
}

template <typename _Scalar>
inline Quaternion<_Scalar> UnitQuaternionInverse(const Eigen::Map<Quaternion<_Scalar>>& q) {
    return q.conjugate();
}

template <typename _Base, typename _Exponent>
inline auto Pow(const _Base base, const _Exponent exponent) {
    if constexpr (std::is_convertible_v<_Base, real_t>) {
        return std::pow(base, exponent);
    }
#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
    else if constexpr (std::is_convertible_v<_Base, ad_scalar_t>) {
        if constexpr (std::is_integral_v<_Exponent>) {
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

template <typename _Scalar>
inline auto ElementaryXRotationMatrix(const _Scalar angle) {
    using ScalarType   = remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle);
    const ScalarType s = sin(angle);
    // clang-format off
    return Matrix3<ScalarType>{{ScalarType{1.0},  ScalarType{0.0},  ScalarType{0.0}},
                               {ScalarType{0.0},  c,                -s},
                               {ScalarType{0.0},  s,                c}};  // clang-format on
}

template <typename _Scalar>
inline auto ElementaryYRotationMatrix(const _Scalar angle) {
    using ScalarType   = remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle);
    const ScalarType s = sin(angle);
    // clang-format off
    return Matrix3<ScalarType>{{c,                ScalarType{0.0},  s},
                               {ScalarType{0.0},  ScalarType{1.0},  ScalarType{0.0}},
                               {-s,               ScalarType{0.0},  c}};  // clang-format on
}

template <typename _Scalar>
inline auto ElementaryZRotationMatrix(const _Scalar angle) {
    using ScalarType   = remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle);
    const ScalarType s = sin(angle);
    // clang-format off
    return Matrix3<ScalarType>{{c,                -s,               ScalarType{0.0}},
                               {s,                c,                ScalarType{0.0}},
                               {ScalarType{0.0},  ScalarType{0.0},  ScalarType{1.0}}};  // clang-format on
}

template <typename _Scalar>
inline auto ElementaryXQuaternion(const _Scalar angle) {
    using ScalarType   = remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle / ScalarType{2.0});
    const ScalarType s = sin(angle / ScalarType{2.0});
    return Quaternion<ScalarType>{c, s, ScalarType{0.0}, ScalarType{0.0}};
}

template <typename _Scalar>
inline auto ElementaryYQuaternion(const _Scalar angle) {
    using ScalarType   = remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle / ScalarType{2.0});
    const ScalarType s = sin(angle / ScalarType{2.0});
    return Quaternion<ScalarType>{c, ScalarType{0.0}, s, ScalarType{0.0}};
}

template <typename _Scalar>
inline auto ElementaryZQuaternion(const _Scalar angle) {
    using ScalarType   = remove_cvref_t<decltype(angle)>;
    const ScalarType c = cos(angle / ScalarType{2.0});
    const ScalarType s = sin(angle / ScalarType{2.0});
    return Quaternion<ScalarType>{c, ScalarType{0.0}, ScalarType{0.0}, s};
}

/// @brief A yaw-pitch-roll angles vector 'ypr' is defined so that
///        the yaw, pitch and roll angles correspond to 'ypr.z()',
///        'ypr.y()' and 'ypr.x()', respectively.
template <typename _Vector, typename = std::enable_if_t<_Vector::RowsAtCompileTime == 3>>
inline auto RotationMatrixFromYawPitchRoll(const Eigen::MatrixBase<_Vector>& ypr) {
    return Matrix3<typename _Vector::Scalar>{ElementaryZRotationMatrix(ypr.z()) *
                                             ElementaryYRotationMatrix(ypr.y()) *
                                             ElementaryXRotationMatrix(ypr.x())};
}

/// @brief A yaw-pitch-roll angles vector 'ypr' is defined so that
///        the yaw, pitch and roll angles correspond to 'ypr.z()',
///        'ypr.y()' and 'ypr.x()', respectively.
template <typename _Vector, typename = std::enable_if_t<_Vector::RowsAtCompileTime == 3>>
inline auto QuaternionFromYawPitchRoll(const Eigen::MatrixBase<_Vector>& ypr) {
    return Quaternion<typename _Vector::Scalar>{ElementaryZQuaternion(ypr.z()) *
                                                ElementaryYQuaternion(ypr.y()) *
                                                ElementaryXQuaternion(ypr.x())};
}

template <typename _Quaternion>
inline auto QuaternionToYawPitchRoll(const Eigen::QuaternionBase<_Quaternion>& q) {
    return Vector3<typename _Quaternion::Scalar>{
        q.toRotationMatrix().eulerAngles(2_idx, 1_idx, 0_idx).reverse()};
}

#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
template <typename _Quaternion,
          typename = std::enable_if_t<std::is_convertible_v<typename _Quaternion::Scalar,
                                                            ad_scalar_t>>>  // clang-format off
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

template <typename _Scalar>
inline _Scalar Min(const _Scalar& a, const type_identity_t<_Scalar>& b) {
    if constexpr (std::is_convertible_v<_Scalar, real_t>) {
        return std::min(a, b);
    }
#ifdef UNGAR_CONFIG_ENABLE_AUTODIFF
    else if constexpr (std::is_convertible_v<_Scalar, ad_scalar_t>) {
        return CppAD::CondExpGt(a, b, b, a);
    }
#endif
    else {
        Unreachable();
    }
}

template <typename _Scalar>
inline _Scalar SmoothMin(const _Scalar& a,
                         const type_identity_t<_Scalar>& b,
                         const type_identity_t<_Scalar>& alpha = _Scalar{8.0}) {
    return (a * exp(-alpha * a) + b * exp(-alpha * b)) / (exp(-alpha * a) + exp(-alpha * b));
}

template <bool _LOWER_TRIANGULAR = false,
          typename _Matrix,
          typename = std::enable_if_t<!_Matrix::IsRowMajor>>
[[nodiscard]] bool CompareMatrices(const _Matrix& testMatrix,
                                   std::string_view testEntriesLabel,
                                   const MatrixXr& groundTruthMatrix,
                                   std::string_view groundTruthEntriesLabel) {
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
                    [[maybe_unused]] const auto etc = "..."sv;
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

inline constexpr real_t PI = 3.14159265358979323846;

}  // namespace Utils

using Utils::Q;
using Utils::Q_c;

}  // namespace Ungar

#endif /* _UNGAR__UTILS__UTILS_HPP_ */
