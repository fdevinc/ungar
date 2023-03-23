/******************************************************************************
 *
 * @file ungar/autodiff/data_types.hpp
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

#ifndef _UNGAR__AUTODIFF__DATA_TYPES_HPP_
#define _UNGAR__AUTODIFF__DATA_TYPES_HPP_

#include <unordered_set>

#include <cppad/cg.hpp>
#include <cppad/cg/support/cppadcg_eigen.hpp>

#include "ungar/data_types.hpp"

namespace Ungar {

using ADCG        = CppAD::cg::CG<real_t>;
using ADFun       = CppAD::ADFun<ADCG>;
using ad_scalar_t = CppAD::AD<ADCG>;

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

UNGAR_MAKE_EIGEN_TYPEDEFS(ad_scalar_t, ad);
#undef UNGAR_MAKE_EIGEN_TYPEDEFS
#undef _UNGAR_MAKE_EIGEN_TYPEDEFS_IMPL

using Quaternionad           = Eigen::Quaternion<ad_scalar_t>;
using MapToQuaternionad      = Eigen::Map<Eigen::Quaternion<ad_scalar_t>>;
using MapToConstQuaternionad = Eigen::Map<const Eigen::Quaternion<ad_scalar_t>>;
using AngleAxisad            = Eigen::AngleAxis<ad_scalar_t>;
using Rotation2Dad           = Eigen::Rotation2D<ad_scalar_t>;

/**
 * @brief An AD function f computes the value of a dependent variable
 *        vector y as:
 *            y = f([x^t p^t]^t) ,
 *        where x is a vector of independent variables and p is a
 *        vector of parameters. In particular, all variables are AD
 *        scalars.
 */
using ADFunction = std::function<void(const VectorXad& xp, VectorXad& y)>;

using SparsityPattern = std::vector<std::unordered_set<size_t>>;

enum class EnabledDerivatives : unsigned {
    NONE     = 1U << 0,
    JACOBIAN = 1U << 1,
    HESSIAN  = 1U << 2,
    ALL      = JACOBIAN | HESSIAN
};

constexpr auto operator|(const EnabledDerivatives lhs, const EnabledDerivatives rhs) {
    return static_cast<EnabledDerivatives>(
        static_cast<std::underlying_type_t<EnabledDerivatives>>(lhs) |
        static_cast<std::underlying_type_t<EnabledDerivatives>>(rhs));
}

constexpr auto operator&(const EnabledDerivatives lhs, const EnabledDerivatives rhs) {
    return static_cast<bool>(static_cast<std::underlying_type_t<EnabledDerivatives>>(lhs) &
                             static_cast<std::underlying_type_t<EnabledDerivatives>>(rhs));
}

}  // namespace Ungar

#endif /* _UNGAR__AUTODIFF__DATA_TYPES_HPP_ */
