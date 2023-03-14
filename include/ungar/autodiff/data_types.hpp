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

#include "ungar/data_types.hpp"

namespace Ungar {

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
