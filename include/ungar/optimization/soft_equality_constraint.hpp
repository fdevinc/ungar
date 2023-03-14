/******************************************************************************
 *
 * @file ungar/optimization/soft_equality_constraint.hpp
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

#ifndef _UNGAR__OPTIMIZATION__SOFT_EQUALITY_CONSTRAINT_HPP_
#define _UNGAR__OPTIMIZATION__SOFT_EQUALITY_CONSTRAINT_HPP_

#include "ungar/data_types.hpp"

namespace Ungar {

class SoftEqualityConstraint {
  public:
    constexpr SoftEqualityConstraint(const real_t rhs, const real_t stiffness = 1.0)
        : _rhs{rhs}, _stiffness{stiffness} {
    }

    template <Ungar::Concepts::Scalar _Scalar>
    _Scalar Evaluate(const _Scalar& lhs) const {
        return 0.5 * _stiffness * pow(lhs - _rhs, 2);
    }

    template <Ungar::Concepts::Scalar _Scalar>
    _Scalar Evaluate(const RefToConstVectorX<_Scalar>& lhs) const {
        return 0.5 * _stiffness * (lhs.array() - _rhs).square().sum();
    }

  private:
    real_t _rhs, _stiffness;
};

}  // namespace Ungar

#endif /* _UNGAR__OPTIMIZATION__SOFT_EQUALITY_CONSTRAINT_HPP_ */
