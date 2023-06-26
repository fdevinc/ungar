/******************************************************************************
 *
 * @file ungar/optimization/soft_inequality_constraint.hpp
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

#ifndef _UNGAR__OPTIMIZATION__SOFT_INEQUALITY_CONSTRAINT_HPP_
#define _UNGAR__OPTIMIZATION__SOFT_INEQUALITY_CONSTRAINT_HPP_

#include "ungar/data_types.hpp"

namespace Ungar {

class LogisticFunction {
  public:
    constexpr LogisticFunction(const real_t midpoint     = 0.0,
                               const real_t steepness    = 1.0,
                               const real_t maximumValue = 1.0)
        : _midpoint{midpoint}, _steepness{steepness}, _maximumValue{maximumValue} {
    }

    template <typename _Scalar>
    _Scalar Evaluate(const _Scalar& x) const {
        return _maximumValue / (1.0 + exp(-_steepness * (x - _midpoint)));
    }

    template <typename _Scalar>
    static constexpr _Scalar SmoothGreaterThan(const _Scalar x,
                                               const real_t lhs,
                                               const real_t steepness = 1024.0) {
        return LogisticFunction{lhs, steepness}.Evaluate<_Scalar>(x);
    }

    template <typename _Scalar>
    static constexpr _Scalar SmoothLessThan(const _Scalar x,
                                            const real_t lhs,
                                            const real_t steepness = 1024.0) {
        return 1.0 - SmoothGreaterThan(x, lhs, steepness);
    }

  private:
    real_t _midpoint;
    real_t _steepness;
    real_t _maximumValue;
};

/**
 * @brief Implement soft inequality constraints of the form 'lhs >= rhs'.
 *
 */
class SoftInequalityConstraint {
  public:
    constexpr SoftInequalityConstraint(const real_t rhs,
                                       const real_t stiffness = 1.0,
                                       const real_t epsilon   = 2e-5)
        : _rhs{rhs},
          _epsilon{epsilon},
          _a1{stiffness},
          _b1{-0.5 * _a1 * _epsilon},
          _c1{-1.0 / 3.0 * (-_b1 - _a1 * _epsilon) * _epsilon - 0.5 * _a1 * pow(_epsilon, 2) -
              _b1 * _epsilon},
          _a2{(-_b1 - _a1 * _epsilon) / pow(_epsilon, 2)},
          _b2{_a1},
          _c2{_b1},
          _d2{_c1} {
    }

    template <typename _Scalar, bool NO_CONDITIONAL_APPROXIMATION = false>
    _Scalar Evaluate(const _Scalar& lhs) const {
        if constexpr (NO_CONDITIONAL_APPROXIMATION) {
            return EvaluateNoConditionalApproximationImpl(lhs - _rhs);
        } else {
            return EvaluateImpl(lhs - _rhs);
        }
    }

    template <typename _Scalar, bool NO_CONDITIONAL_APPROXIMATION = false>
    _Scalar Evaluate(const RefToConstVectorX<_Scalar>& lhs) const {
        return (lhs.array() - _rhs)
            .unaryExpr([this](const auto& coeff) -> _Scalar {
                if constexpr (NO_CONDITIONAL_APPROXIMATION) {
                    return EvaluateNoConditionalApproximationImpl(coeff);
                } else {
                    return EvaluateImpl(coeff);
                }
            })
            .sum();
    }

  private:
    real_t EvaluateImpl(const real_t& x) const {
        if (x < 0.0) {
            return 0.5 * _a1 * pow(x, 2) + _b1 * x + _c1;
        } else if (x < _epsilon) {
            return 1.0 / 3.0 * _a2 * pow(x, 3) + 0.5 * _b2 * pow(x, 2) + _c2 * x + _d2;
        } else {
            return 0.0;
        }
    }

    ad_scalar_t EvaluateImpl(const ad_scalar_t& x) const {
        return CppAD::CondExpLt(
            x,
            ad_scalar_t{0.0},
            0.5 * _a1 * pow(x, 2) + _b1 * x + _c1,
            CppAD::CondExpLt(x,
                             ad_scalar_t{_epsilon},
                             1.0 / 3.0 * _a2 * pow(x, 3) + 0.5 * _b2 * pow(x, 2) + _c2 * x + _d2,
                             ad_scalar_t{0.0}));
    }

    template <typename _Scalar>
    _Scalar EvaluateNoConditionalApproximationImpl(const _Scalar& x) const {
        const real_t steepness = 1.0 / _epsilon;
        return LogisticFunction::SmoothLessThan(x, 0.0, steepness) *
                   (0.5 * _a1 * pow(x, 2) + _b1 * x + _c1) +
               LogisticFunction::SmoothGreaterThan(x, 0.0, steepness) *
                   LogisticFunction::SmoothLessThan(x, _epsilon, steepness) *
                   (1.0 / 3.0 * _a2 * pow(x, 3) + 0.5 * _b2 * pow(x, 2) + _c2 * x + _d2);
    }

    real_t _rhs, _epsilon;
    real_t _a1, _b1, _c1;
    real_t _a2, _b2, _c2, _d2;
};

class SoftBoundConstraint {
  public:
    constexpr SoftBoundConstraint(const real_t lowerBound,
                                  const real_t upperBound,
                                  const real_t stiffness       = 1.0,
                                  const real_t relativeEpsilon = 1e-1)
        : _epsilon{(upperBound - lowerBound) * relativeEpsilon},
          _lowerBound{lowerBound, stiffness, _epsilon},
          _upperBound{-upperBound, stiffness, _epsilon} {
    }

    template <typename _Scalar, bool NO_CONDITIONAL_APPROXIMATION = false>
    _Scalar Evaluate(const _Scalar& x) const {
        return _lowerBound.Evaluate<_Scalar, NO_CONDITIONAL_APPROXIMATION>(x) +
               _upperBound.Evaluate<_Scalar, NO_CONDITIONAL_APPROXIMATION>(-x);
    }

    template <typename _Scalar, bool NO_CONDITIONAL_APPROXIMATION = false>
    _Scalar Evaluate(const RefToConstVectorX<_Scalar>& x) const {
        return _lowerBound.Evaluate<_Scalar, NO_CONDITIONAL_APPROXIMATION>(x) +
               _upperBound.Evaluate<_Scalar, NO_CONDITIONAL_APPROXIMATION>(-x);
    }

  private:
    real_t _epsilon;
    SoftInequalityConstraint _lowerBound;
    SoftInequalityConstraint _upperBound;
};

}  // namespace Ungar

#endif /* _UNGAR__OPTIMIZATION__SOFT_INEQUALITY_CONSTRAINT_HPP_ */
