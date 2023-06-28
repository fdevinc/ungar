/******************************************************************************
 *
 * @file ungar/optimization/backtracking_line_search.hpp
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

#ifndef _UNGAR__OPTIMIZATION__BACKTRACKING_LINE_SEARCH_HPP_
#define _UNGAR__OPTIMIZATION__BACKTRACKING_LINE_SEARCH_HPP_

#include "ungar/assert.hpp"
#include "ungar/data_types.hpp"

namespace Ungar {

class BacktrackingLineSearch {
  public:
    struct Parameters {
        BOOST_HANA_DEFINE_STRUCT(Parameters,
                                 (real_t, alphaMin),
                                 (real_t, thetaMin),
                                 (real_t, thetaMax),
                                 (real_t, eta),
                                 (real_t, gammaPhi),
                                 (real_t, gammaTheta),
                                 (real_t, gammaAlpha));
    };

    constexpr BacktrackingLineSearch(
        const bool verbose, Parameters parameters = {1e-4, 1e-6, 1e-2, 1e-4, 1e-6, 1e-6, 0.5})
        : _verbose{verbose}, _parameters(std::move(parameters)) {
    }

    template <typename _CostFunctionGradient,
              typename _SearchDirection,
              typename _CostFunction,
              typename _ConstraintViolation,
              typename _W>
    [[nodiscard]] bool Do(const Eigen::MatrixBase<_CostFunctionGradient>& costFunctionGradient,
                          const Eigen::MatrixBase<_SearchDirection>& dw,
                          const _CostFunction& costFunction,
                          const _ConstraintViolation& constraintViolation,
                          Eigen::MatrixBase<_W> const& w,
                          const bool verbose = false) const {
        UNGAR_ASSERT(costFunctionGradient.cols() == 1_idx);
        UNGAR_ASSERT(dw.cols() == 1_idx);
        UNGAR_ASSERT(costFunctionGradient.rows() == dw.rows());

        const real_t dwProjection = (costFunctionGradient.array() * dw.array()).sum();

        real_t alpha       = 1.0;
        const real_t theta = constraintViolation(w);
        const real_t phi   = costFunction(w);
        if (_verbose) {
            UNGAR_LOG(trace, "Performing a backtracking line search...");
            UNGAR_LOG(trace,
                      "\t{:>5}{:>36}{:>36}{:>36}",
                      "",
                      "",
                      "Initial constraint violation",
                      "Initial cost function value");
            UNGAR_LOG(trace, "\t{:>5}{:>36}{:>36}{:>36}", "", "", theta, phi);
        }

        bool accepted          = false;
        [[maybe_unused]] int i = 1;
        if (_verbose) {
            UNGAR_LOG(trace,
                      "\t{:>5}{:>36}{:>36}{:>36}",
                      "It.",
                      "Step size",
                      "Constraint violation",
                      "Cost function value");
        }
        while (!accepted && alpha >= _parameters.alphaMin) {
            const VectorXr wNext   = w + alpha * dw;
            const real_t thetaNext = constraintViolation(wNext);
            const real_t phiNext   = costFunction(wNext);
            if (_verbose) {
                UNGAR_LOG(trace, "\t{:>5}{:>36}{:>36}{:>36}", i++, alpha, thetaNext, phiNext);
            }

            if (thetaNext > _parameters.thetaMax) {
                if (thetaNext < (1.0 - _parameters.gammaTheta) * theta) {
                    accepted = true;
                    if (_verbose) {
                        UNGAR_LOG(trace, "Iteration accepted to reduce the constraint violations.");
                    }
                }
            } else if (std::max(theta, thetaNext) < _parameters.thetaMin && dwProjection < 0.0) {
                if (phiNext < phi + _parameters.eta * alpha * dwProjection) {
                    accepted = true;
                    if (_verbose) {
                        UNGAR_LOG(trace, "Armijo condition satisfied: iteration accepted.");
                    }
                }
            } else {
                if (phiNext < (1.0 - _parameters.gammaPhi) * phi ||
                    thetaNext < (1.0 - _parameters.gammaTheta) * theta) {
                    accepted = true;
                    if (_verbose) {
                        UNGAR_LOG(trace, "Iteration accepted.");
                    }
                }
            }

            if (!accepted) {
                alpha *= _parameters.gammaAlpha;
            }
        }

        if (accepted) {
            w.const_cast_derived() += alpha * dw;
        } else {
            if (_verbose) {
                UNGAR_LOG(
                    trace,
                    "The line search could not find a satisfactory step size and the solution "
                    "was not updated.");
            }
        }

        return accepted;
    }

    const Parameters& Parameters() const {
        return _parameters;
    }

    void SetParameters(struct Parameters parameters) {
        _parameters = std::move(parameters);
    }

  private:
    bool _verbose;
    struct Parameters _parameters;
};

}  // namespace Ungar

#endif /* _UNGAR__BACKTRACKING_LINE_SEARCH_HPP_ */
