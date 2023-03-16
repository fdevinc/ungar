/******************************************************************************
 *
 * @file ungar/optimization/soft_sqp.hpp
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

#ifndef _UNGAR__OPTIMIZATION__SOFT_SQP_HPP_
#define _UNGAR__OPTIMIZATION__SOFT_SQP_HPP_

#include <osqp++.h>

#include "ungar/optimization/backtracking_line_search.hpp"
#include "ungar/optimization/concepts.hpp"
#include "ungar/optimization/soft_inequality_constraint.hpp"
#include "ungar/utils/utils.hpp"

namespace Ungar {

template <Concepts::NLPProblem _NLPProblem = NLPProblem<>>
class SoftSQPOptimizer {
  public:
    SoftSQPOptimizer(const bool verbose,
                     const real_t constraintViolationMultiplier = 1.0,
                     const index_t maxIterations                = 10_idx,
                     const real_t stiffness                     = 1.0,
                     const real_t epsilon                       = 2e-5,
                     const bool polish                          = true)
        : _constraintViolationMultiplier{constraintViolationMultiplier},
          _maxIterations{maxIterations},
          _stiffness{stiffness},
          _epsilon{epsilon} {
        _osqpSettings.verbose = verbose;
        _osqpSettings.polish  = polish;
    }

    void Initialize(std::unique_ptr<_NLPProblem> nlpProblem) {
        _nlpProblem = std::move(nlpProblem);
        _softInequalityCnstrs =
            std::make_unique<Autodiff::Function>(MakeSoftInequalityConstraintFunction());
    }

    RefToConstVectorXr Optimize(const VectorXr& xp) {
        UNGAR_ASSERT(xp.size() == _nlpProblem->objective.IndependentVariableSize() +
                                      _nlpProblem->objective.ParameterSize());

        VectorXr xpHelper = xp;
        _cache.xp         = xp;

        for (const auto i : enumerate(_maxIterations)) {
            if (_osqpSettings.verbose) {
                UNGAR_LOG(trace, "Starting soft SQP iteration {}...", i);
            }
            const real_t objective =
                FunctionInterface::Invoke(_nlpProblem->objective, _cache.xp)[0_idx];

            SolveLocalQPProblem(_cache.xp);
            const bool accepted = BacktrackingLineSearch{_osqpSettings.verbose}.Do(
                FunctionInterface::Jacobian(_nlpProblem->objective, _cache.xp)
                    .toDense()
                    .transpose(),
                _osqpSolver.primal_solution(),
                [&](const RefToConstVectorXr& x) -> real_t {
                    xpHelper.head(x.size()) = x;
                    return FunctionInterface::Invoke(_nlpProblem->objective, xpHelper)[0_idx] +
                           EvaluateSoftInequalityConstraints(xpHelper);
                },
                [&](const RefToConstVectorXr& x) -> real_t {
                    xpHelper.head(x.size()) = x;

                    real_t constraintViolation =
                        FunctionInterface::Invoke(_nlpProblem->equalityConstraints, xpHelper)
                            .squaredNorm();

                    return _constraintViolationMultiplier * std::sqrt(constraintViolation);
                },
                _cache.xp.head(_nlpProblem->objective.IndependentVariableSize()));
            if (!accepted) {
                break;
            }

            const real_t updatedObjective =
                FunctionInterface::Invoke(_nlpProblem->objective, _cache.xp)[0_idx];
            const real_t objectiveDifference = updatedObjective - objective;
            if (objectiveDifference < 0.0 && std::abs(objectiveDifference) < 1e-6) {
                if (_osqpSettings.verbose) {
                    UNGAR_LOG(trace, "Soft SQP convergence criterion met.");
                }
                break;
            }
        }

        return _cache.xp.head(_nlpProblem->objective.IndependentVariableSize());
    }

  private:
    void AssembleOSQPInstance(const VectorXr& xp) {
        // Regularized objective matrix.
        _osqpInstance.objective_matrix =
            SparseMatrix<real_t>{_nlpProblem->objective.Hessian(0_idx, xp)} +
            SoftInequalityConstraintsHessianApproximation(xp) +
            SparseMatrix<real_t>{
                VectorXr::Constant(_nlpProblem->objective.IndependentVariableSize(), 1e-6)
                    .asDiagonal()};
        _osqpInstance.objective_vector = _nlpProblem->objective.Jacobian(xp).transpose().toDense() +
                                         SoftInequalityConstraintsJacobian(xp).transpose();

        _osqpInstance.constraint_matrix = _nlpProblem->equalityConstraints.Jacobian(xp);
        _osqpInstance.lower_bounds =
            -FunctionInterface::Invoke(_nlpProblem->equalityConstraints, xp);
        _osqpInstance.upper_bounds =
            -FunctionInterface::Invoke(_nlpProblem->equalityConstraints, xp);
    }

    void InitializeImpl(const VectorXr& xp) {
        AssembleOSQPInstance(xp);

        UNGAR_ASSERT(_osqpInstance.objective_matrix.cols() ==
                     _osqpInstance.constraint_matrix.cols());
        UNGAR_ASSERT(_osqpInstance.lower_bounds.size() == _osqpInstance.upper_bounds.size());

        UNGAR_ASSERT(!Utils::HasNaN(_osqpInstance.objective_matrix));
        UNGAR_ASSERT(!Utils::HasNaN(_osqpInstance.constraint_matrix));
        UNGAR_ASSERT(!_osqpInstance.objective_vector.hasNaN());

        auto status = _osqpSolver.Init(_osqpInstance, _osqpSettings);
        if (!status.ok()) {
            UNGAR_LOG(error,
                      "OSQP solver initialization failed with error message: \"{}\".",
                      status.message());
            UNGAR_ASSERT(status.ok());
        }
    }

    void SolveLocalQPProblem(const VectorXr& xp) {
        if (!_osqpSolver.IsInitialized()) {
            InitializeImpl(xp);
        } else {
            AssembleOSQPInstance(xp);
        }

        auto status = _osqpSolver.UpdateObjectiveAndConstraintMatrices(
            _osqpInstance.objective_matrix, _osqpInstance.constraint_matrix);
        if (!status.ok()) {
            UNGAR_LOG(error,
                      "OSQP solver objective and constraint matrices update failed with error "
                      "message: \"{}\".",
                      status.message());
            UNGAR_ASSERT(status.ok());
        }

        status = _osqpSolver.SetObjectiveVector(_osqpInstance.objective_vector);
        if (!status.ok()) {
            UNGAR_LOG(error,
                      "OSQP solver objective vector update failed with error message: \"{}\".",
                      status.message());
            UNGAR_ASSERT(status.ok());
        }

        status = _osqpSolver.SetBounds(_osqpInstance.lower_bounds, _osqpInstance.upper_bounds);
        if (!status.ok()) {
            UNGAR_LOG(
                error,
                "OSQP solver lower and upper bounds update failed with error message: \"{}\".",
                status.message());
            UNGAR_ASSERT(status.ok());
        }

        UNGAR_ASSERT(!Utils::HasNaN(_osqpInstance.objective_matrix));
        UNGAR_ASSERT(!Utils::HasNaN(_osqpInstance.constraint_matrix));
        UNGAR_ASSERT(!_osqpInstance.objective_vector.hasNaN());
        UNGAR_ASSERT(!_osqpInstance.lower_bounds.hasNaN());
        UNGAR_ASSERT(!_osqpInstance.upper_bounds.hasNaN());

        const osqp::OsqpExitCode exitCode = _osqpSolver.Solve();
        if (exitCode != osqp::OsqpExitCode::kOptimal) {
            UNGAR_LOG(error,
                      "The OSQP solver could not converge. Exit code: '{}'.",
                      osqp::ToString(exitCode));
            /// @todo Manage all different exit codes.
            UNGAR_ASSERT(exitCode == osqp::OsqpExitCode::kOptimal);
        }
    }

    Autodiff::Function MakeSoftInequalityConstraintFunction() const {
        auto Zsoft = [&](const VectorXad& variables, VectorXad& Zsoft) -> void {
            Zsoft.resize(1_idx);
            Zsoft << SoftInequalityConstraint{0.0, _stiffness, _epsilon}.Evaluate<ad_scalar_t>(
                -variables);
        };

        std::string softIneqModelName = Utils::ToSnakeCase(
            "soft_sqp_soft_ineq_sz_"s +
            std::to_string(_nlpProblem->inequalityConstraints.DependentVariableSize()) + "_k_" +
            std::to_string(_stiffness) + "_eps_" + std::to_string(_epsilon));
        return Autodiff::MakeFunction({Zsoft,
                                       _nlpProblem->inequalityConstraints.DependentVariableSize(),
                                       0_idx,
                                       softIneqModelName,
                                       EnabledDerivatives::ALL},
                                      false);
    }

    real_t EvaluateSoftInequalityConstraints(const VectorXr& xp) {
        const VectorXr Zineq{FunctionInterface::Invoke(_nlpProblem->inequalityConstraints, xp)};

        return (*_softInequalityCnstrs)(Zineq)[0_idx];
    }

    RowVectorXr SoftInequalityConstraintsJacobian(const VectorXr& xp) {
        const VectorXr Zineq{FunctionInterface::Invoke(_nlpProblem->inequalityConstraints, xp)};
        const auto ineqJacobian =
            FunctionInterface::Jacobian(_nlpProblem->inequalityConstraints, xp);

        return _softInequalityCnstrs->Jacobian(Zineq) * ineqJacobian;
    }

    Eigen::SparseMatrix<real_t> SoftInequalityConstraintsHessianApproximation(const VectorXr& xp) {
        const VectorXr Zineq{FunctionInterface::Invoke(_nlpProblem->inequalityConstraints, xp)};
        const auto ineqJacobian =
            FunctionInterface::Jacobian(_nlpProblem->inequalityConstraints, xp);

        return ineqJacobian.transpose() * _softInequalityCnstrs->Hessian(0_idx, Zineq) *
               ineqJacobian;
    }

    osqp::OsqpInstance _osqpInstance;
    osqp::OsqpSolver _osqpSolver;
    osqp::OsqpSettings _osqpSettings;

    std::unique_ptr<_NLPProblem> _nlpProblem;

    real_t _constraintViolationMultiplier;
    index_t _maxIterations;

    struct {
        VectorXr xp;
    } _cache;

    real_t _stiffness;
    real_t _epsilon;
    std::unique_ptr<Autodiff::Function> _softInequalityCnstrs;
};

}  // namespace Ungar

#endif /* _UNGAR__OPTIMIZATION__SOFT_SQP_HPP_ */
