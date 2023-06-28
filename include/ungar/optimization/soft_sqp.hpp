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

namespace Ungar {

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

    template <typename _NLPProblem,
              typename _XP,
              std::enable_if_t<std::is_same_v<typename _XP::Scalar, real_t> &&
                                   is_nlp_problem_v<_NLPProblem>,
                               bool> = true>
    RefToConstVectorXr Optimize(const _NLPProblem& nlpProblem, const Eigen::MatrixBase<_XP>& xp) {
        UNGAR_ASSERT(xp.size() == nlpProblem.objective.IndependentVariableSize() +
                                      nlpProblem.objective.ParameterSize());

        VectorXr xpHelper = xp;
        _cache.xp         = xp;

        for (const auto i : enumerate(_maxIterations)) {
            if (_osqpSettings.verbose) {
                UNGAR_LOG(trace, "Starting soft SQP iteration {}...", i);
            }
            const real_t objective =
                FunctionInterface::Invoke(nlpProblem.objective, _cache.xp)[0_idx];

            SolveLocalQPProblem(nlpProblem, _cache.xp);
            const bool accepted = BacktrackingLineSearch{_osqpSettings.verbose}.Do(
                FunctionInterface::Jacobian(nlpProblem.objective, _cache.xp).toDense().transpose(),
                _osqpSolver.primal_solution(),
                [&](const RefToConstVectorXr& x) -> real_t {
                    xpHelper.head(x.size()) = x;
                    return FunctionInterface::Invoke(nlpProblem.objective, xpHelper)[0_idx] +
                           EvaluateSoftInequalityConstraints(nlpProblem, xpHelper);
                },
                [&](const RefToConstVectorXr& x) -> real_t {
                    xpHelper.head(x.size()) = x;

                    real_t constraintViolation =
                        FunctionInterface::Invoke(nlpProblem.equalityConstraints, xpHelper)
                            .squaredNorm();

                    return _constraintViolationMultiplier * std::sqrt(constraintViolation);
                },
                _cache.xp.head(nlpProblem.objective.IndependentVariableSize()));
            if (!accepted) {
                break;
            }

            const real_t updatedObjective =
                FunctionInterface::Invoke(nlpProblem.objective, _cache.xp)[0_idx];
            const real_t objectiveDifference = updatedObjective - objective;
            if (objectiveDifference < 0.0 && std::abs(objectiveDifference) < 1e-6) {
                if (_osqpSettings.verbose) {
                    UNGAR_LOG(trace, "Soft SQP convergence criterion met.");
                }
                break;
            }
        }

        return _cache.xp.head(nlpProblem.objective.IndependentVariableSize());
    }

  private:
    template <typename _NLPProblem,
              typename _XP,
              std::enable_if_t<std::is_same_v<typename _XP::Scalar, real_t> &&
                                   is_nlp_problem_v<_NLPProblem>,
                               bool> = true>
    void AssembleOSQPInstance(const _NLPProblem& nlpProblem, const Eigen::MatrixBase<_XP>& xp) {
        // Regularized objective matrix.
        _osqpInstance.objective_matrix =
            SparseMatrix<real_t>{nlpProblem.objective.Hessian(0_idx, xp)} +
            SoftInequalityConstraintsHessianApproximation(nlpProblem, xp) +
            SparseMatrix<real_t>{
                VectorXr::Constant(nlpProblem.objective.IndependentVariableSize(), 1e-6)
                    .asDiagonal()};
        _osqpInstance.objective_vector =
            nlpProblem.objective.Jacobian(xp).transpose().toDense() +
            SoftInequalityConstraintsJacobian(nlpProblem, xp).transpose();

        _osqpInstance.constraint_matrix = nlpProblem.equalityConstraints.Jacobian(xp);
        _osqpInstance.lower_bounds = -FunctionInterface::Invoke(nlpProblem.equalityConstraints, xp);
        _osqpInstance.upper_bounds = -FunctionInterface::Invoke(nlpProblem.equalityConstraints, xp);
    }

    template <typename _NLPProblem,
              typename _XP,
              std::enable_if_t<std::is_same_v<typename _XP::Scalar, real_t> &&
                                   is_nlp_problem_v<_NLPProblem>,
                               bool> = true>
    void InitializeImpl(const _NLPProblem& nlpProblem, const Eigen::MatrixBase<_XP>& xp) {
        _softInequalityCnstrs =
            std::make_unique<Autodiff::Function>(MakeSoftInequalityConstraintFunction(nlpProblem));
        AssembleOSQPInstance(nlpProblem, xp);

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

    template <typename _NLPProblem,
              typename _XP,
              std::enable_if_t<std::is_same_v<typename _XP::Scalar, real_t> &&
                                   is_nlp_problem_v<_NLPProblem>,
                               bool> = true>
    void SolveLocalQPProblem(const _NLPProblem& nlpProblem, const Eigen::MatrixBase<_XP>& xp) {
        if (!_osqpSolver.IsInitialized()) {
            InitializeImpl(nlpProblem, xp);
        } else {
            AssembleOSQPInstance(nlpProblem, xp);
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

    template <typename _NLPProblem, std::enable_if_t<is_nlp_problem_v<_NLPProblem>, bool> = true>
    Autodiff::Function MakeSoftInequalityConstraintFunction(const _NLPProblem& nlpProblem) const {
        auto Zsoft = [&](const VectorXad& variables, VectorXad& Zsoft) -> void {
            Zsoft.resize(1_idx);
            Zsoft << SoftInequalityConstraint{0.0, _stiffness, _epsilon}.Evaluate<ad_scalar_t>(
                -variables);
        };

        std::string softIneqModelName = Utils::ToSnakeCase(
            "soft_sqp_soft_ineq_sz_"s +
            std::to_string(nlpProblem.inequalityConstraints.DependentVariableSize()) + "_k_" +
            std::to_string(_stiffness) + "_eps_" + std::to_string(_epsilon));
        return Autodiff::MakeFunction({Zsoft,
                                       nlpProblem.inequalityConstraints.DependentVariableSize(),
                                       0_idx,
                                       softIneqModelName,
                                       EnabledDerivatives::ALL},
                                      false);
    }

    template <typename _NLPProblem,
              typename _XP,
              std::enable_if_t<std::is_same_v<typename _XP::Scalar, real_t> &&
                                   is_nlp_problem_v<_NLPProblem>,
                               bool> = true>
    real_t EvaluateSoftInequalityConstraints(const _NLPProblem& nlpProblem,
                                             const Eigen::MatrixBase<_XP>& xp) {
        const VectorXr Zineq{FunctionInterface::Invoke(nlpProblem.inequalityConstraints, xp)};

        return (*_softInequalityCnstrs)(Zineq)[0_idx];
    }

    template <typename _NLPProblem,
              typename _XP,
              std::enable_if_t<std::is_same_v<typename _XP::Scalar, real_t> &&
                                   is_nlp_problem_v<_NLPProblem>,
                               bool> = true>
    RowVectorXr SoftInequalityConstraintsJacobian(const _NLPProblem& nlpProblem,
                                                  const Eigen::MatrixBase<_XP>& xp) {
        const VectorXr Zineq{FunctionInterface::Invoke(nlpProblem.inequalityConstraints, xp)};
        const auto ineqJacobian = FunctionInterface::Jacobian(nlpProblem.inequalityConstraints, xp);

        return _softInequalityCnstrs->Jacobian(Zineq) * ineqJacobian;
    }

    template <typename _NLPProblem,
              typename _XP,
              std::enable_if_t<std::is_same_v<typename _XP::Scalar, real_t> &&
                                   is_nlp_problem_v<_NLPProblem>,
                               bool> = true>
    Eigen::SparseMatrix<real_t> SoftInequalityConstraintsHessianApproximation(
        const _NLPProblem& nlpProblem, const Eigen::MatrixBase<_XP>& xp) {
        const VectorXr Zineq{FunctionInterface::Invoke(nlpProblem.inequalityConstraints, xp)};
        const auto ineqJacobian = FunctionInterface::Jacobian(nlpProblem.inequalityConstraints, xp);

        return ineqJacobian.transpose() * _softInequalityCnstrs->Hessian(0_idx, Zineq) *
               ineqJacobian;
    }

    osqp::OsqpInstance _osqpInstance;
    osqp::OsqpSolver _osqpSolver;
    osqp::OsqpSettings _osqpSettings;

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
