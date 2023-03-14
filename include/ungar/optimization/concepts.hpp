/******************************************************************************
 *
 * @file ungar/optimization/concepts.hpp
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

#ifndef _UNGAR__OPTIMIZATION__CONCEPTS_HPP_
#define _UNGAR__OPTIMIZATION__CONCEPTS_HPP_

#include <unsupported/Eigen/SparseExtra>

#include "ungar/autodiff/function.hpp"

namespace Ungar {
namespace Concepts {

// clang-format off
template <typename _EvaluableFunction, typename _XP, typename _Y>
concept EvaluableFunction =
    Ungar::Concepts::DenseVectorExpression<_XP> &&
    requires (const _EvaluableFunction function,
              Eigen::MatrixBase<_XP> xp,
              Eigen::MatrixBase<_Y> y) {
    { function.IndependentVariableSize() } -> std::same_as<index_t>;
    { function.ParameterSize() } -> std::same_as<index_t>;
    { function.DependentVariableSize() } -> std::same_as<index_t>;

    requires std::same_as<typename _XP::Scalar, typename _Y::Scalar>;
    { function.Evaluate(xp, y) }
        -> std::same_as<void>;
};

template <typename _DifferentiableFunction, typename _XP>
concept DifferentiableFunction =
    Ungar::Concepts::DenseVectorExpression<_XP> &&
    requires (const _DifferentiableFunction function,
              Eigen::MatrixBase<_XP> xp) {
    { function.IndependentVariableSize() } -> std::same_as<index_t>;
    { function.ParameterSize() } -> std::same_as<index_t>;
    { function.DependentVariableSize() } -> std::same_as<index_t>;

    { function(xp) }
        -> Ungar::Concepts::DenseVectorExpression;
    { function.Jacobian(xp) }
        -> Ungar::Concepts::SparseMatrixExpression;
};

template <typename _TwiceDifferentiableFunction, typename _XP>
concept TwiceDifferentiableFunction =
    DifferentiableFunction<_TwiceDifferentiableFunction, _XP> &&
    requires (const _TwiceDifferentiableFunction function,
              index_t dependentVariableIndex,
              Eigen::MatrixBase<_XP> xp) {
    { function.Hessian(dependentVariableIndex, xp) }
        -> Ungar::Concepts::SparseMatrixExpression;
};
// clang-format on

}  // namespace Concepts

static_assert(Concepts::EvaluableFunction<Autodiff::Function, VectorXr, VectorXr>,
              "The EvaluableFunction concept must be satisfied by the Autodiff::Function class.");
static_assert(
    Concepts::DifferentiableFunction<Autodiff::Function, VectorXr>,
    "The DifferentiableFunction concept must be satisfied by the Autodiff::Function class.");
static_assert(
    Concepts::TwiceDifferentiableFunction<Autodiff::Function, VectorXr>,
    "The TwiceDifferentiableFunction concept must be satisfied by the Autodiff::Function class.");

class IdleTwiceDifferentiableFunction {
  public:
    constexpr IdleTwiceDifferentiableFunction(const index_t independentVariableSize,
                                              const index_t parameterSize) noexcept
        : _independentVariableSize{independentVariableSize}, _parameterSize{parameterSize} {
    }

    template <typename _XP>
    VectorX<typename _XP::Scalar> operator()(
        [[maybe_unused]] const Eigen::MatrixBase<_XP>& xp) const {
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);

        using ScalarType = typename _XP::Scalar;
        return VectorX<ScalarType>{};
    }

    template <typename _XP, typename _Y>
    requires std::same_as<typename _XP::Scalar, typename _Y::Scalar>
    void Evaluate([[maybe_unused]] const Eigen::MatrixBase<_XP>& xp,
                  [[maybe_unused]] Eigen::MatrixBase<_Y> const& y) const {
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);
        UNGAR_ASSERT(y.size() == 0_idx);
    }

    template <typename _XP>
    SparseMatrix<typename _XP::Scalar> Jacobian(
        [[maybe_unused]] const Eigen::MatrixBase<_XP>& xp) const {
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);

        using ScalarType = typename _XP::Scalar;
        return SparseMatrix<ScalarType>{0_idx, _independentVariableSize};
    }

    template <typename _XP>
    SparseMatrix<typename _XP::Scalar> Hessian(
        [[maybe_unused]] const index_t dependentVariableIndex,
        [[maybe_unused]] const Eigen::MatrixBase<_XP>& xp) const {
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);
        UNGAR_ASSERT(dependentVariableIndex == 0_idx);

        using ScalarType = typename _XP::Scalar;
        return SparseMatrix<ScalarType>{_independentVariableSize, _independentVariableSize};
    }

    index_t IndependentVariableSize() const {
        return _independentVariableSize;
    }

    index_t ParameterSize() const {
        return _parameterSize;
    }

    constexpr index_t DependentVariableSize() const {
        return 0_idx;
    }

  private:
    index_t _independentVariableSize;
    index_t _parameterSize;
};

template <Ungar::Concepts::DenseVectorExpression _XP                   = VectorXr,
          Concepts::TwiceDifferentiableFunction<_XP> _Objective        = Autodiff::Function,
          Concepts::DifferentiableFunction<_XP> _EqualityConstraints   = Autodiff::Function,
          Concepts::DifferentiableFunction<_XP> _InequalityConstraints = Autodiff::Function>
struct NLPProblem {
    _Objective objective;
    _EqualityConstraints equalityConstraints;
    _InequalityConstraints inequalityConstraints;
};

template <class _T>
struct is_nlp_problem : std::false_type {};
template <Ungar::Concepts::DenseVectorExpression _XP,
          Concepts::TwiceDifferentiableFunction<_XP> _Objective,
          Concepts::DifferentiableFunction<_XP> _EqualityConstraints,
          Concepts::DifferentiableFunction<_XP> _InequalityConstraints>
struct is_nlp_problem<NLPProblem<_XP, _Objective, _EqualityConstraints, _InequalityConstraints>>
    : std::true_type {};
template <class _T>
constexpr bool is_nlp_problem_v = is_nlp_problem<_T>::value;

namespace Concepts {

template <typename _NLPProblem>
concept NLPProblem = is_nlp_problem_v<_NLPProblem>;

template <typename _NLPOptimizer, typename _NLPProblem, typename _XP>
concept NLPOptimizer = NLPProblem<_NLPProblem> &&
    requires(_NLPOptimizer optimizer, std::unique_ptr<_NLPProblem> nlp, Eigen::MatrixBase<_XP> xp) {
    { optimizer.Initialize(std::move(nlp)) } -> std::same_as<void>;
    { optimizer.Optimize(xp) } -> Ungar::Concepts::DenseMatrixExpression;
};

}  // namespace Concepts

template <Ungar::Concepts::DenseVectorExpression _XP = VectorXr>
inline auto MakeNLPProblem(Concepts::TwiceDifferentiableFunction<_XP> auto obj,
                           Ungar::Concepts::HanaOptional auto optionalEqs,
                           Ungar::Concepts::HanaOptional auto optionalIneqs) {
    using Objective             = decltype(obj);
    using EqualityConstraints   = std::remove_cvref_t<decltype(optionalEqs.value_or(
        std::declval<IdleTwiceDifferentiableFunction>()))>;
    using InequalityConstraints = std::remove_cvref_t<decltype(optionalIneqs.value_or(
        std::declval<IdleTwiceDifferentiableFunction>()))>;
    using NLPProblemType = NLPProblem<_XP, Objective, EqualityConstraints, InequalityConstraints>;

    const index_t independentVariableSize = obj.IndependentVariableSize();
    const index_t parameterSize           = obj.ParameterSize();

    return std::make_unique<NLPProblemType>(
        std::move(obj),
        std::move(optionalEqs)
            .value_or(IdleTwiceDifferentiableFunction{independentVariableSize, parameterSize}),
        std::move(optionalIneqs)
            .value_or(IdleTwiceDifferentiableFunction{independentVariableSize, parameterSize}));
}

struct FunctionInterface {
    template <typename _XP, typename _Y>
    static void Evaluate(const Concepts::EvaluableFunction<_XP, _Y> auto& function,
                         const Eigen::MatrixBase<_XP>& xp,
                         Eigen::MatrixBase<_Y> const& y) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());
        UNGAR_ASSERT(y.size() == function.DependentVariableSize());

        function.Evaluate(xp, y.const_cast_derived());
    }

    template <typename _XP>
    static auto Invoke(
        const Concepts::EvaluableFunction<_XP, VectorX<typename _XP::Scalar>> auto& function,
        const Eigen::MatrixBase<_XP>&
            xp) requires(!Concepts::DifferentiableFunction<decltype(function), _XP>) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());

        using ScalarType = typename _XP::Scalar;
        VectorX<ScalarType> y{function.DependentVariableSize()};
        function.Evaluate(xp, y);
        return y;
    }

    template <typename _XP>
    static decltype(auto) Invoke(const Concepts::DifferentiableFunction<_XP> auto& function,
                                 const Eigen::MatrixBase<_XP>& xp) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());

        return function(xp);
    }

    template <typename _XP>
    static decltype(auto) Jacobian(const Concepts::DifferentiableFunction<_XP> auto& function,
                                   const Eigen::MatrixBase<_XP>& xp) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());
        return function.Jacobian(xp);
    }

    template <typename _XP>
    static decltype(auto) Hessian(const Concepts::TwiceDifferentiableFunction<_XP> auto& function,
                                  const index_t dependentVariableIndex,
                                  const Eigen::MatrixBase<_XP>& xp) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());
        UNGAR_ASSERT(dependentVariableIndex >= 0_idx &&
                     dependentVariableIndex < function.DependentVariableSize());
        return function.Hessian(dependentVariableIndex, xp);
    }
};

template <Concepts::NLPProblem _NLPProblem, typename _XP, int _P = 2>
typename _XP::Scalar EvaluateConstraintViolation(const _NLPProblem& nlpProblem,
                                                 const Eigen::MatrixBase<_XP>& xp) {
    UNGAR_ASSERT(xp.size() == nlpProblem.objective.IndependentVariableSize() +
                                  nlpProblem.objective.ParameterSize());
    using ScalarType = typename _XP::Scalar;

    VectorX<ScalarType> eqs   = FunctionInterface::Invoke(nlpProblem.equalityConstraints, xp);
    VectorX<ScalarType> ineqs = FunctionInterface::Invoke(nlpProblem.inequalityConstraints, xp);
    ineqs                     = (ineqs.array() < ScalarType{0.0}).select(ScalarType{0.0}, ineqs);

    return Utils::Compose(eqs, ineqs).ToDynamic().template lpNorm<_P>();
}

}  // namespace Ungar

#endif /* _UNGAR__OPTIMIZATION__CONCEPTS_HPP_ */
