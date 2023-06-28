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

#include "ungar/autodiff/function.hpp"

namespace Ungar {
class IdleTwiceDifferentiableFunction {
  public:
    constexpr IdleTwiceDifferentiableFunction(const index_t independentVariableSize,
                                              const index_t parameterSize) noexcept
        : _independentVariableSize{independentVariableSize}, _parameterSize{parameterSize} {
    }

    // clang-format off
    template <typename _XP>
    VectorX<typename _XP::Scalar> operator()(
        [[maybe_unused]] const Eigen::MatrixBase<_XP>& xp) const {  // clang-format on
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);

        using ScalarType = typename _XP::Scalar;
        return VectorX<ScalarType>{};
    }

    template <
        typename _XP,
        typename _Y,
        std::enable_if_t<std::is_same_v<typename _XP::Scalar, typename _Y::Scalar>, bool> = true>
    void Evaluate([[maybe_unused]] const Eigen::MatrixBase<_XP>& xp,
                  [[maybe_unused]] Eigen::MatrixBase<_Y> const& y) const {
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);
        UNGAR_ASSERT(y.size() == 0_idx);
    }

    // clang-format off
    template <typename _XP>
    SparseMatrix<typename _XP::Scalar> Jacobian(
        [[maybe_unused]] const Eigen::MatrixBase<_XP>& xp) const {  // clang-format on
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

template <typename _XP                    = VectorXr,
          typename _Objective             = Autodiff::Function,
          typename _EqualityConstraints   = Autodiff::Function,
          typename _InequalityConstraints = Autodiff::Function>
struct NLPProblem {
    _Objective objective;
    _EqualityConstraints equalityConstraints;
    _InequalityConstraints inequalityConstraints;
};

template <class _T>
struct is_nlp_problem : std::false_type {};
template <typename _XP,
          typename _Objective,
          typename _EqualityConstraints,
          typename _InequalityConstraints>
struct is_nlp_problem<NLPProblem<_XP, _Objective, _EqualityConstraints, _InequalityConstraints>>
    : std::true_type {};
template <class _T>
constexpr bool is_nlp_problem_v = is_nlp_problem<_T>::value;

template <typename _Objective, typename _XP = VectorXr>
inline auto MakeNLPProblem(_Objective obj, decltype(hana::nothing), decltype(hana::nothing)) {
    using Objective             = decltype(obj);
    using EqualityConstraints   = IdleTwiceDifferentiableFunction;
    using InequalityConstraints = IdleTwiceDifferentiableFunction;
    using NLPProblemType = NLPProblem<_XP, Objective, EqualityConstraints, InequalityConstraints>;

    const index_t independentVariableSize = obj.IndependentVariableSize();
    const index_t parameterSize           = obj.ParameterSize();

    return NLPProblemType{std::move(obj),
                          IdleTwiceDifferentiableFunction{independentVariableSize, parameterSize},
                          IdleTwiceDifferentiableFunction{independentVariableSize, parameterSize}};
}

template <typename _Objective, typename _EqualityConstraints, typename _XP = VectorXr>
inline auto MakeNLPProblem(_Objective obj, _EqualityConstraints eqs, decltype(hana::nothing)) {
    using Objective             = decltype(obj);
    using EqualityConstraints   = decltype(eqs);
    using InequalityConstraints = IdleTwiceDifferentiableFunction;
    using NLPProblemType = NLPProblem<_XP, Objective, EqualityConstraints, InequalityConstraints>;

    const index_t independentVariableSize = obj.IndependentVariableSize();
    const index_t parameterSize           = obj.ParameterSize();

    return NLPProblemType{std::move(obj),
                          std::move(eqs),
                          IdleTwiceDifferentiableFunction{independentVariableSize, parameterSize}};
}

template <typename _Objective, typename _InequalityConstraints, typename _XP = VectorXr>
inline auto MakeNLPProblem(_Objective obj, decltype(hana::nothing), _InequalityConstraints ineqs) {
    using Objective             = decltype(obj);
    using EqualityConstraints   = IdleTwiceDifferentiableFunction;
    using InequalityConstraints = decltype(ineqs);
    using NLPProblemType = NLPProblem<_XP, Objective, EqualityConstraints, InequalityConstraints>;

    const index_t independentVariableSize = obj.IndependentVariableSize();
    const index_t parameterSize           = obj.ParameterSize();

    return NLPProblemType{std::move(obj),
                          IdleTwiceDifferentiableFunction{independentVariableSize, parameterSize},
                          std::move(ineqs)};
}

template <typename _Objective,
          typename _EqualityConstraints,
          typename _InequalityConstraints,
          typename _XP = VectorXr>
inline auto MakeNLPProblem(_Objective obj, _EqualityConstraints eqs, _InequalityConstraints ineqs) {
    using Objective             = decltype(obj);
    using EqualityConstraints   = decltype(eqs);
    using InequalityConstraints = decltype(ineqs);
    using NLPProblemType = NLPProblem<_XP, Objective, EqualityConstraints, InequalityConstraints>;

    const index_t independentVariableSize = obj.IndependentVariableSize();
    const index_t parameterSize           = obj.ParameterSize();

    return NLPProblemType{std::move(obj), std::move(eqs), std::move(ineqs)};
}

struct FunctionInterface {
    template <typename _EvaluableFunction, typename _XP, typename _Y>
    static void Evaluate(const _EvaluableFunction& function,
                         const Eigen::MatrixBase<_XP>& xp,
                         Eigen::MatrixBase<_Y> const& y) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());
        UNGAR_ASSERT(y.size() == function.DependentVariableSize());

        function.Evaluate(xp, y.const_cast_derived());
    }

    template <typename _EvaluableFunction, typename _XP>
    static decltype(auto) Invoke(const _EvaluableFunction& function,
                                 const Eigen::MatrixBase<_XP>& xp) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());

        const auto EVALUATE = !hana::is_valid(
            [](const auto& f, const auto& v) -> decltype((void)function(v)) {}, function, xp);
        if constexpr (decltype(EVALUATE)::value) {
            using ScalarType = typename _XP::Scalar;
            VectorX<ScalarType> y{function.DependentVariableSize()};
            function.Evaluate(xp, y);
            return y;
        } else {
            return function(xp);
        }
    }

    template <typename _DifferentiableFunction, typename _XP>
    static decltype(auto) Jacobian(const _DifferentiableFunction& function,
                                   const Eigen::MatrixBase<_XP>& xp) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());
        return function.Jacobian(xp);
    }

    template <typename _DifferentiableFunction, typename _XP>
    static decltype(auto) Hessian(const _DifferentiableFunction& function,
                                  const index_t dependentVariableIndex,
                                  const Eigen::MatrixBase<_XP>& xp) {
        UNGAR_ASSERT(xp.size() == function.IndependentVariableSize() + function.ParameterSize());
        UNGAR_ASSERT(dependentVariableIndex >= 0_idx &&
                     dependentVariableIndex < function.DependentVariableSize());
        return function.Hessian(dependentVariableIndex, xp);
    }
};

template <typename _NLPProblem,
          typename _XP,
          int _P                                                = 2,
          std::enable_if_t<is_nlp_problem_v<_NLPProblem>, bool> = true>
typename _XP::Scalar EvaluateConstraintViolation(const _NLPProblem& nlpProblem,
                                                 const Eigen::MatrixBase<_XP>& xp) {
    UNGAR_ASSERT(xp.size() == nlpProblem.objective.IndependentVariableSize() +
                                  nlpProblem.objective.ParameterSize());
    using ScalarType = typename _XP::Scalar;

    VectorX<ScalarType> eqs   = FunctionInterface::Invoke(nlpProblem.equalityConstraints, xp);
    VectorX<ScalarType> ineqs = FunctionInterface::Invoke(nlpProblem.inequalityConstraints, xp);
    ineqs                     = (ineqs.array() < ScalarType{0.0}).select(ScalarType{0.0}, ineqs);

    return (VectorXr{eqs.size() + ineqs.size()} << eqs, ineqs).finished().template lpNorm<_P>();
}

}  // namespace Ungar

#endif /* _UNGAR__OPTIMIZATION__CONCEPTS_HPP_ */
