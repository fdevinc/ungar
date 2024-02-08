/******************************************************************************
 *
 * @file ungar/example/autodiff/function.example.cpp
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

#include "ungar/autodiff/function.hpp"
#include "ungar/variable_map.hpp"

int main() {
    using namespace Ungar;

    /***************                        POINT MASS                        ***************/
    /***************     Define numeric invariants as integral constants.     ***************/
    constexpr auto N = 30_c;  // Discrete time horizon.

    /***************                 Define "leaf" variables.                 ***************/
    // Positions are 3-dimensional vectors, masses are scalars, etc.
    UNGAR_VARIABLE(position, 3);           // := p
    UNGAR_VARIABLE(linear_velocity, 3);    // := pDot
    UNGAR_VARIABLE(force, 3);              // := f
    UNGAR_VARIABLE(mass, 1);               // := m
    UNGAR_VARIABLE(state_cost_weight, 1);  // := wx
    UNGAR_VARIABLE(input_cost_weight, 1);  // := wu

    /***************                Define "branch" variables.                ***************/
    // States are stacked poses and velocities, while inputs are forces applied
    // to the point mass.
    UNGAR_VARIABLE(x) <<= (position, linear_velocity);  // x := [p pDot]
    UNGAR_VARIABLE(u) <<= force;                        // u := f
    UNGAR_VARIABLE(X) <<= (N + 1_c) * x;                // X := [x0 x1 ... xN]
    UNGAR_VARIABLE(U) <<= N * u;                        // U := [u0 u1 ... uN-1]

    UNGAR_VARIABLE(xRef) <<= (position, linear_velocity);  // xRef := [p pDot]
    UNGAR_VARIABLE(XRef) <<= (N + 1_c) * xRef;             // XRef := [xRef0 xRef1 ... xRefN]

    UNGAR_VARIABLE(decision_variables) <<= (X, U);
    UNGAR_VARIABLE(parameters) <<= (mass, XRef, state_cost_weight, input_cost_weight);
    UNGAR_VARIABLE(variables) <<= (decision_variables, parameters);

    /***************   Generate derivatives for functions of variable maps.   ***************/
    const Autodiff::Function::Blueprint functionBlueprint{
        [&](const VectorXad& xp, VectorXad& y) {
            // Create variables maps on existing arrays of data.
            const auto vars = MakeVariableLazyMap(xp, variables);

            // You can further split the stacked independent variables and parameters.
            // const auto decVars =
            //     MakeVariableLazyMap(varsMap.Get(decision_variables), decision_variables);
            // const auto params = MakeVariableLazyMap(varsMap.Get(parameters), parameters);

            y = VectorXr::Zero(1);
            for (auto k : enumerate(N)) {
                y[0] += vars.Get(state_cost_weight) *
                            (vars.Get(x, k) - vars.Get(xRef, k)).squaredNorm() +
                        vars.Get(input_cost_weight) * vars.Get(u, k).squaredNorm();
            }
            y[0] +=
                vars.Get(state_cost_weight) * (vars.Get(x, N) - vars.Get(xRef, N)).squaredNorm();
        },
        decision_variables.Size(),
        parameters.Size(),
        "function_example"sv,
        EnabledDerivatives::JACOBIAN | EnabledDerivatives::HESSIAN};
    const auto function = Autodiff::MakeFunction(functionBlueprint, true);

    auto vars = MakeVariableMap<real_t>(variables);
    vars.Get().setRandom();
    const VectorXr value                = function(vars.Get());
    const SparseMatrix<real_t> jacobian = function.Jacobian(vars.Get());
    const SparseMatrix<real_t> hessian  = function.Hessian(vars.Get());

    UNGAR_ASSERT(value.size() == function.DependentVariableSize() && value.size() == 1);
    UNGAR_ASSERT(function.TestJacobian(vars.Get()));
    UNGAR_ASSERT(function.TestHessian(vars.Get()));

    return 0;
}
