/******************************************************************************
 *
 * @file ungar/example/mpc/rc_car.example.cpp
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
 * -----------------------------------------------------------------------
 *
 * @section DESCRIPTION
 *
 * This file implements a nonlinear model predictive controller for
 * the autonomous racing of 1:43 scale RC cars. The dynamical models
 * and physical parameters are based on [1].
 *
 * @see [1] Alexander Liniger, Alexander Domahidi and Manfred Morari.
 *          "Optimization‐based autonomous racing of 1:43 scale RC cars."
 *          Optimal Control Applications and Methods 36 (2015): 628 - 647.
 *
 ******************************************************************************/

#include "ungar/autodiff/vector_composer.hpp"
#include "ungar/optimization/soft_sqp.hpp"
#include "ungar/variable_map.hpp"

int main() {
    using namespace Ungar;

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|                 PART I: RC CAR VARIABLES                 |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /***************     Define numeric invariants as integral constants.     ***************/
    constexpr auto N = 30_c;  // Discrete time horizon.

    /***************                Define decision variables.                ***************/
    // RC cars navigate on flat terrains, so positions are 2-dimensional vectors,
    // orientations are scalars, etc.
    UNGAR_VARIABLE(position, 2);                                         // := p
    UNGAR_VARIABLE(yaw, 1);                                              // := phi
    UNGAR_VARIABLE(b_linear_velocity, 2);                                // := v
    UNGAR_VARIABLE(yaw_rate, 1);                                         // := omega
    UNGAR_VARIABLE(x) <<= (position, yaw, b_linear_velocity, yaw_rate);  // x := [p phi v omega]
    UNGAR_VARIABLE(X) <<= (N + 1_c) * x;                                 // X := [x0 x1 ... xN]

    // The control inputs are the PWM duty cycle of the electric drive train motor and
    // the steering angle.
    UNGAR_VARIABLE(pwm_duty_cycle, 1);                       // := d
    UNGAR_VARIABLE(steering_angle, 1);                       // := delta
    UNGAR_VARIABLE(u) <<= (pwm_duty_cycle, steering_angle);  // u := [d delta]
    UNGAR_VARIABLE(U) <<= N * u;                             // U := [u0 u1 ... uN-1]

    /***************                    Define parameters.                    ***************/
    // Step size.
    UNGAR_VARIABLE(step_size, 1);

    // Inertial and geometric parameters.
    UNGAR_VARIABLE(mass, 1);
    UNGAR_VARIABLE(b_moi, 1);
    UNGAR_VARIABLE(front_wheel_distance, 1);
    UNGAR_VARIABLE(rear_wheel_distance, 1);

    // Simplified Pacejka tire model parameters.
    UNGAR_VARIABLE(ptm_front_b, 1);
    UNGAR_VARIABLE(ptm_front_c, 1);
    UNGAR_VARIABLE(ptm_front_d, 1);
    UNGAR_VARIABLE(ptm_rear_b, 1);
    UNGAR_VARIABLE(ptm_rear_c, 1);
    UNGAR_VARIABLE(ptm_rear_d, 1);
    UNGAR_VARIABLE(ptm_cm1, 1);
    UNGAR_VARIABLE(ptm_cm2, 1);
    UNGAR_VARIABLE(ptm_cr0, 1);
    UNGAR_VARIABLE(ptm_cr2, 1);

    // Reference trajectories.
    UNGAR_VARIABLE(reference_position, 2);
    UNGAR_VARIABLE(reference_trajectory) <<= (N + 1_c) * reference_position;

    // Measurements.
    UNGAR_VARIABLE(measured_position, 2);
    UNGAR_VARIABLE(measured_yaw, 1);
    UNGAR_VARIABLE(b_measured_linear_velocity, 2);
    UNGAR_VARIABLE(measured_yaw_rate, 1);
    UNGAR_VARIABLE(measured_state) <<=
        (measured_position, measured_yaw, b_measured_linear_velocity, measured_yaw_rate);

    /***************                    Define variables.                     ***************/
    UNGAR_VARIABLE(decision_variables) <<= (X, U);
    UNGAR_VARIABLE(parameters) <<= (step_size,
                                    mass,
                                    b_moi,
                                    front_wheel_distance,
                                    rear_wheel_distance,
                                    ptm_front_b,
                                    ptm_front_c,
                                    ptm_front_d,
                                    ptm_rear_b,
                                    ptm_rear_c,
                                    ptm_rear_d,
                                    ptm_cm1,
                                    ptm_cm2,
                                    ptm_cr0,
                                    ptm_cr2,
                                    reference_trajectory,
                                    measured_state);
    UNGAR_VARIABLE(variables) <<= (decision_variables, parameters);

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|                 PART II: RC CAR DYNAMICS                 |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /// @brief Given vectors of autodiff scalars corresponding to the system's state,
    ///        input and parameters at a given time step, compute the state at the
    ///        next time step using a semi-implicit Euler method.
    /***************      Define discrete-time RC car dynamics equation.      ***************/
    const auto rcCarDynamics = [&](const VectorXad& xUnderlying,
                                   const VectorXad& uUnderlying,
                                   const VectorXad& parametersUnderlying) -> VectorXad {
        // Create variable lazy maps for the system's state, input and parameters.
        /// @note As a convention, we name the underlying data representation of a
        ///       variable \c v as \c vUnderlying, and we name \c v_ the associated
        ///       map object.
        const auto x_          = MakeVariableLazyMap(xUnderlying, x);
        const auto u_          = MakeVariableLazyMap(uUnderlying, u);
        const auto parameters_ = MakeVariableLazyMap(parametersUnderlying, parameters);

        // Retrieve all variables.
        const auto& dt = parameters_.Get(step_size);

        const auto [Bf, Cf, Df] = parameters_.GetTuple(ptm_front_b, ptm_front_c, ptm_front_d);
        const auto [Br, Cr, Dr] = parameters_.GetTuple(ptm_rear_b, ptm_rear_c, ptm_rear_d);
        const auto [Cm1, Cm2]   = parameters_.GetTuple(ptm_cm1, ptm_cm2);
        const auto [Cr0, Cr2]   = parameters_.GetTuple(ptm_cr0, ptm_cr2);

        const auto [m, bMOI, lf, lr] =
            parameters_.GetTuple(mass, b_moi, front_wheel_distance, rear_wheel_distance);

        const auto [p, phi, v, omega] = x_.GetTuple(position, yaw, b_linear_velocity, yaw_rate);
        const auto [d, delta]         = u_.GetTuple(pwm_duty_cycle, steering_angle);

        // Compute forces acting on the RC cars.
        /// @note To avoid numerical issues, add a small constant to the numerators.
        const ad_scalar_t alphaf =
            -atan((omega * lf + v.y()) / (v.x() + Eigen::NumTraits<real_t>::epsilon())) + delta;
        const ad_scalar_t alphar =
            atan((omega * lr - v.y()) / (v.x() + Eigen::NumTraits<real_t>::epsilon()));
        const ad_scalar_t Ffy = Df * sin(Cf * atan(Bf * alphaf));
        const ad_scalar_t Fry = Dr * sin(Cr * atan(Br * alphar));
        const ad_scalar_t Frx = (Cm1 - Cm2 * v.x()) * d - Cr0 - Cr2 * Utils::Pow(v.x(), 2);

        // Calculate linear acceleration and angular acceleration of the RC car.
        const Vector2ad vDot{(Frx - Ffy * sin(delta) + m * v.y() * omega) / m,
                             (Fry + Ffy * cos(delta) - m * v.x() * omega) / m};
        const ad_scalar_t omegaDot = (Ffy * lf * cos(delta) - Fry * lr) / bMOI;

        // Create variable map with autodiff scalar type for the next state.
        auto xNext_ = MakeVariableMap<ad_scalar_t>(x);

        auto [pNext, phiNext, vNext, omegaNext] =
            xNext_.GetTuple(position, yaw, b_linear_velocity, yaw_rate);

        // Update next state using semi-implicit Euler method and return underlying data.
        vNext     = v + dt * vDot;
        omegaNext = omega + dt * omegaDot;
        pNext     = p + dt * Vector2ad{vNext.x() * cos(phi) - vNext.y() * sin(phi),
                                   vNext.x() * sin(phi) + vNext.y() * cos(phi)};
        phiNext   = phi + dt * omegaNext;

        return xNext_.Get();
    };

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|            PART III: OPTIMAL CONTROL PROBLEM             |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /// @brief Given vectors of autodiff scalars corresponding to the system's variables,
    ///        implement an optimal control problem (OCP) for the RC car. The objective
    ///        function only comprises position tracking and input regularization goals.
    ///        The equality constraints enforce the system dynamics, while the inequality
    ///        constraints bound the inputs and impose a minimum forward velocity of
    ///        0.3 m/s for the RC car.
    /***************                Define objective function.                ***************/
    const auto objectiveFunction = [&](const VectorXad& variablesUnderlying,
                                       VectorXad& objectiveFunctionUnderlying) {
        // Create variable lazy maps for the system's variables, which include both
        // decision variables and parameters.
        const auto variables_ = MakeVariableLazyMap(variablesUnderlying, variables);

        ad_scalar_t value{0.0};
        for (const auto k : enumerate(N)) {
            const auto p    = variables_.Get(position, k);
            const auto pRef = variables_.Get(reference_position, k);

            const auto uk = variables_.Get(u, k);

            // Reference position tracking.
            value += (p - pRef).squaredNorm();

            // Input regularization.
            value += 1e-6 * uk.squaredNorm();
            if (k) {
                // Regularization of input variations.
                const auto ukm1 = variables_.Get(u, k - 1_step);
                value += 1e-6 * (uk - ukm1).squaredNorm();
            }
        }
        // Final reference position tracking.
        const auto p    = variables_.Get(position, N);
        const auto pRef = variables_.Get(reference_position, N);
        value += (p - pRef).squaredNorm();

        /// @note Autodiff functions must return Eigen vectors, therefore the
        ///       objective function returns a vector of size 1 containing the
        ///       objective value.
        objectiveFunctionUnderlying.resize(1_idx);
        objectiveFunctionUnderlying << value;
    };

    /***************               Define equality constraints.               ***************/
    /// @brief Equality constraints are a function \c g of \c variables such that
    ///        \c g(variables) = 0.
    const auto equalityConstraints = [&](const VectorXad& variablesUnderlying,
                                         VectorXad& equalityConstraintsUnderlying) {
        const auto variables_ = MakeVariableLazyMap(variablesUnderlying, variables);

        // Define helper for composing equality constraints into a single Eigen vector.
        Autodiff::VectorComposer composer;

        // Add equality constraint for the initial state.
        const auto x0 = variables_.Get(x, 0_step);
        const auto xm = variables_.Get(measured_state);
        composer << x0 - xm;

        // Add system dynamics constraint for each time step.
        for (const auto k : enumerate(N)) {
            const auto xk     = variables_.Get(x, k);
            const auto xkp1   = variables_.Get(x, k + 1_step);
            const auto uk     = variables_.Get(u, k);
            const auto params = variables_.Get(parameters);

            composer << xkp1 - rcCarDynamics(xk, uk, params);
        }

        equalityConstraintsUnderlying = composer.Compose();
    };

    /***************              Define inequality constraints.              ***************/
    /// @brief Inequality constraints are a function \c h of \c variables such that
    ///        \c h(variables) <= 0.
    const auto inequalityConstraints = [&](const VectorXad& variablesUnderlying,
                                           VectorXad& inequalityConstraintsUnderlying) {
        const auto variables_ = MakeVariableLazyMap(variablesUnderlying, variables);

        // Define helper for composing inequality constraints into a single Eigen vector.
        Autodiff::VectorComposer composer;

        for (const auto k : enumerate(N)) {
            const auto v      = variables_.Get(b_linear_velocity, k);
            const auto& d     = variables_.Get(pwm_duty_cycle, k);
            const auto& delta = variables_.Get(steering_angle, k);

            // Input bound constraints.
            composer << Utils::Abs(d) - 15.0;
            composer << Utils::Abs(delta) - 15.0;

            // Minimum forward velocity constraint.
            composer << 0.3 - v.x();
        }

        inequalityConstraintsUnderlying = composer.Compose();
    };

    /***************          Define optimal control problem (OCP).           ***************/
    // Based on the autodiff functions defined above, generate code for the
    // corresponding derivatives and compile it just-in-time.
    Autodiff::Function::Blueprint objectiveFunctionBlueprint{objectiveFunction,
                                                             decision_variables.Size(),
                                                             parameters.Size(),
                                                             "rc_car_mpc_obj"sv,
                                                             EnabledDerivatives::ALL};
    Autodiff::Function::Blueprint equalityConstraintsBlueprint{equalityConstraints,
                                                               decision_variables.Size(),
                                                               parameters.Size(),
                                                               "rc_car_mpc_eqs"sv,
                                                               EnabledDerivatives::JACOBIAN};
    Autodiff::Function::Blueprint inequalityConstraintsBlueprint{inequalityConstraints,
                                                                 decision_variables.Size(),
                                                                 parameters.Size(),
                                                                 "rc_car_mpc_ineqs"sv,
                                                                 EnabledDerivatives::JACOBIAN};

    const bool recompileLibraries = true;
    auto ocp =
        MakeNLPProblem(Autodiff::MakeFunction(objectiveFunctionBlueprint, recompileLibraries),
                       Autodiff::MakeFunction(equalityConstraintsBlueprint, recompileLibraries),
                       Autodiff::MakeFunction(inequalityConstraintsBlueprint, recompileLibraries));

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|            PART IV: MODEL PREDICTIVE CONTROL             |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /***************                Initialize OCP variables.                 ***************/
    // Create variable map storing all decision variables and parameters for the RC car.
    auto variables_ = MakeVariableMap<real_t>(variables);

    // Inertial and geometric parameters.
    variables_.Get(mass)                 = 0.041;
    variables_.Get(b_moi)                = 27.8e-6;
    variables_.Get(front_wheel_distance) = 0.029;
    variables_.Get(rear_wheel_distance)  = 0.033;

    variables_.Get(step_size) = 1.0 / static_cast<real_t>(N);

    // Simplified Pacejka tire model parameters.
    variables_.Get(ptm_front_b) = 2.579;
    variables_.Get(ptm_front_c) = 1.2;
    variables_.Get(ptm_front_d) = 0.192;
    variables_.Get(ptm_rear_b)  = 3.3852;
    variables_.Get(ptm_rear_c)  = 1.2691;
    variables_.Get(ptm_rear_d)  = 0.1737;
    variables_.Get(ptm_cm1)     = 0.287;
    variables_.Get(ptm_cm2)     = 0.0545;
    variables_.Get(ptm_cr0)     = 0.0518;
    variables_.Get(ptm_cr2)     = 0.00035;

    // Measurements.
    variables_.Get(measured_position).setZero();
    variables_.Get(measured_yaw) = 0.0;
    variables_.Get(b_measured_linear_velocity).setUnit(0_idx);
    variables_.Get(measured_yaw_rate) = 0.0;

    // Decision variables.
    for (const auto k : enumerate(N + 1_step)) {
        variables_.Get(x, k) = variables_.Get(measured_state);

        const real_t t                  = static_cast<real_t>(k) * variables_.Get(step_size);
        variables_.Get(position, k).x() = variables_.Get(b_measured_linear_velocity).x() * t;
    }
    variables_.Get(U).setZero();

    // Reference trajectories.
    /// @brief Command the RC car to have a constant velocity along the x-axis while
    ///        tracking a sinusoidal trajectory along the y-axis.
    const real_t xDotReference       = 1.0;
    const real_t yPeriodReference    = 8.0;
    const real_t yAmplitudeReference = 0.2;

    /***************             Solve OCP over receding horizon.             ***************/
    // Define OCP optimizer.
    SoftSQPOptimizer optimizer{false, variables_.Get(step_size), 40_idx, 100.0, 1e-2};

    const real_t finalTime = 10.0;
    for (real_t time = 0.0; time < finalTime; time += variables_.Get(step_size)) {
        // Integrate RC car dynamics using the optimized input.
        variables_.Get(measured_state) = Utils::ToRealFunction(rcCarDynamics)(
            variables_.Get(measured_state), variables_.Get(u, 0_step), variables_.Get(parameters));

        // Update reference trajectories.
        for (const auto k : enumerate(N + 1_step)) {
            const real_t t = time + static_cast<real_t>(k) * variables_.Get(step_size);

            variables_.Get(reference_position, k) << xDotReference * t,
                yAmplitudeReference * sin(2.0 * std::numbers::pi / yPeriodReference * t);
        }

        // Shift the previous solution by one time step to warm start the optimization.
        for (const auto k : enumerate(N - 1_step)) {
            variables_.Get(x, k) = variables_.Get(x, k + 1_step);
            variables_.Get(u, k) = variables_.Get(u, k + 1_step);
        }
        variables_.Get(x, N - 1_step) = variables_.Get(x, N);

        // Solve OCP and log optimization results.
        variables_.Get(decision_variables) = optimizer.Optimize(ocp, variables_.Get());
        UNGAR_LOG(info,
                  "t = {:.3f}, obj = {:.3f}, eqs = {:.3f}, ineqs = {:.3f}, p ref = [{:.3f}], p = "
                  "[{:.3f}], d = {:.3f}, delta = {:.3f}",
                  time,
                  Utils::Squeeze(ocp.objective(variables_.Get())),
                  ocp.equalityConstraints(variables_.Get()).lpNorm<Eigen::Infinity>(),
                  ocp.inequalityConstraints(variables_.Get()).maxCoeff(),
                  fmt::join(variables_.Get(reference_position, 0_step), ", "),
                  fmt::join(variables_.Get(position, 0_step), ", "),
                  variables_.Get(pwm_duty_cycle, 0_step),
                  variables_.Get(steering_angle, 0_step));
    }

    return 0;
}
