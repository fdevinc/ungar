/******************************************************************************
 *
 * @file ungar/example/mpc/quadrotor.example.cpp
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
 * a quadrotor living in SE(3). The dynamical model of the quadrotor
 * is adapted from the lecture notes of the course "Robot Dynamics"
 * [1] held at ETH Zurich.
 *
 * @see [1] Marco Hutter, Roland Siegwart, Class Lecture, Topic:
 *          "Dynamic Modeling of Rotorcraft & Control." 151-0851-00L.
 *          Department of Mechanical and Process Engineering,
 *          ETH Zurich, Zurich, 2022.
 *
 ******************************************************************************/

#include "ungar/autodiff/vector_composer.hpp"
#include "ungar/optimization/soft_sqp.hpp"
#include "ungar/variable_map.hpp"

int main() {
    using namespace Ungar;

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|                 PART I: QUADROTOR MODEL                  |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /***************     Define numeric invariants as integral constants.     ***************/
    constexpr auto N          = 30_c;  // Discrete time horizon.
    constexpr auto NUM_ROTORS = 4_c;

    /***************                Define decision variables.                ***************/
    // Positions are 3-dimensional vectors, orientations are unit quaternions,
    // etc. The states are stacked poses and velocities.
    UNGAR_VARIABLE(position, 3);            // := p
    UNGAR_VARIABLE(orientation, Q);         // := q
    UNGAR_VARIABLE(linear_velocity, 3);     // := pDot
    UNGAR_VARIABLE(b_angular_velocity, 3);  // := bOmega
    UNGAR_VARIABLE(x) <<=
        (position, orientation, linear_velocity, b_angular_velocity);  // x := [p q pDot bOmega]
    UNGAR_VARIABLE(X) <<= (N + 1_c) * x;                               // X := [x0 x1 ... xN]

    // The control inputs are the stacked rotor speeds, one for each rotor.
    UNGAR_VARIABLE(rotor_speed, 1);                  // := r
    UNGAR_VARIABLE(u) <<= NUM_ROTORS * rotor_speed;  // u := [r0 r1 r2 r3]
    UNGAR_VARIABLE(U) <<= N * u;                     // U := [u0 u1 ... uN-1]

    /***************                    Define parameters.                    ***************/
    // Step size.
    UNGAR_VARIABLE(step_size, 1);

    // Inertial and geometric parameters.
    UNGAR_VARIABLE(mass, 1);
    UNGAR_VARIABLE(b_moi_diagonal, 3);
    UNGAR_VARIABLE(b_propeller_position, 3);

    // Physical constants and general parameters.
    UNGAR_VARIABLE(standard_gravity, 1);
    UNGAR_VARIABLE(thrust_constant, 1);
    UNGAR_VARIABLE(drag_constant, 1);
    UNGAR_VARIABLE(max_rotor_speed, 1);

    // Reference trajectories.
    UNGAR_VARIABLE(reference_position, 3);
    UNGAR_VARIABLE(reference_orientation, Q);
    UNGAR_VARIABLE(reference_linear_velocity, 3);
    UNGAR_VARIABLE(b_reference_angular_velocity, 3);

    // Measurements.
    UNGAR_VARIABLE(measured_position, 3);
    UNGAR_VARIABLE(measured_orientation, Q);
    UNGAR_VARIABLE(measured_linear_velocity, 3);
    UNGAR_VARIABLE(b_measured_angular_velocity, 3);
    UNGAR_VARIABLE(measured_state) <<= (measured_position,
                                        measured_orientation,
                                        measured_linear_velocity,
                                        b_measured_angular_velocity);

    /***************                    Define variables.                     ***************/
    UNGAR_VARIABLE(decision_variables) <<= (X, U);
    UNGAR_VARIABLE(parameters) <<= (step_size,
                                    mass,
                                    b_moi_diagonal,
                                    NUM_ROTORS * b_propeller_position,
                                    standard_gravity,
                                    thrust_constant,
                                    drag_constant,
                                    max_rotor_speed,
                                    (N + 1_c) * reference_position,
                                    (N + 1_c) * reference_orientation,
                                    (N + 1_c) * reference_linear_velocity,
                                    (N + 1_c) * b_reference_angular_velocity,
                                    measured_state);
    UNGAR_VARIABLE(variables) <<= (decision_variables, parameters);

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|              PART II: QUADROTOR DYNAMICS                 |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /// @brief Given vectors of autodiff scalars corresponding to the system's state,
    ///        input and parameters at a given time step, compute the state at the
    ///        next time step using a Lie group semi-implicit Euler method.
    /***************    Define discrete-time quadrotor dynamics equation.     ***************/
    const auto quadrotorDynamics = [&](const VectorXad& xUnderlying,
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
        const auto [dt, g0, b, d] =
            parameters_.GetTuple(step_size, standard_gravity, thrust_constant, drag_constant);
        const auto [m, bMOIDiagonal] = parameters_.GetTuple(mass, b_moi_diagonal);

        const auto [p, q, pDot, bOmega] =
            x_.GetTuple(position, orientation, linear_velocity, b_angular_velocity);

        // Calculate thrust forces and drag moments for each rotor.
        // Thrust force:    bTi = b * ri^2 * ez
        // Thrust moment:   bMi = pPi x bTi
        // Drag moment:     bD  = d * ri^2 * ez * (-1)^i
        std::vector<Vector3ad> bThrustForces{}, bThrustMoments{}, bDragMoments{};
        for (const auto i : enumerate(NUM_ROTORS)) {
            const auto& r = u_.Get(rotor_speed, i);
            const auto pP = parameters_.Get(b_propeller_position, i);

            bThrustForces.emplace_back(b * Utils::Pow(r, 2) * Vector3ad::UnitZ());
            bThrustMoments.emplace_back(pP.cross(bThrustForces.back()));
            bDragMoments.emplace_back(d * Utils::Pow(r, 2) * Vector3ad::UnitZ() *
                                      Utils::Pow(-1.0, i));
        }

        // Calculate linear acceleration and angular acceleration.
        const auto summation = [](const std::vector<Vector3ad>& seq) {
            return std::accumulate(seq.begin(), seq.end(), Vector3ad::Zero().eval());
        };
        const Vector3ad pDotDot = (q * summation(bThrustForces) - m * g0 * Vector3ad::UnitZ()) / m;
        const Vector3ad bOmegaDot = bMOIDiagonal.cwiseInverse().cwiseProduct(
            summation(bThrustMoments) + summation(bDragMoments) -
            bOmega.cross(bMOIDiagonal.cwiseProduct(bOmega)));

        // Create variable map with autodiff scalar type for the next state.
        auto xNext_ = MakeVariableMap<ad_scalar_t>(x);

        auto [pNext, qNext, pDotNext, bOmegaNext] =
            xNext_.GetTuple(position, orientation, linear_velocity, b_angular_velocity);

        // Update next state using Lie group semi-implicit Euler method and
        // return underlying data.
        /// @note For a description of the integration method, refer to [2].
        ///       This approach embeds the quaternion unit norm constraints
        ///       directly into the discretized dynamics equations.
        ///
        /// @see [2] Flavio De Vincenti and Stelian Coros. "Centralized Model
        ///          Predictive Control for Collaborative Loco-Manipulation."
        ///          Robotics: Science and Systems (2023).
        pDotNext   = pDot + dt * pDotDot;
        bOmegaNext = bOmega + dt * bOmegaDot;
        pNext      = p + dt * pDotNext;
        qNext      = q * Utils::ApproximateExponentialMap(dt * bOmegaNext);

        return xNext_.Get();
    };

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|            PART III: OPTIMAL CONTROL PROBLEM             |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /***************                Define objective function.                ***************/
    const auto objectiveFunction = [&](const VectorXad& variablesUnderlying,
                                       VectorXad& objectiveFunctionUnderlying) {
        // Create variable lazy maps for the system's variables, which include both
        // decision variables and parameters.
        const auto variables_ = MakeVariableLazyMap(variablesUnderlying, variables);

        ad_scalar_t value{0.0};
        for (const auto k : enumerate(N + 1_step)) {
            const auto p      = variables_.Get(position, k);
            const auto q      = variables_.Get(orientation, k);
            const auto pDot   = variables_.Get(linear_velocity, k);
            const auto bOmega = variables_.Get(b_angular_velocity, k);

            const auto pRef      = variables_.Get(reference_position, k);
            const auto qRef      = variables_.Get(reference_orientation, k);
            const auto pDotRef   = variables_.Get(reference_linear_velocity, k);
            const auto bOmegaRef = variables_.Get(b_reference_angular_velocity, k);

            const auto uk = variables_.Get(u, k);

            // Reference state tracking.
            value += ((p - pRef).squaredNorm() +
                      Utils::Min((q.coeffs() - qRef.coeffs()).squaredNorm(),
                                 (q.coeffs() + qRef.coeffs()).squaredNorm()) +
                      (pDot - pDotRef).squaredNorm() + (bOmega - bOmegaRef).squaredNorm());
            if (k) {
                // Regularization of input variations.
                const auto ukm1 = variables_.Get(u, k - 1_step);

                value += 1e-6 * (uk - ukm1).squaredNorm();
            }
            if (k != N) {
                // Input regularization.
                value += 1e-6 * uk.squaredNorm();
            }
        }

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

        // Add equality constraints for the initial state.
        const auto x0 = variables_.Get(x, 0_step);
        const auto xm = variables_.Get(measured_state);
        composer << x0 - xm;

        // Add system dynamics constraint for each time step.
        for (const auto k : enumerate(N)) {
            const auto xk     = variables_.Get(x, k);
            const auto xkp1   = variables_.Get(x, k + 1_step);
            const auto uk     = variables_.Get(u, k);
            const auto params = variables_.Get(parameters);

            composer << xkp1 - quadrotorDynamics(xk, uk, params);
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

        const auto& rMax = variables_.Get(max_rotor_speed);

        for (const auto k : enumerate(N)) {
            for (const auto i : enumerate(NUM_ROTORS)) {
                const auto r = variables_.Get(rotor_speed, k, i);

                // Input bound constraints.
                composer << r - rMax;
                composer << -r;
            }
        }

        inequalityConstraintsUnderlying = composer.Compose();
    };

    /***************          Define optimal control problem (OCP).           ***************/
    // Based on the autodiff functions defined above, generate code for the
    // corresponding derivatives and compile it just-in-time.
    Autodiff::Function::Blueprint objectiveFunctionBlueprint{objectiveFunction,
                                                             decision_variables.Size(),
                                                             parameters.Size(),
                                                             "quadrotor_mpc_obj"sv,
                                                             EnabledDerivatives::ALL};
    Autodiff::Function::Blueprint equalityConstraintsBlueprint{equalityConstraints,
                                                               decision_variables.Size(),
                                                               parameters.Size(),
                                                               "quadrotor_mpc_eqs"sv,
                                                               EnabledDerivatives::JACOBIAN};
    Autodiff::Function::Blueprint inequalityConstraintsBlueprint{inequalityConstraints,
                                                                 decision_variables.Size(),
                                                                 parameters.Size(),
                                                                 "quadrotor_mpc_ineqs"sv,
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
    // Create variable map storing all decision variables and parameters for the quadrotor.
    auto variables_ = MakeVariableMap<real_t>(variables);

    // Step size.
    variables_.Get(step_size) = 1.0 / static_cast<real_t>(N);

    // Inertial and geometric parameters.
    variables_.Get(mass) = 1.5;
    variables_.Get(b_moi_diagonal).setConstant(3e-2);

    // Propeller positions with respect to the quadrotor's center of mass.
    // These values represent a square-shaped quadrotor configuration.
    variables_.Get(b_propeller_position, 0_idx) = Vector3r(0.2, 0.2, 0.0);
    variables_.Get(b_propeller_position, 1_idx) = Vector3r(-0.2, 0.2, 0.0);
    variables_.Get(b_propeller_position, 2_idx) = Vector3r(-0.2, -0.2, 0.0);
    variables_.Get(b_propeller_position, 3_idx) = Vector3r(0.2, -0.2, 0.0);

    // Physical constants.
    variables_.Get(standard_gravity) = 9.80665;
    variables_.Get(thrust_constant)  = 0.015;
    variables_.Get(drag_constant)    = 0.1;
    variables_.Get(max_rotor_speed)  = 1e2;

    // Measurements.
    const real_t initialHeight        = 4.0;
    variables_.Get(measured_position) = initialHeight * Vector3r::UnitZ();
    variables_.Get(measured_orientation).setIdentity();
    variables_.Get(measured_linear_velocity).setZero();
    variables_.Get(b_measured_angular_velocity).setZero();

    // Decision variables.
    for (const auto k : enumerate(N + 1_step)) {
        variables_.Get(x, k) = variables_.Get(measured_state);
    }
    variables_.Get(U).setConstant(
        Utils::Sqrt(variables_.Get(mass) * variables_.Get(standard_gravity) /
                    variables_.Get(thrust_constant) / static_cast<real_t>(NUM_ROTORS)));

    // Reference trajectories.
    /// @brief Command the quadrotor to track a sinusoidal trajectory along the z-axis
    ///        while rotating about the z-axis.
    const real_t zPeriodReference    = 4.0;
    const real_t zAmplitudeReference = 1.0;
    const real_t yawRateReference    = std::numbers::pi;
    const real_t missionStartTime    = zPeriodReference;

    /***************             Solve OCP over receding horizon.             ***************/
    // Define OCP optimizer.
    SoftSQPOptimizer optimizer{false, default_value, 40_idx};

    const real_t finalTime = 10.0;
    for (real_t time = 0.0; time < finalTime; time += variables_.Get(step_size)) {
        // Integrate quadrotor dynamics using the optimized input.
        variables_.Get(measured_state) = Utils::ToRealFunction(quadrotorDynamics)(
            variables_.Get(measured_state), variables_.Get(u, 0_step), variables_.Get(parameters));

        // Update reference trajectories.
        for (const auto k : enumerate(N + 1_step)) {
            const real_t t              = time + static_cast<real_t>(k) * variables_.Get(step_size);
            const real_t missionStarted = static_cast<real_t>(t > missionStartTime);

            variables_.Get(reference_position, k) =
                (initialHeight + missionStarted * zAmplitudeReference *
                                     sin(2.0 * std::numbers::pi / zPeriodReference * t)) *
                Vector3r::UnitZ();
            variables_.Get(reference_orientation, k) =
                missionStarted ? Utils::ElementaryZQuaternion(yawRateReference * t)
                               : Quaternionr::Identity();
            variables_.Get(reference_linear_velocity, k) =
                missionStarted * 2.0 * std::numbers::pi / zPeriodReference * zAmplitudeReference *
                cos(2.0 * std::numbers::pi / zPeriodReference * t) * Vector3r::UnitZ();
            variables_.Get(b_reference_angular_velocity, k) =
                missionStarted * yawRateReference * Vector3r::UnitZ();
        }

        // Shift the previous solution by one time step to warm start the optimization.
        for (const auto k : enumerate(N - 1_step)) {
            variables_.Get(x, k) = variables_.Get(x, k + 1_step);
            variables_.Get(u, k) = variables_.Get(u, k + 1_step);
        }
        variables_.Get(x, N - 1_step) = variables_.Get(x, N);

        /// @note This trick prevents quaternion sign switches from
        ///       spoiling the optimization. Find more details in [2].
        const auto& qm = std::as_const(variables_).Get(measured_orientation);
        const auto& q0 = std::as_const(variables_).Get(orientation, 0_step);
        if ((qm.coeffs() - q0.coeffs()).squaredNorm() >
            (-qm.coeffs() - q0.coeffs()).squaredNorm()) {
            variables_.Get(measured_orientation).coeffs() *= -1.0;
        }

        // Solve OCP and log optimization results.
        variables_.Get(decision_variables) = optimizer.Optimize(ocp, variables_.Get());
        UNGAR_LOG(
            info,
            "t = {:.3f}, obj = {:.3f}, eqs = {:.3f}, ineqs = {:.3f}, z ref = {:.3f}, z = "
            "{:.3f}, yaw ref = {:.3f}, yaw = {:.3f}, u = {:.3f}",
            time,
            Utils::Squeeze(ocp.objective(variables_.Get())),
            ocp.equalityConstraints(variables_.Get()).lpNorm<Eigen::Infinity>(),
            ocp.inequalityConstraints(variables_.Get()).maxCoeff(),
            variables_.Get(reference_position, 0_step).z(),
            variables_.Get(position, 0_step).z(),
            Utils::QuaternionToYawPitchRoll(variables_.Get(reference_orientation, 0_step)).z(),
            Utils::QuaternionToYawPitchRoll(variables_.Get(orientation, 0_step)).z(),
            fmt::join(variables_.Get(u, 0_step), ", "));
    }

    return 0;
}
