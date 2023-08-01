/******************************************************************************
 *
 * @file ungar/example/mpc/quadruped.example.cpp
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
 * quadrupedal locomotion using the single rigid body dynamics model.
 * This controller is based on [1] but does not linearize the dynamics
 * equations and adopts the Lie group time stepping method described
 * in [2].
 *
 * @see [1] Gerardo Bledt and Sangbae Kim. “Implementing Regularized
 *          Predictive Control for Simultaneous Real-Time Footstep and
 *          Ground Reaction Force Optimization.” 2019 IEEE/RSJ
 *          International Conference on Intelligent Robots and Systems
 *          (IROS) (2019): 6316-6323.
 *      [2] Flavio De Vincenti and Stelian Coros. "Centralized Model
 *          Predictive Control for Collaborative Loco-Manipulation."
 *          Robotics: Science and Systems (2023).
 *
 ******************************************************************************/

#include "ungar/autodiff/vector_composer.hpp"
#include "ungar/optimization/soft_sqp.hpp"
#include "ungar/variable_map.hpp"

int main() {
    using namespace Ungar;

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|                 PART I: QUADRUPED MODEL                  |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /***************     Define numeric invariants as integral constants.     ***************/
    constexpr auto N        = 30_c;  // Discrete time horizon.
    constexpr auto NUM_LEGS = 4_c;

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

    // The control inputs are the stacked ground reaction forces (GRFs) and
    // foot positions, the latter expressed in the base frame.
    UNGAR_VARIABLE(ground_reaction_force, 3);                                // := f
    UNGAR_VARIABLE(b_foot_position, 3);                                      // := r
    UNGAR_VARIABLE(leg_input) <<= (ground_reaction_force, b_foot_position);  // uL = [f r]
    UNGAR_VARIABLE(u) <<= NUM_LEGS * leg_input;  // u := [uL0 uL1 uL2 uL3]
    UNGAR_VARIABLE(U) <<= N * u;                 // U := [u0 u1 ... uN-1]

    /***************             Define time-varying parameters.              ***************/
    // Reference trajectories.
    UNGAR_VARIABLE(reference_position, 3);
    UNGAR_VARIABLE(reference_orientation, Q);
    UNGAR_VARIABLE(reference_linear_velocity, 3);
    UNGAR_VARIABLE(b_reference_angular_velocity, 3);
    UNGAR_VARIABLE(reference_state) <<= (reference_position,
                                         reference_orientation,
                                         reference_linear_velocity,
                                         b_reference_angular_velocity);

    UNGAR_VARIABLE(reference_contact_state, 1);  // := s
    UNGAR_VARIABLE(b_reference_foot_position, 3);
    UNGAR_VARIABLE(reference_leg_state) <<= (reference_contact_state, b_reference_foot_position);

    UNGAR_VARIABLE(p) <<= (reference_state, NUM_LEGS * reference_leg_state);
    UNGAR_VARIABLE(P) <<= (N + 1_c) * p;

    /***************              Define stationary parameters.               ***************/
    // Step size.
    UNGAR_VARIABLE(step_size, 1);

    // Inertial and geometric parameters.
    UNGAR_VARIABLE(mass, 1);
    UNGAR_VARIABLE(b_moi_diagonal, 3);
    UNGAR_VARIABLE(inertial_properties) <<= (mass, b_moi_diagonal);

    UNGAR_VARIABLE(b_hip_position, 3);
    UNGAR_VARIABLE(leg_length, 1);
    UNGAR_VARIABLE(geometric_data) <<= (NUM_LEGS * b_hip_position, leg_length);

    // Physical constants.
    UNGAR_VARIABLE(standard_gravity, 1);
    UNGAR_VARIABLE(friction_coefficient, 1);
    UNGAR_VARIABLE(physical_constants) <<= (standard_gravity, friction_coefficient);

    // Measurements.
    UNGAR_VARIABLE(measured_position, 3);
    UNGAR_VARIABLE(measured_orientation, Q);
    UNGAR_VARIABLE(measured_linear_velocity, 3);
    UNGAR_VARIABLE(b_measured_angular_velocity, 3);
    UNGAR_VARIABLE(measured_state) <<= (measured_position,
                                        measured_orientation,
                                        measured_linear_velocity,
                                        b_measured_angular_velocity);

    UNGAR_VARIABLE(measured_contact_state, 1);
    UNGAR_VARIABLE(measured_foot_position, 3);
    UNGAR_VARIABLE(measured_leg_state) <<= (measured_contact_state, measured_foot_position);

    UNGAR_VARIABLE(Rho) <<= (step_size,
                             inertial_properties,
                             geometric_data,
                             physical_constants,
                             measured_state,
                             NUM_LEGS * measured_leg_state);

    /***************                    Define variables.                     ***************/
    UNGAR_VARIABLE(decision_variables) <<= (X, U);
    UNGAR_VARIABLE(parameters) <<= (P, Rho);
    UNGAR_VARIABLE(variables) <<= (decision_variables, parameters);

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|              PART II: QUADRUPED DYNAMICS                 |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /// @brief Given vectors of autodiff scalars corresponding to the system's state,
    ///        input and parameters at a given time step, compute the state at the
    ///        next time step using a Lie group semi-implicit Euler method.
    /***************    Define discrete-time quadruped dynamics equation.     ***************/
    const auto quadrupedDynamics = [&](const VectorXad& xUnderlying,
                                       const VectorXad& uUnderlying,
                                       const VectorXad& pUnderlying,
                                       const VectorXad& RhoUnderlying) -> VectorXad {
        // Create variable lazy maps for the system's state, input and parameters.
        /// @note As a convention, we name the underlying data representation of a
        ///       variable \c v as \c vUnderlying, and we name \c v_ the associated
        ///       map object.
        const auto x_   = MakeVariableLazyMap(xUnderlying, x);
        const auto u_   = MakeVariableLazyMap(uUnderlying, u);
        const auto p_   = MakeVariableLazyMap(pUnderlying, p);
        const auto Rho_ = MakeVariableLazyMap(RhoUnderlying, Rho);

        // Retrieve all variables.
        const auto [dt, g0, m, bMOIDiagonal] =
            Rho_.GetTuple(step_size, standard_gravity, mass, b_moi_diagonal);

        const auto [p, q, pDot, bOmega] =
            x_.GetTuple(position, orientation, linear_velocity, b_angular_velocity);

        // Calculate linear acceleration and angular acceleration.
        Vector3ad pDotDot   = -g0 * Vector3r::UnitZ();
        Vector3ad bOmegaDot = -bOmega.cross(bMOIDiagonal.asDiagonal() * bOmega);
        for (const auto i : enumerate(NUM_LEGS)) {
            const auto f  = u_.Get(ground_reaction_force, i);
            const auto r  = u_.Get(b_foot_position, i);
            const auto& s = p_.Get(reference_contact_state, i);

            pDotDot += s * f / m;
            bOmegaDot += s * r.cross(q * f);
        }
        bOmegaDot = bOmegaDot.array() / bMOIDiagonal.array();

        // Create variable map with autodiff scalar type for the next state.
        VectorXad xNextUnderlying{x.Size()};
        auto xNext_ = MakeVariableLazyMap(xNextUnderlying, x);
        // Alternatively, the following line creates both the underlying data
        // representation and the variable map: it achieves the best real-time
        // performance at the cost of longer compile time.
        // auto xNext_ = MakeVariableMap<ad_scalar_t>(x);

        auto [pNext, qNext, pDotNext, bOmegaNext] =
            xNext_.GetTuple(position, orientation, linear_velocity, b_angular_velocity);

        // Update next state using Lie group semi-implicit Euler method and
        // return underlying data.
        /// @note For a description of the integration method, refer to [2].
        ///       This approach embeds the quaternion unit norm constraints
        ///       directly into the discretized dynamics equations.
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
            value += (Vector3r{0.1, 0.1, 10.0}.cwiseProduct(p - pRef).squaredNorm() +
                      Utils::Min((q.coeffs() - qRef.coeffs()).squaredNorm(),
                                 (q.coeffs() + qRef.coeffs()).squaredNorm()) +
                      (pDot - pDotRef).squaredNorm() + (bOmega - bOmegaRef).squaredNorm());
            if (k != N) {
                // Input regularization.
                for (const auto i : enumerate(NUM_LEGS)) {
                    const auto f = variables_.Get(ground_reaction_force, k, i);
                    const auto r = variables_.Get(b_foot_position, k, i);

                    const auto rRef = variables_.Get(b_reference_foot_position, k, i);

                    value += (r - rRef).squaredNorm();
                    value += 1e-8 * f.squaredNorm();
                }
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
            const auto xk   = variables_.Get(x, k);
            const auto xkp1 = variables_.Get(x, k + 1_step);
            const auto uk   = variables_.Get(u, k);
            const auto pk   = variables_.Get(p, k);
            const auto rho  = variables_.Get(Rho);

            composer << xkp1 - quadrupedDynamics(xk, uk, pk, rho);
        }

        for (const auto k : enumerate(N)) {
            for (const auto i : enumerate(NUM_LEGS)) {
                const auto& s     = variables_.Get(reference_contact_state, k, i);
                const auto& sPrev = k ? variables_.Get(reference_contact_state, k - 1_step, i)
                                      : variables_.Get(measured_contact_state, i);

                const auto p          = variables_.Get(position, k);
                const auto q          = variables_.Get(orientation, k);
                const auto r          = variables_.Get(b_foot_position, k, i);
                const Vector3ad pFoot = p + q * r;

                Vector3ad pFootPrev;
                if (k) {
                    const auto pPrev = variables_.Get(position, k - 1_step);
                    const auto qPrev = variables_.Get(orientation, k - 1_step);
                    const auto rPrev = variables_.Get(b_foot_position, k - 1_step, i);

                    pFootPrev = pPrev + qPrev * rPrev;
                } else {
                    pFootPrev = variables_.Get(measured_foot_position, i);
                }

                composer << (1.0 - sPrev) * s * pFoot.z();
                composer << sPrev * s * (pFoot - pFootPrev);
            }
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

        const auto& mu = variables_.Get(friction_coefficient);

        for (const auto k : enumerate(N)) {
            const auto p = variables_.Get(position, k);
            const auto q = variables_.Get(orientation, k);

            for (const auto i : enumerate(NUM_LEGS)) {
                const auto& s = variables_.Get(reference_contact_state, k, i);
                const auto f  = variables_.Get(ground_reaction_force, k, i);
                const auto r  = variables_.Get(b_foot_position, k, i);

                composer << -s * f.z();
                composer << s * Utils::ApproximateNorm(f.head<2>()) - mu * f.z();
                composer << s * Utils::ApproximateNorm(r - variables_.Get(b_hip_position, i)) -
                                variables_.Get(leg_length);
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
                                                             "quadruped_mpc_obj"sv,
                                                             EnabledDerivatives::ALL};
    Autodiff::Function::Blueprint equalityConstraintsBlueprint{equalityConstraints,
                                                               decision_variables.Size(),
                                                               parameters.Size(),
                                                               "quadruped_mpc_eqs"sv,
                                                               EnabledDerivatives::JACOBIAN};
    Autodiff::Function::Blueprint inequalityConstraintsBlueprint{inequalityConstraints,
                                                                 decision_variables.Size(),
                                                                 parameters.Size(),
                                                                 "quadruped_mpc_ineqs"sv,
                                                                 EnabledDerivatives::JACOBIAN};

    const bool recompileLibraries = false;
    auto ocp =
        MakeNLPProblem(Autodiff::MakeFunction(objectiveFunctionBlueprint, recompileLibraries),
                       Autodiff::MakeFunction(equalityConstraintsBlueprint, recompileLibraries),
                       Autodiff::MakeFunction(inequalityConstraintsBlueprint, recompileLibraries));

    /*======================================================================================*/
    /*~~~~~~~~~~~~~|            PART IV: MODEL PREDICTIVE CONTROL             |~~~~~~~~~~~~~*/
    /*======================================================================================*/
    /***************                Initialize OCP variables.                 ***************/
    // Create variable map storing all decision variables and parameters for the quadruped.
    VectorXr variablesUnderlying{variables.Size()};
    auto variables_ = MakeVariableLazyMap(variablesUnderlying, variables);
    // Alternatively, the following line creates both the underlying data
    // representation and the variable map: it achieves the best real-time
    // performance at the cost of longer compile time.
    // auto variables_ = MakeVariableMap<real_t>(variables);

    // Step size.
    variables_.Get(step_size) = 1.0 / static_cast<real_t>(N);

    // Inertial and geometric parameters.
    variables_.Get(mass)           = 25.0;
    variables_.Get(b_moi_diagonal) = Vector3r{0.048125, 0.093125, 0.055625};

    variables_.Get(b_hip_position, 0_idx) = Vector3r{0.2, 0.15, -0.1};    // RF
    variables_.Get(b_hip_position, 1_idx) = Vector3r{0.2, -0.15, -0.1};   // LF
    variables_.Get(b_hip_position, 2_idx) = Vector3r{-0.2, 0.15, -0.1};   // RH
    variables_.Get(b_hip_position, 3_idx) = Vector3r{-0.2, -0.15, -0.1};  // LH
    variables_.Get(leg_length)            = 0.42;

    // Physical constants.
    variables_.Get(standard_gravity)     = 9.80665;
    variables_.Get(friction_coefficient) = 0.7;

    // Measurements.
    const real_t initialHeight        = 0.38;
    variables_.Get(measured_position) = initialHeight * Vector3r::UnitZ();
    variables_.Get(measured_orientation).setIdentity();
    variables_.Get(measured_linear_velocity).setZero();
    variables_.Get(b_measured_angular_velocity).setZero();

    variables_.Get(measured_contact_state, 0_idx) = 1.0;
    variables_.Get(measured_foot_position, 0_idx) = Vector3r(0.2, 0.1, 0.0);
    variables_.Get(measured_contact_state, 1_idx) = 1.0;
    variables_.Get(measured_foot_position, 1_idx) = Vector3r(0.2, -0.1, 0.0);
    variables_.Get(measured_contact_state, 2_idx) = 1.0;
    variables_.Get(measured_foot_position, 2_idx) = Vector3r(-0.2, 0.1, 0.0);
    variables_.Get(measured_contact_state, 3_idx) = 1.0;
    variables_.Get(measured_foot_position, 3_idx) = Vector3r(-0.2, -0.1, 0.0);

    // Decision variables.
    for (const auto k : enumerate(N + 1_step)) {
        variables_.Get(x, k) = variables_.Get(measured_state);
    }
    for (const auto k : enumerate(N)) {
        for (const auto i : enumerate(NUM_LEGS)) {
            variables_.Get(ground_reaction_force, k, i) =
                variables_.Get(mass) * variables_.Get(standard_gravity) / 4.0 * Vector3r::UnitZ();
            variables_.Get(b_foot_position, k, i) = variables_.Get(measured_orientation).inverse() *
                                                    variables_.Get(measured_foot_position, i);
        }
    }

    // Reference trajectories.
    /// @brief Command the quadruped to track a sinusoidal trajectory along the z-axis
    ///        while rotating about the z-axis using a pace gait.
    const real_t zPeriodReference    = 8.0;
    const real_t zAmplitudeReference = 0.03;
    const real_t yawRateReference    = std::numbers::pi / 6.0;

    const real_t missionStartTime = 2.0;
    const real_t paceGaitPeriod   = 0.4;

    /***************             Solve OCP over receding horizon.             ***************/
    // Define OCP optimizer.
    SoftSQPOptimizer optimizer{false, variables_.Get(step_size), 4_idx, 1.0, 1.0};

    const real_t finalTime = 10.0;
    for (real_t time = 0.0; time < finalTime; time += variables_.Get(step_size)) {
        // Integrate quadruped dynamics using the optimized input.
        variables_.Get(measured_state) =
            Utils::ToRealFunction(quadrupedDynamics)(variables_.Get(measured_state),
                                                     variables_.Get(u, 0_step),
                                                     variables_.Get(p, 0_step),
                                                     variables_.Get(Rho));
        for (const auto i : enumerate(NUM_LEGS)) {
            variables_.Get(measured_foot_position, i) =
                variables_.Get(measured_position) +
                variables_.Get(measured_orientation) * variables_.Get(b_foot_position, 1_step, i);
            variables_.Get(measured_contact_state, i) =
                variables_.Get(reference_contact_state, 1_step, i);
        }

        // Update reference trajectories.
        for (const auto k : enumerate(N + 1_step)) {
            const real_t t              = time + static_cast<real_t>(k) * variables_.Get(step_size);
            const real_t missionStarted = static_cast<real_t>(t > missionStartTime);

            variables_.Get(reference_position, k) =
                (initialHeight +
                 missionStarted * zAmplitudeReference *
                     sin(2.0 * std::numbers::pi / zPeriodReference * (t - missionStartTime))) *
                Vector3r::UnitZ();
            variables_.Get(reference_orientation, k) =
                missionStarted
                    ? Utils::ElementaryZQuaternion(yawRateReference * (t - missionStartTime))
                    : Quaternionr::Identity();
            variables_.Get(reference_linear_velocity, k) =
                missionStarted * 2.0 * std::numbers::pi / zPeriodReference * zAmplitudeReference *
                cos(2.0 * std::numbers::pi / zPeriodReference * (t - missionStartTime)) *
                Vector3r::UnitZ();
            variables_.Get(b_reference_angular_velocity, k) =
                missionStarted * yawRateReference * Vector3r::UnitZ();

            for (const bool paceGaitPhase = sin(2.0 * std::numbers::pi / paceGaitPeriod * t) > 0.0;
                 const auto i : enumerate(NUM_LEGS)) {
                const auto odd = [](const index_t n) { return static_cast<bool>(n & 0b1); };
                variables_.Get(reference_contact_state, k, i) =
                    missionStarted ? static_cast<real_t>(odd(i) ? paceGaitPhase : !paceGaitPhase)
                                   : 1.0;
                variables_.Get(b_reference_foot_position, k, i) =
                    variables_.Get(b_hip_position, i) -
                    0.8 * variables_.Get(leg_length) * Vector3r::UnitZ();
            }
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
        index_t maxCoeffIndex;
        const real_t maxCoeff =
            ocp.inequalityConstraints(variables_.Get()).maxCoeff(&maxCoeffIndex);
        UNGAR_LOG(
            info,
            "t = {:.3f}, obj = {:.3f}, eqs = {:.3f}, ineqs = {:.3f} ({}), z ref = {:.3f}, z = "
            "{:.3f}, yaw ref = {:.3f}, yaw = {:.3f}, grf z = {:.3f}",
            time,
            Utils::Squeeze(ocp.objective(variables_.Get())),
            ocp.equalityConstraints(variables_.Get()).lpNorm<Eigen::Infinity>(),
            maxCoeff,
            maxCoeffIndex,
            variables_.Get(reference_position, 0_step).z(),
            variables_.Get(position, 0_step).z(),
            Utils::QuaternionToYawPitchRoll(variables_.Get(reference_orientation, 0_step)).z(),
            Utils::QuaternionToYawPitchRoll(variables_.Get(orientation, 0_step)).z(),
            fmt::join(enumerate(NUM_LEGS) | std::views::transform([&](const auto i) {
                          return variables_.Get(reference_contact_state, 0_step, i) *
                                 variables_.Get(ground_reaction_force, 0_step, i).z();
                      }),
                      ", "));
    }

    return 0;
}
