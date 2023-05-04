/******************************************************************************
 *
 * @file ungar/example/variable.example.cpp
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

#include "ungar/variable.hpp"

int main() {
    using namespace Ungar;

    /***************                 SINGLE RIGID BODY MODEL                  ***************/
    /***************     Define numeric invariants as integral constants.     ***************/
    constexpr auto N        = 30_c;  // Discrete time horizon.
    constexpr auto NUM_LEGS = 4_c;

    /***************                 Define "leaf" variables.                 ***************/
    // Positions are 3-dimensional vectors, orientations are unit quaternions, etc.
    UNGAR_VARIABLE(position, 3);          // := p
    UNGAR_VARIABLE(orientation, Q);       // := q
    UNGAR_VARIABLE(linear_velocity, 3);   // := pDot
    UNGAR_VARIABLE(angular_velocity, 3);  // := omega

    UNGAR_VARIABLE(relative_position, 3);  // := r
    UNGAR_VARIABLE(force, 3);              // := f

    /***************                Define "branch" variables.                ***************/
    // States are stacked poses and velocities, while inputs are ground reaction
    // forces and relative positions for each foot.
    UNGAR_VARIABLE(x) <<=
        (position, orientation, linear_velocity, angular_velocity);  // x := [p q pDot omega]
    UNGAR_VARIABLE(leg_input) <<= (force, relative_position);        // lu := [f r]
    UNGAR_VARIABLE(u) <<= NUM_LEGS * leg_input;                      // u := [lu0 lu1 lu2 lu3]
    UNGAR_VARIABLE(X) <<= (N + 1_c) * x;                             // X := [x0 x1 ... xN]
    UNGAR_VARIABLE(U) <<= N * u;                                     // U := [u0 u1 ... uN-1]

    UNGAR_VARIABLE(variables) <<= (X, U);

    /***************   Retrieve information about variables at compile time.  ***************/
    static_assert(position.Size() == 3);
    static_assert(orientation.Size() == 4);
    static_assert(x.Size() == position.Size() + orientation.Size() + linear_velocity.Size() +
                                  angular_velocity.Size());
    static_assert(X.Size() == x.Size() * (N + 1));
    static_assert(variables.Size() == X.Size() + U.Size());

    // Access subvariables using an intuitive syntax.
    static_assert(variables(x, 0).Index() == 0);         // [{x0} x1  x2  ...  xN  u0  ... ]
    static_assert(variables(x, 1).Index() == 13);        // [ x0 {x1} x2  ...  xN  u0  ... ]
    static_assert(variables(x, 2).Index() == 26);        // [ x0  x1 {x2} ...  xN  u0  ... ]
    static_assert(variables(u, 0).Index() == X.Size());  // [ x0  x1  x2  ...  xN {u0} ... ]

    static_assert(variables(u, 0, leg_input, 0).Index() ==
                  X.Size());  // [ X {lu00} lu01  lu02  lu03  lu10  lu11  lu12  lu13  ... ]
    static_assert(variables(u, 0, leg_input, 1).Index() ==
                  X.Size() + 6);  // [ X  lu00 {lu01} lu02  lu03  lu10  lu11  lu12  lu13  ... ]
    static_assert(variables(u, 0, leg_input, 2).Index() ==
                  X.Size() + 12);  // [ X  lu00  lu01 {lu02} lu03  lu10  lu11  lu12  lu13  ... ]
    static_assert(variables(u, 1, leg_input, 3).Index() ==
                  X.Size() + u.Size() +
                      18);  // [ X  lu00  lu01  lu02  lu03  lu10  lu11  lu12 {lu13} ... ]

    // Use concise notations to select subvariables when there is no ambiguity.
    static_assert(variables(u, 0, leg_input, 0).Index() == variables(leg_input, 0, 0).Index());
    static_assert(variables(u, 0, leg_input, 1).Index() == variables(leg_input, 0, 1).Index());
    static_assert(variables(u, 0, leg_input, 2).Index() == variables(leg_input, 0, 2).Index());
    static_assert(variables(u, 1, leg_input, 3).Index() == variables(leg_input, 1, 3).Index());

    static_assert(variables(u, 4, leg_input, 0, force).Index() == variables(force, 4, 0).Index());
    static_assert(variables(u, 4, leg_input, 1, force).Index() == variables(force, 4, 1).Index());
    static_assert(variables(u, 4, leg_input, 2, force).Index() == variables(force, 4, 2).Index());

    // Use verbose notations if you prefer.
    static_assert(X.At<"x">(0).Index() == X(x, 0).Index());
    static_assert(X.At<"x">(1).Index() == X(x, 1).Index());
    static_assert(X.At<"x">(2).Index() == X(x, 2).Index());
    static_assert(U.At<"u">(8).At<"leg_input">(2).At<"relative_position">().Index() ==
                  U(relative_position, 8, 2).Index());

    /***************     Operate on each subvariable in a given hierarchy.    ***************/
    const auto getKind = [](Concepts::Variable auto v) {
        if (v.IsScalar()) {
            return "'scalar'"s;
        } else if (v.IsVector()) {
            return "'vector'"s;
        } else if (v.IsQuaternion()) {
            return "'quaternion'"s;
        } else {
            return "'unknown'"s;
        }
    };

    // Define a Boost.Hana struct to benefit from the printing capabilities of UNGAR_LOG.
    struct VariableInfo {
        BOOST_HANA_DEFINE_STRUCT(VariableInfo,
                                 (std::string, name),
                                 (index_t, index),
                                 (index_t, size),
                                 (std::string, kind));
    };

    UNGAR_LOG(info,
              "Printing all subvariables of 'x' with the format {{ name, index, size, kind }}...");
    x.ForEach([&](Concepts::Variable auto var) {
        UNGAR_LOG(
            info, "{:c}", VariableInfo{var.Name().c_str(), var.Index(), var.Size(), getKind(var)});
    });

    return 0;
}