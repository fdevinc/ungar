/******************************************************************************
 *
 * @file ungar/test/variable.test.cpp
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

#include <gtest/gtest.h>

#include "ungar/variable_map.hpp"

namespace Ungar {
namespace Test {
namespace {

constexpr auto N         = 10_c;
constexpr auto NUM_IPMS  = 4_c;
constexpr auto NUM_SRBDS = 2_c;

UNGAR_VARIABLE(position, 3);
UNGAR_VARIABLE(orientation, Q);
UNGAR_VARIABLE(linear_velocity, 3);
UNGAR_VARIABLE(b_angular_velocity, 3);

UNGAR_VARIABLE(srbd_state) <<= (position, orientation, linear_velocity, b_angular_velocity);
UNGAR_VARIABLE(ipm_state) <<= (position, linear_velocity);

UNGAR_VARIABLE(x) <<= (NUM_IPMS * ipm_state, NUM_SRBDS* srbd_state);
UNGAR_VARIABLE(X) <<= N * x;

UNGAR_VARIABLE(ipm_mass, 1);
UNGAR_VARIABLE(srbd_mass, 1);
UNGAR_VARIABLE(Rho) <<= (NUM_IPMS * ipm_mass, NUM_SRBDS* srbd_mass);

UNGAR_VARIABLE(rollout_variables) <<= (X, Rho);

TEST(RolloutVariablesTest, VariableMapAccess) {
    using namespace Ungar;

    auto map     = MakeVariableMap<real_t>(rollout_variables);
    auto lazyMap = VariableLazyMap{map.Get(), rollout_variables};

    for (const auto i : enumerate(1024)) {
        map.Get().setRandom();

        EXPECT_EQ(map.Get(rollout_variables), map.Get());
        EXPECT_EQ(map.Get(rollout_variables), lazyMap.Get(rollout_variables));
        EXPECT_EQ(map.Get(X), lazyMap.Get(X));
        EXPECT_EQ(map.Get(Rho), lazyMap.Get(Rho));
        for (const auto k : enumerate(N)) {
            EXPECT_EQ(map.Get(X, x, k), lazyMap.Get(X, x, k));

            for (const auto ipmIndex : enumerate(NUM_IPMS)) {
                EXPECT_EQ(map.Get(ipm_state, k, ipmIndex), lazyMap.Get(ipm_state, k, ipmIndex));
                EXPECT_EQ(map.Get(ipm_state, k, ipmIndex, position),
                          lazyMap.Get(ipm_state, k, ipmIndex, position));
                EXPECT_EQ(map.Get(ipm_state, k, ipmIndex, linear_velocity),
                          lazyMap.Get(ipm_state, k, ipmIndex, linear_velocity));
            }

            for (const auto srbdIndex : enumerate(NUM_SRBDS)) {
                const auto& var = rollout_variables(srbd_state, k, srbdIndex);
                EXPECT_EQ(map.Get1(var), lazyMap.Get1(var));
                EXPECT_EQ(map.Get1(var(position)), lazyMap.Get1(var(position)));
                EXPECT_EQ(map.Get1(var(orientation)), lazyMap.Get1(var(orientation)));
                EXPECT_EQ(map.Get1(var(linear_velocity)), lazyMap.Get1(var(linear_velocity)));
                EXPECT_EQ(map.Get1(var(b_angular_velocity)), lazyMap.Get1(var(b_angular_velocity)));
            }
        }

        for (const auto ipmIndex : enumerate(NUM_IPMS)) {
            EXPECT_EQ(map.Get(ipm_mass, ipmIndex), lazyMap.Get(ipm_mass, ipmIndex));
        }

        for (const auto srbdIndex : enumerate(NUM_SRBDS)) {
            EXPECT_EQ(map.Get(srbd_mass, srbdIndex), lazyMap.Get(srbd_mass, srbdIndex));
        }
    }
}

TEST(RolloutVariablesTest, VariableMapAssignment) {
    using namespace Ungar;

    auto map = MakeVariableMap<real_t>(rollout_variables);

    map.Get().setZero();

    for (const auto k : enumerate(N)) {
        for (const auto ipmIndex : enumerate(NUM_IPMS)) {
            const auto& var = rollout_variables(ipm_state, k, ipmIndex);
            map.Get1(var).setOnes();
        }

        for (const auto srbdIndex : enumerate(NUM_SRBDS)) {
            map.Get(srbd_state, k, srbdIndex).setOnes();
        }

        EXPECT_EQ(map.Get(x, k), VectorXr::Ones(x.Size()));
    }

    for (const auto ipmIndex : enumerate(NUM_IPMS)) {
        const auto& var = rollout_variables(ipm_mass, ipmIndex);
        map.Get1(var)   = 1.0;
    }

    for (const auto srbdIndex : enumerate(NUM_SRBDS)) {
        map.Get(srbd_mass, srbdIndex) = 1.0;
    }
    EXPECT_EQ(map.Get(Rho), VectorXr::Ones(Rho.Size()));
}

}  // namespace
}  // namespace Test
}  // namespace Ungar

int main() {
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
