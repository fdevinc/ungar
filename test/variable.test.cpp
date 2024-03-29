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

constexpr auto position           = var_c<"position", 3>;
constexpr auto orientation        = var_c<"orientation", Q>;
constexpr auto linear_velocity    = var_c<"linear_velocity", 3>;
constexpr auto b_angular_velocity = var_c<"b_angular_velocity", 3>;

constexpr auto srbd_state = var_c<"srbd_state"> <<=
    (position, orientation, linear_velocity, b_angular_velocity);
constexpr auto ipm_state = var_c<"ipm_state"> <<= (position, linear_velocity);

constexpr auto x = var_c<"x"> <<= (NUM_IPMS * ipm_state, NUM_SRBDS* srbd_state);
constexpr auto X = var_c<"X"> <<= N * x;

constexpr auto ipm_mass  = var_c<"ipm_mass", 1>;
constexpr auto srbd_mass = var_c<"srbd_mass", 1>;
constexpr auto Rho       = var_c<"Rho"> <<= (NUM_IPMS * ipm_mass, NUM_SRBDS* srbd_mass);

constexpr auto variables = var_c<"variables"> <<= (X, Rho);

TEST(RolloutVariablesTest, VariableMapAccess) {
    using namespace Ungar;

    auto map     = MakeVariableMap<real_t>(variables);
    auto lazyMap = VariableLazyMap{map.Get(), variables};

    for (const auto i : enumerate(1024)) {
        map.Get().setRandom();

        EXPECT_EQ(map.Get(variables), map.Get());
        EXPECT_EQ(map.Get(variables), lazyMap.Get(variables));
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
                const auto& var = variables(srbd_state, k, srbdIndex);
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

    auto map = MakeVariableMap<real_t>(variables);

    map.Get().setZero();

    for (const auto k : enumerate(N)) {
        for (const auto ipmIndex : enumerate(NUM_IPMS)) {
            const auto& var = variables(ipm_state, k, ipmIndex);
            map.Get1(var).setOnes();
        }

        for (const auto srbdIndex : enumerate(NUM_SRBDS)) {
            map.Get(srbd_state, k, srbdIndex).setOnes();
        }

        EXPECT_EQ(map.Get(x, k), VectorXr::Ones(x.Size()));
    }

    for (const auto ipmIndex : enumerate(NUM_IPMS)) {
        const auto& var = variables(ipm_mass, ipmIndex);
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
