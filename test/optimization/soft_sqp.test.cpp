/******************************************************************************
 *
 * @file ungar/test/soft_sqp.test.cpp
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

#include "ungar/optimization/soft_sqp.hpp"

namespace Ungar {
namespace Test {
namespace {

TEST(OptimizationTest, SoftSQP) {
    using namespace Ungar;
    using namespace Ungar::Autodiff;

    const index_t xSize = 2_idx;
    const index_t pSize = 0_idx;

    const auto predicate = [](const auto& lhs, const auto& rhs) { return lhs.isApprox(rhs, 1e-1); };

    const auto objBlueprint =
        Function::Blueprint{[](const VectorXad& xp, VectorXad& y) {
                                y = VectorXad{{Utils::Pow(xp.x() - ad_scalar_t{3.0}, 2) +
                                               Utils::Pow(xp.y() - ad_scalar_t{2.0}, 2)}};
                            },
                            xSize,
                            pSize,
                            "obj_soft_sqp_test"sv};
    const auto eqsBlueprint = Function::Blueprint{
        [](const VectorXad& xp, VectorXad& y) { y = VectorXad{{xp.x() - xp.y()}}; },
        xSize,
        pSize,
        "eqs_soft_sqp_test"sv,
        EnabledDerivatives::JACOBIAN};

    const auto ineqsBlueprint1 =
        Function::Blueprint{[](const VectorXad& xp, VectorXad& y) {
                                y = VectorXad{{xp.y() - ad_scalar_t{1.0}, -xp.x()}};
                            },
                            xSize,
                            pSize,
                            "ineqs_1_soft_sqp_test"sv,
                            EnabledDerivatives::JACOBIAN};
    auto nlpProblem1 = MakeNLPProblem(FunctionFactory::Make(objBlueprint, true),
                                      hana::nothing,
                                      hana::just(FunctionFactory::Make(ineqsBlueprint1, true)));
    SoftSQPOptimizer<decltype(nlpProblem1)::element_type> optimizer1{
        false, 1.0, 100_idx, 100.0, 2e-8};
    optimizer1.Initialize(std::move(nlpProblem1));

    const VectorXr xOpt1 = optimizer1.Optimize(VectorXr::Zero(xSize + pSize));
    const VectorXr xOptGroundTruth1{{3.0, 1.0}};
    UNGAR_LOG(debug, Utils::DASH_LINE_SEPARATOR);
    UNGAR_LOG(debug, "PROBLEM 1");
    UNGAR_LOG(debug,
              "The ground truth optimal solution is {}; SQP converged to {}.",
              xOptGroundTruth1,
              xOpt1);
    ASSERT_PRED2(predicate, xOpt1, xOptGroundTruth1);

    const auto ineqsBlueprint2 =
        Function::Blueprint{[](const VectorXad& xp, VectorXad& y) {
                                y = VectorXad{{Utils::Pow(xp.x(), 2) - xp.y() - ad_scalar_t{3.0},
                                               xp.y() - ad_scalar_t{1.0},
                                               -xp.x()}};
                            },
                            xSize,
                            pSize,
                            "ineqs_2_soft_sqp_test"sv,
                            EnabledDerivatives::JACOBIAN};
    auto nlpProblem2 = MakeNLPProblem(FunctionFactory::Make(objBlueprint, false),
                                      hana::nothing,
                                      hana::just(FunctionFactory::Make(ineqsBlueprint2, true)));
    SoftSQPOptimizer<decltype(nlpProblem2)::element_type> optimizer2{
        false, 1.0, 100_idx, 100.0, 2e-8};
    optimizer2.Initialize(std::move(nlpProblem2));

    const VectorXr xOpt2 = optimizer2.Optimize(VectorXr::Zero(xSize + pSize));
    const VectorXr xOptGroundTruth2{{2.0, 1.0}};
    UNGAR_LOG(debug, Utils::DASH_LINE_SEPARATOR);
    UNGAR_LOG(debug, "PROBLEM 2");
    UNGAR_LOG(debug,
              "The ground truth optimal solution is {}; SQP converged to {}.",
              xOptGroundTruth2,
              xOpt2);
    ASSERT_PRED2(predicate, xOpt2, xOptGroundTruth2);

    auto nlpProblem3 = MakeNLPProblem(FunctionFactory::Make(objBlueprint, false),
                                      hana::just(FunctionFactory::Make(eqsBlueprint, true)),
                                      hana::just(FunctionFactory::Make(ineqsBlueprint2, false)));
    SoftSQPOptimizer<decltype(nlpProblem3)::element_type> optimizer3{
        false, 1.0, 100_idx, 100.0, 2e-8};
    optimizer3.Initialize(std::move(nlpProblem3));

    const VectorXr xOpt3 = optimizer3.Optimize(VectorXr::Zero(xSize + pSize));
    const VectorXr xOptGroundTruth3{{1.0, 1.0}};
    UNGAR_LOG(debug, Utils::DASH_LINE_SEPARATOR);
    UNGAR_LOG(debug, "PROBLEM 3");
    UNGAR_LOG(debug,
              "The ground truth optimal solution is {}; SQP converged to {}.",
              xOptGroundTruth3,
              xOpt3);
    ASSERT_PRED2(predicate, xOpt3, xOptGroundTruth3);
}

}  // namespace
}  // namespace Test
}  // namespace Ungar

int main() {
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
