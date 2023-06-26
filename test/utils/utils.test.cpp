/******************************************************************************
 *
 * @file ungar/test/utils/utils.test.cpp
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

#include "ungar/utils/utils.hpp"

namespace {

TEST(UtilsTest, StringManipulation) {
    using namespace Ungar;

    const std::string camelCaseWord = "CamelCaseWord";
    ASSERT_EQ("camel_case_word", Utils::ToSnakeCase(camelCaseWord));
    const std::string acronymWord = "CCWord";
    ASSERT_EQ("cc_word", Utils::ToSnakeCase(acronymWord));
    const std::string dirtyWord = "Dirty&Word";
    ASSERT_EQ("dirty_word", Utils::ToSnakeCase(dirtyWord));
}

TEST(UtilsTest, QuaternionTransformations) {
    using namespace Ungar;

    std::mt19937 gen{0U};
    for (const auto i : enumerate(1024)) {
        const Quaternionr q0 = Quaternionr::UnitRandom();
        const Vector3r ypr0  = Utils::QuaternionToYawPitchRoll(q0);
        const Matrix3r R0    = Utils::RotationMatrixFromYawPitchRoll(ypr0);
        const Quaternionr q1{R0};
        const Quaternionr q2{Utils::QuaternionFromYawPitchRoll(ypr0)};

        const auto predicate = [](const auto& lhs, const auto& rhs) {
            return lhs.isApprox(rhs, 1e-6);
        };
        ASSERT_PRED2(predicate, R0, q0.toRotationMatrix());
        ASSERT_PRED2(predicate, q0.toRotationMatrix(), q1.toRotationMatrix());
        ASSERT_NEAR(std::abs(q0.dot(q1)), 1.0, Eigen::NumTraits<real_t>::epsilon());
        ASSERT_PRED2(predicate, q0.toRotationMatrix(), q2.toRotationMatrix());

        std::uniform_real_distribution<> dis(-Utils::PI, Utils::PI);
        const real_t angle = dis(gen);
        ASSERT_PRED2(predicate,
                     Utils::ElementaryXRotationMatrix(angle),
                     Utils::ElementaryXQuaternion(angle).toRotationMatrix());
        ASSERT_PRED2(predicate,
                     Utils::ElementaryYRotationMatrix(angle),
                     Utils::ElementaryYQuaternion(angle).toRotationMatrix());
        ASSERT_PRED2(predicate,
                     Utils::ElementaryZRotationMatrix(angle),
                     Utils::ElementaryZQuaternion(angle).toRotationMatrix());
    }

    for (const auto i : enumerate(1024)) {
        const Vector3r ypr = Vector3r::Random();

        const auto t0 = [](const Vector3r& ypr) { return Vector3r{ypr.array() + 2.0 * Utils::PI}; };
        const auto t1 = [](const Vector3r& ypr) {
            return Vector3r{ypr.x() + Utils::PI, Utils::PI - ypr.y(), ypr.z() + Utils::PI};
        };
        const auto t2 = [](const Vector3r& ypr) {
            return Vector3r{ypr.x() + Utils::PI, Utils::PI - ypr.y(), ypr.z() - Utils::PI};
        };
        const auto t3 = [](const Vector3r& ypr) {
            return Vector3r{ypr.x() - Utils::PI, Utils::PI - ypr.y(), ypr.z() + Utils::PI};
        };
        const auto t4 = [](const Vector3r& ypr) {
            return Vector3r{ypr.x() - Utils::PI, Utils::PI - ypr.y(), ypr.z() - Utils::PI};
        };
        const auto t5 = [](const Vector3r& ypr) {
            return Vector3r{ypr.x() + Utils::PI, -Utils::PI - ypr.y(), ypr.z() + Utils::PI};
        };
        const auto t6 = [](const Vector3r& ypr) {
            return Vector3r{ypr.x() + Utils::PI, -Utils::PI - ypr.y(), ypr.z() - Utils::PI};
        };
        const auto t7 = [](const Vector3r& ypr) {
            return Vector3r{ypr.x() - Utils::PI, -Utils::PI - ypr.y(), ypr.z() + Utils::PI};
        };
        const auto t8 = [](const Vector3r& ypr) {
            return Vector3r{ypr.x() - Utils::PI, -Utils::PI - ypr.y(), ypr.z() - Utils::PI};
        };

        const auto predicate = [](const auto& lhs, const auto& rhs) {
            return lhs.isApprox(rhs, 1e-6);
        };

        const Matrix3r R = Utils::RotationMatrixFromYawPitchRoll(ypr);
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t0(ypr)));
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t1(ypr)));
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t2(ypr)));
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t3(ypr)));
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t4(ypr)));
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t5(ypr)));
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t6(ypr)));
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t7(ypr)));
        ASSERT_PRED2(predicate, R, Utils::RotationMatrixFromYawPitchRoll(t8(ypr)));
    }
}

TEST(UtilsTest, SparseMatrixManipulations) {
    using namespace Ungar;

    Matrix2r m1{{1.0, 0.0}, {0.0, 1.0}};
    Matrix2r m2{{0.0, 0.0}, {2.0, 0.0}};
    const SparseMatrix<real_t> s1 = m1.sparseView();
    const SparseMatrix<real_t> s2 = m2.sparseView();

    SparseMatrix<real_t> st1 = Utils::VerticallyStackSparseMatrices(s1, s2).ToSparse();
    ASSERT_EQ(st1.toDense().topRows(2_idx), m1);
    ASSERT_EQ(st1.toDense().bottomRows(2_idx), m2);

    SparseMatrix<real_t> st2;
    Utils::VerticallyStackSparseMatrices(s2, m2.sparseView(), s1, st1).In(st2);
    ASSERT_EQ(st2.toDense().topRows(2_idx), s2.toDense());
    ASSERT_EQ(st2.toDense().middleRows(2_idx, 2_idx), m2);
    ASSERT_EQ(st2.toDense().middleRows(4_idx, 2_idx), s1.toDense());
    ASSERT_EQ(st2.toDense().bottomRows(4_idx), st1.toDense());
}

}  // namespace

int main() {
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
