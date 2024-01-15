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

TEST(UtilsTest, VectorManipulation) {
    using namespace Ungar;

    const index_t variableSize = 4;

    const VectorXr x     = VectorXr::Random(variableSize);
    const auto [a, b, c] = Utils::Decompose<2, 1, 1>(x);

    ASSERT_TRUE((std::same_as<const MapToConstVector2r, decltype(a)>));
    ASSERT_TRUE((Concepts::Same<const real_t&, decltype(b), decltype(c)>));
    ASSERT_EQ(a.RowsAtCompileTime, 2);
    ASSERT_TRUE(x.data() == a.data());
    EXPECT_EQ(a, x.head<2>());
    EXPECT_EQ(b, x[2]);
    EXPECT_EQ(c, x[3]);

    const auto [d] = Utils::Decompose<4>(x);
    ASSERT_EQ(d, x);
    ASSERT_TRUE((std::same_as<const MapToConstVector4r, decltype(d)>));
    ASSERT_EQ(d.RowsAtCompileTime, 4);

    VectorXr y{variableSize};
    Utils::Compose(a, b, c).In(y);
    EXPECT_EQ(x, y);
    EXPECT_EQ(x, Utils::Compose(a, b, c).ToDynamic());
    EXPECT_EQ(x, Utils::Compose(a, b, c).ToFixed());
    ASSERT_EQ(Utils::Compose(a, b, c).ToDynamic().RowsAtCompileTime, Eigen::Dynamic);
    ASSERT_EQ(Utils::Compose(a, b, c).ToFixed().RowsAtCompileTime, variableSize);

    VectorXr r{variableSize};
    Utils::Compose(VectorXr{a}, b, c).In(r);
    EXPECT_EQ(x, r);
    EXPECT_EQ(x, Utils::Compose(a, b, c).ToDynamic());

    auto [e, f, g, h] = Utils::Decompose<2, 1, 1, 0>(y);
    ASSERT_TRUE((std::same_as<MapToVector2r, decltype(e)>));
    ASSERT_TRUE((Concepts::Same<real_t&, decltype(f), decltype(g)>));
    ASSERT_EQ(e.RowsAtCompileTime, 2);
    ASSERT_TRUE(y.data() == e.data());
    EXPECT_EQ(e, y.head<2>());
    EXPECT_EQ(f, y[2]);
    EXPECT_EQ(g, y[3]);
    e += Vector2r::Ones();
    EXPECT_EQ(e, y.head<2>());
    f += g;
    EXPECT_EQ(f, y[2]);
    g += e.norm();
    EXPECT_EQ(g, y[3]);
    ASSERT_TRUE(h.RowsAtCompileTime == 0);
    const auto [q1] = Utils::Decompose<Q>(x);
    ASSERT_TRUE((std::same_as<const Eigen::Map<const Quaternionr>, decltype(q1)>));

    y.normalize();
    auto [q2] = Utils::Decompose<Q>(y);
    ASSERT_TRUE((std::same_as<Eigen::Map<Quaternionr>, decltype(q2)>));
    EXPECT_EQ(y, q2.coeffs());
    q2 = q2.inverse();
    EXPECT_EQ(y, q2.coeffs());

    Vector4r z;
    Utils::Compose(e, f, g, h).In(z);
    EXPECT_EQ(y, z);

    VectorXr w{1};
    auto vec = std::vector{Utils::Decompose<1>(w)};
    auto [i] = vec.front();
    ASSERT_TRUE((Concepts::Same<real_t&, decltype(i)>));

    const auto [l, m, n] = Utils::Decompose<2, 1, 1>(x.head<4>());
    ASSERT_TRUE((std::same_as<const MapToConstVector2r, decltype(l)>));
    ASSERT_TRUE((Concepts::Same<const real_t&, decltype(m), decltype(n)>));
    ASSERT_EQ(l.RowsAtCompileTime, 2);
    ASSERT_TRUE(x.data() == l.data());
    EXPECT_EQ(l, x.head<2>());
    EXPECT_EQ(m, x[2]);
    EXPECT_EQ(n, x[3]);

    struct S1 {
        MapToVector2r a;
        Eigen::Map<Quaternionr> b;
        real_t& c;
        real_t& d;
    };
    VectorXr s1Vec = VectorXr::Random(8);
    auto s1        = hana::unpack(Utils::Decompose<2, Q, 1, 1>(s1Vec),
                           [&](auto&&... args) { return S1{args...}; });
    ASSERT_TRUE((std::same_as<MapToVector2r, decltype(s1.a)>));
    ASSERT_TRUE((std::same_as<Eigen::Map<Quaternionr>, decltype(s1.b)>));
    ASSERT_TRUE((Concepts::Same<real_t&, decltype(s1.c), decltype(s1.d)>));
    ASSERT_TRUE(s1Vec.data() == s1.a.data());
    EXPECT_EQ(s1.a, s1Vec.head<2>());
    EXPECT_EQ(s1.b.coeffs(), s1Vec.segment<4>(2_idx));
    EXPECT_EQ(s1.c, s1Vec[6_idx]);
    EXPECT_EQ(s1.d, s1Vec[7_idx]);
    std::vector<S1> vec1{};
    hana::unpack(Utils::Decompose<2, Q, 1, 1>(s1Vec),
                 [&](auto&&... args) { vec1.emplace_back(args...); });
    ASSERT_TRUE((std::same_as<MapToVector2r, decltype(vec1.front().a)>));
    ASSERT_TRUE((std::same_as<Eigen::Map<Quaternionr>, decltype(vec1.front().b)>));
    ASSERT_TRUE((Concepts::Same<real_t&, decltype(vec1.front().c), decltype(vec1.front().d)>));
    ASSERT_TRUE(s1Vec.data() == vec1.front().a.data());
    EXPECT_EQ(vec1.front().a, s1Vec.head<2>());
    EXPECT_EQ(vec1.front().b.coeffs(), s1Vec.segment<4>(2_idx));
    EXPECT_EQ(vec1.front().c, s1Vec[6_idx]);
    EXPECT_EQ(vec1.front().d, s1Vec[7_idx]);

    struct S2 {
        MapToConstVector2r a;
        Eigen::Map<const Quaternionr> b;
        const real_t& c;
        const real_t& d;
    };
    const VectorXr s2Vec = VectorXr::Random(8);
    const auto s2        = hana::unpack(Utils::Decompose<2, Q, 1, 1>(s2Vec),
                                 [&](auto&&... args) { return S2{args...}; });
    ASSERT_TRUE((std::same_as<MapToConstVector2r, decltype(s2.a)>));
    ASSERT_TRUE((std::same_as<Eigen::Map<const Quaternionr>, decltype(s2.b)>));
    ASSERT_TRUE((Concepts::Same<const real_t&, decltype(s2.c), decltype(s2.d)>));
    ASSERT_TRUE(s2Vec.data() == s2.a.data());
    EXPECT_EQ(s2.a, s2Vec.head<2>());
    EXPECT_EQ(s2.b.coeffs(), s2Vec.segment<4>(2_idx));
    EXPECT_EQ(s2.c, s2Vec[6_idx]);
    EXPECT_EQ(s2.d, s2Vec[7_idx]);
    std::vector<S2> vec2{};
    hana::unpack(Utils::Decompose<2, Q, 1, 1>(s2Vec),
                 [&](auto&&... args) { vec2.emplace_back(args...); });
    ASSERT_TRUE((std::same_as<MapToConstVector2r, decltype(vec2.front().a)>));
    ASSERT_TRUE((std::same_as<Eigen::Map<const Quaternionr>, decltype(vec2.front().b)>));
    ASSERT_TRUE(
        (Concepts::Same<const real_t&, decltype(vec2.front().c), decltype(vec2.front().d)>));
    ASSERT_TRUE(s2Vec.data() == vec2.front().a.data());
    EXPECT_EQ(vec2.front().a, s2Vec.head<2>());
    EXPECT_EQ(vec2.front().b.coeffs(), s2Vec.segment<4>(2_idx));
    EXPECT_EQ(vec2.front().c, s2Vec[6_idx]);
    EXPECT_EQ(vec2.front().d, s2Vec[7_idx]);
}

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

        std::uniform_real_distribution<> dis(-std::numbers::pi, std::numbers::pi);
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
