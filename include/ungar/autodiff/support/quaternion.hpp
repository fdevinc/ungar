/******************************************************************************
 *
 * @file ungar/autodiff/support/quaternion.hpp
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

#ifndef _UNGAR__AUTODIFF__SUPPORT__QUATERNION_HPP_
#define _UNGAR__AUTODIFF__SUPPORT__QUATERNION_HPP_

#include "ungar/data_types.hpp"

namespace Eigen {

template <>
EIGEN_DEVICE_FUNC inline Quaternion<typename internal::traits<Ungar::Quaternionad>::Scalar>
QuaternionBase<Ungar::Quaternionad>::inverse() const {
    Scalar n2 = this->squaredNorm();
    return Quaternion<Scalar>{conjugate().coeffs() /
                              CppAD::CondExpGt(n2, RealScalar{0}, n2, RealScalar{1.0})};
}
template <>
EIGEN_DEVICE_FUNC inline Quaternion<typename internal::traits<Ungar::Quaternionad>::Scalar>
QuaternionBase<Eigen::Map<Ungar::Quaternionad>>::inverse() const {
    Scalar n2 = this->squaredNorm();
    return Quaternion<Scalar>{conjugate().coeffs() /
                              CppAD::CondExpGt(n2, RealScalar{0}, n2, RealScalar{1.0})};
}
template <>
EIGEN_DEVICE_FUNC inline Quaternion<typename internal::traits<Ungar::Quaternionad>::Scalar>
QuaternionBase<Eigen::Map<const Ungar::Quaternionad>>::inverse() const {
    Scalar n2 = this->squaredNorm();
    return Quaternion<Scalar>{conjugate().coeffs() /
                              CppAD::CondExpGt(n2, RealScalar{0}, n2, RealScalar{1.0})};
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void MatrixBase<Ungar::Vector4ad>::normalize() {
    RealScalar z = squaredNorm();
    derived() /= CppAD::CondExpGt(z, RealScalar{0}, numext::sqrt(z), RealScalar{1.0});
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void MatrixBase<Eigen::Map<Ungar::Vector4ad>>::normalize() {
    RealScalar z = squaredNorm();
    derived() /= CppAD::CondExpGt(z, RealScalar{0}, numext::sqrt(z), RealScalar{1.0});
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const typename MatrixBase<Ungar::Vector4ad>::PlainObject
MatrixBase<Ungar::Vector4ad>::normalized() const {
    typedef typename internal::nested_eval<Ungar::Vector4ad, 2>::type _Nested;
    _Nested n(derived());
    RealScalar z = n.squaredNorm();
    return n / CppAD::CondExpGt(z, RealScalar{0}, numext::sqrt(z), RealScalar{1.0});
}
template <>
EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const typename MatrixBase<Eigen::Map<Ungar::Vector4ad>>::PlainObject
    MatrixBase<Eigen::Map<Ungar::Vector4ad>>::normalized() const {
    typedef typename internal::nested_eval<Eigen::Map<Ungar::Vector4ad>, 2>::type _Nested;
    _Nested n(derived());
    RealScalar z = n.squaredNorm();
    return n / CppAD::CondExpGt(z, RealScalar{0}, numext::sqrt(z), RealScalar{1.0});
}
template <>
EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const typename MatrixBase<Eigen::Map<const Ungar::Vector4ad>>::PlainObject
    MatrixBase<Eigen::Map<const Ungar::Vector4ad>>::normalized() const {
    typedef typename internal::nested_eval<Eigen::Map<const Ungar::Vector4ad>, 2>::type _Nested;
    _Nested n(derived());
    RealScalar z = n.squaredNorm();
    return n / CppAD::CondExpGt(z, RealScalar{0}, numext::sqrt(z), RealScalar{1.0});
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const typename MatrixBase<Ungar::Vector3ad>::PlainObject
MatrixBase<Ungar::Vector3ad>::normalized() const {
    typedef typename internal::nested_eval<Ungar::Vector3ad, 2>::type _Nested;
    _Nested n(derived());
    RealScalar z = n.squaredNorm();
    return n / CppAD::CondExpGt(z, RealScalar{0}, numext::sqrt(z), RealScalar{1.0});
}
template <>
EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const typename MatrixBase<Eigen::Map<Ungar::Vector3ad>>::PlainObject
    MatrixBase<Eigen::Map<Ungar::Vector3ad>>::normalized() const {
    typedef typename internal::nested_eval<Eigen::Map<Ungar::Vector3ad>, 2>::type _Nested;
    _Nested n(derived());
    RealScalar z = n.squaredNorm();
    return n / CppAD::CondExpGt(z, RealScalar{0}, numext::sqrt(z), RealScalar{1.0});
}
template <>
EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const typename MatrixBase<Eigen::Map<const Ungar::Vector3ad>>::PlainObject
    MatrixBase<Eigen::Map<const Ungar::Vector3ad>>::normalized() const {
    typedef typename internal::nested_eval<Eigen::Map<const Ungar::Vector3ad>, 2>::type _Nested;
    _Nested n(derived());
    RealScalar z = n.squaredNorm();
    return n / CppAD::CondExpGt(z, RealScalar{0}, numext::sqrt(z), RealScalar{1.0});
}

namespace internal {

template <>
struct quaternionbase_assign_impl<Ungar::Matrix3ad, 3, 3> {
    typedef typename Ungar::Matrix3ad::Scalar Scalar;
    template <class Derived>
    EIGEN_DEVICE_FUNC static inline void run(QuaternionBase<Derived>& q,
                                             const Ungar::Matrix3ad& a_mat) {
        static_assert(Ungar::dependent_false<Derived>,
                      "The construction of unit quaternions from rotation matrices with scalar "
                      "type 'ad_scalar_t' is not implemented.");
    }
};

}  // namespace internal

template <>
template <class OtherDerived>
EIGEN_DEVICE_FUNC Quaternion<typename internal::traits<Ungar::Quaternionad>::Scalar>
QuaternionBase<Ungar::Quaternionad>::slerp(const Scalar& t,
                                           const QuaternionBase<OtherDerived>& other) const {
    EIGEN_USING_STD(acos)
    EIGEN_USING_STD(sin)
    const Scalar one  = Scalar{1.0} - NumTraits<Scalar>::epsilon();
    const Scalar d    = this->dot(other);
    const Scalar absD = numext::abs(d);

    Scalar scale0;
    Scalar scale1;

    scale0 = CppAD::CondExpGe(
        absD, one, Scalar{1.0} - t, sin((Scalar{1.0} - t) * acos(absD)) / sin(acos(absD)));
    scale1 = CppAD::CondExpGe(absD, one, t, sin((t * acos(absD))) / sin(acos(absD)));
    scale1 = CppAD::CondExpLt(d, Scalar{0.0}, -scale1, scale1);

    return Quaternion<Scalar>{scale0 * coeffs() + scale1 * other.coeffs()};
}
template <>
template <class OtherDerived>
EIGEN_DEVICE_FUNC Quaternion<typename internal::traits<Ungar::Quaternionad>::Scalar>
QuaternionBase<Eigen::Map<Ungar::Quaternionad>>::slerp(
    const Scalar& t, const QuaternionBase<OtherDerived>& other) const {
    EIGEN_USING_STD(acos)
    EIGEN_USING_STD(sin)
    const Scalar one  = Scalar{1.0} - NumTraits<Scalar>::epsilon();
    const Scalar d    = this->dot(other);
    const Scalar absD = numext::abs(d);

    Scalar scale0;
    Scalar scale1;

    scale0 = CppAD::CondExpGe(
        absD, one, Scalar{1.0} - t, sin((Scalar{1.0} - t) * acos(absD)) / sin(acos(absD)));
    scale1 = CppAD::CondExpGe(absD, one, t, sin((t * acos(absD))) / sin(acos(absD)));
    scale1 = CppAD::CondExpLt(d, Scalar{0.0}, -scale1, scale1);

    return Quaternion<Scalar>{scale0 * coeffs() + scale1 * other.coeffs()};
}
template <>
template <class OtherDerived>
EIGEN_DEVICE_FUNC Quaternion<typename internal::traits<Ungar::Quaternionad>::Scalar>
QuaternionBase<Eigen::Map<const Ungar::Quaternionad>>::slerp(
    const Scalar& t, const QuaternionBase<OtherDerived>& other) const {
    EIGEN_USING_STD(acos)
    EIGEN_USING_STD(sin)
    const Scalar one  = Scalar{1.0} - NumTraits<Scalar>::epsilon();
    const Scalar d    = this->dot(other);
    const Scalar absD = numext::abs(d);

    Scalar scale0;
    Scalar scale1;

    scale0 = CppAD::CondExpGe(
        absD, one, Scalar{1.0} - t, sin((Scalar{1.0} - t) * acos(absD)) / sin(acos(absD)));
    scale1 = CppAD::CondExpGe(absD, one, t, sin((t * acos(absD))) / sin(acos(absD)));
    scale1 = CppAD::CondExpLt(d, Scalar{0.0}, -scale1, scale1);

    return Quaternion<Scalar>{scale0 * coeffs() + scale1 * other.coeffs()};
}

template <>
template <typename Derived1, typename Derived2>
EIGEN_DEVICE_FUNC inline Ungar::Quaternionad&
QuaternionBase<Ungar::Quaternionad>::setFromTwoVectors(const MatrixBase<Derived1>& a,
                                                       const MatrixBase<Derived2>& b) {
    static_assert(Ungar::dependent_false<Derived1, Derived2>,
                  "The construction of unit quaternions with scalar type 'ad_scalar_t' from two "
                  "vectors is not implemented.");
    return derived();
}
template <>
template <typename Derived1, typename Derived2>
EIGEN_DEVICE_FUNC inline Eigen::Map<Ungar::Quaternionad>&
QuaternionBase<Eigen::Map<Ungar::Quaternionad>>::setFromTwoVectors(const MatrixBase<Derived1>& a,
                                                                   const MatrixBase<Derived2>& b) {
    static_assert(Ungar::dependent_false<Derived1, Derived2>,
                  "The construction of unit quaternions with scalar type 'ad_scalar_t' from two "
                  "vectors is not implemented.");
    return derived();
}
template <>
template <typename Derived1, typename Derived2>
EIGEN_DEVICE_FUNC inline Eigen::Map<const Ungar::Quaternionad>&
QuaternionBase<Eigen::Map<const Ungar::Quaternionad>>::setFromTwoVectors(
    const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b) {
    static_assert(Ungar::dependent_false<Derived1, Derived2>,
                  "The construction of unit quaternions with scalar type 'ad_scalar_t' from two "
                  "vectors is not implemented.");
    return derived();
}

}  // namespace Eigen

#endif /* _UNGAR__AUTODIFF__SUPPORT__QUATERNION_HPP_ */
