/******************************************************************************
 *
 * @file ungar/variable_lazy_map.hpp
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

#ifndef _UNGAR__VARIABLE_LAZY_MAP_HPP_
#define _UNGAR__VARIABLE_LAZY_MAP_HPP_

#include "ungar/variable.hpp"

namespace Ungar {

/**
 * @brief Class representing a lazy mapping between a variable and an underlying data container.
 *
 * The VariableLazyMap class provides a lazy mapping between a Variable object and an underlying
 * data container, such as an Eigen vector or an std::array. It allows efficient access to the
 * data associated with the variable without explicitly copying or storing the data.
 *
 * @tparam _Scalar               Type of the underlying data (e.g., real_t, ad_scalar_t).
 * @tparam _Variable             Variable type representing the variable being mapped.
 * @tparam _EnableMutableMembers Boolean integral constant indicating whether mutable members are
 *                               enabled or not.
 *
 * @warning The size of the underlying data must be identical to the size of the variable being.
 *          mapped. Incorrect type mapping may lead to unexpected behavior or runtime errors.
 */
template <typename _Scalar, Concepts::Variable _Variable, Concepts::HanaBool _EnableMutableMembers>
class VariableLazyMap {
  private:
    template <typename _Underlying>
    static constexpr auto UnderlyingSize(hana::basic_type<_Underlying>) {
        if constexpr (Concepts::DenseMatrixExpression<_Underlying>) {
            return std::remove_cvref_t<_Underlying>::RowsAtCompileTime;
        } else {
            return static_cast<index_t>(std::ranges::size(_Underlying{}));
        }
    }

  public:
    using ScalarType = _Scalar;

    /**
     * @brief Default constructor (not intended for use).
     *
     * @warning Calling this constructor results in undefined behavior.
     */
    constexpr VariableLazyMap() : _data{nullptr}, _variable{} {
        Unreachable();
    }

    /**
     * @brief Construct variable lazy map with a constant underlying data container and a variable.
     *
     * This constructor creates a VariableLazyMap with a given constant VectorX<_Scalar> as the
     * underlying data and a Variable object representing the variable being mapped. The sizes of
     * the data container and the variable must match, and the variable must have an index of 0. The
     * resulting map has only const member functions.
     *
     * @param[in] underlying VectorX<_Scalar> representing the underlying data container.
     * @param[in] var        Variable object representing the variable being mapped.
     *
     * @warning The sizes of the underlying data and the Variable must match, and the variable's index
     *          must be 0. Otherwise, calling this constructor results in an assertion failure.
     */
    VariableLazyMap(const VectorX<_Scalar>& underlying, const _Variable& var)
        : _data{underlying.data()}, _variable{var} {
        UNGAR_ASSERT(underlying.size() == var.Size() && !var.Index());
    }

    /**
     * @brief Construct variable lazy map with an lvalue VectorX<_Scalar> and a variable.
     *
     * This constructor creates a variable lazy map with a given lvalue VectorX<_Scalar> as the
     * underlying data and a Variable object representing the variable being mapped. The sizes
     * of the data container and the variable must match, and the variable must have an index of 0.
     * The resulting map has both const and non-const member functions.
     *
     * @param[in] underlying Lvalue VectorX<_Scalar> representing the underlying data container.
     * @param[in] var        Variable object representing the variable being mapped.
     *
     * @warning The sizes of the underlying data and the variable must match, and the variable's index
     *          must be 0. Otherwise, calling this constructor results in an assertion failure.
     */
    VariableLazyMap(VectorX<_Scalar>& underlying, const _Variable& var)
        : _data{underlying.data()}, _variable{var} {
        UNGAR_ASSERT(underlying.size() == var.Size() && !var.Index());
    }

    /**
     * @brief Construct variable lazy map with a constant lvalue data container and a variable.
     *
     * This constructor creates a variable lazy map with a given constant lvalue data container,
     * `_Underlying`, and a Variable object representing the variable being mapped. The sizes of
     * the data container and the variable must match, and the variable must have an index of 0.
     * The resulting map has only const member functions.
     *
     * @tparam _Underlying      Type of the constant lvalue data container.
     * @param[in] underlying    Constant lvalue data container.
     * @param[in] var           Variable object representing the variable being mapped.
     *
     * @warning The sizes of the underlying data and the variable must match, and the variable's index
     *          must be 0. Otherwise, calling this constructor results in an assertion failure.
     */
    template <typename _Underlying>
    constexpr VariableLazyMap(const _Underlying& underlying, const _Variable& var)
        requires(UnderlyingSize(hana::type_c<_Underlying>) ==
                 static_cast<size_t>(_Variable::Size()))
        : _data{std::ranges::data(underlying)}, _variable{var} {
        UNGAR_ASSERT(!var.Index());
    }

    /**
     * @brief Construct variable lazy map with a mutable lvalue data container and a variable.
     *
     * This constructor creates a variable lazy map with a given mutable lvalue data container,
     * `_Underlying`, and a Variable object representing the variable being mapped. The sizes of
     * the data container and the variable must match, and the Variable must have an index of 0.
     * The resulting map has both const and non-const member functions.
     *
     * @tparam _Underlying      Type of the mutable lvalue data container.
     * @param[in] underlying    Mutable lvalue data container.
     * @param[in] var           Variable object representing the variable being mapped.
     *
     * @warning The sizes of the underlying data and the variable must match, and the variable's index
     *          must be 0. Otherwise, calling this constructor results in an assertion failure.
     */
    template <typename _Underlying>
    constexpr VariableLazyMap(_Underlying& underlying, const _Variable& var)
        requires(UnderlyingSize(hana::type_c<_Underlying>) ==
                 static_cast<size_t>(_Variable::Size()))
        : _data{std::ranges::data(underlying)}, _variable{var} {
        UNGAR_ASSERT(!var.Index());
    }

    /**
     * @brief Get underlying data associated with the variable in the variable lazy map.
     *
     * This member function retrieves the underlying data associated with the variable stored in the
     * variable lazy map. The data is returned as a read-only reference. See [1] for more details.
     *
     * @return Read-only Eigen::Map to the underlying data associated with the Variable.
     *
     * @see   [1] Flavio De Vincenti and Stelian Coros. "Ungar -- A C++ Framework for
     *            Real-Time Optimal Control Using Template Metaprogramming." 2023 IEEE/RSJ
     *            International Conference on Intelligent Robots and Systems (IROS) (2023).
     */
    decltype(auto) Get() const {
        return Get1(_variable);
    }

    /**
     * @brief Get underlying data associated with the variable in the variable lazy map.
     *
     * This member function retrieves the underlying data associated with the variable stored in the
     * variable lazy map.
     *
     * @return Eigen::Map to the underlying data associated with the Variable.
     *
     * @see VariableLazyMap::Get.
     */
    decltype(auto) Get()
        requires _EnableMutableMembers::value
    {
        return Get1(_variable);
    }

    /**
     * @brief Get underlying data associated with sub-variable in the variable lazy map.
     *
     * This member function retrieves the underlying data associated with a sub-variable stored in the
     * variable lazy map. The data is returned as a read-only reference.
     *
     * @param[in] args Arguments specifying the sub-variable.
     * @return Read-only Eigen::Map to the underlying data associated with the specified Variable.
     *
     * @see VariableLazyMap::Get.
     */
    decltype(auto) Get(auto&&... args) const {
        return Get1(_variable(std::forward<decltype(args)>(args)...));
    }

    /**
     * @brief Get underlying data associated with sub-variable in the variable lazy map.
     *
     * This member function retrieves the underlying data associated with a sub-variable stored in the
     * variable lazy map.
     *
     * @param[in] args Arguments specifying the sub-variable.
     * @return Eigen::Map to the underlying data associated with the specified Variable.
     *
     * @see VariableLazyMap::Get.
     */
    decltype(auto) Get(auto&&... args)
        requires _EnableMutableMembers::value
    {
        return Get1(_variable(std::forward<decltype(args)>(args)...));
    }

    /**
     * @brief Get tuple of underlying data associated with multiple sub-variables in the variable lazy map.
     *
     * This member function retrieves a tuple of underlying data associated with multiple Variables stored
     * in the variable lazy map. The data is returned as a tuple of read-only references.
     *
     * @param[in] vars Sub-variables to be retrieved simultaneously.
     * @return Tuple of read-only Eigen::Map objects to the specified sub-variables.
     *
     * @see VariableLazyMap::Get.
     */
    decltype(auto) GetTuple(auto&&... vars) const {
        return std::tuple<decltype(Get(std::forward<decltype(vars)>(vars)))...>(
            Get(std::forward<decltype(vars)>(vars))...);
    }

    /**
     * @brief Get tuple of underlying data associated with multiple sub-variables in the variable lazy map.
     *
     * This member function retrieves a tuple of underlying data associated with multiple Variables stored
     * in the variable lazy map.
     *
     * @param[in] vars Sub-variables to be retrieved simultaneously.
     * @return Tuple of Eigen::Map objects to the specified sub-variables.
     *
     * @see VariableLazyMap::Get.
     */
    decltype(auto) GetTuple(auto&&... vars)
        requires _EnableMutableMembers::value
    {
        return std::tuple<decltype(Get(std::forward<decltype(vars)>(vars)))...>(
            Get(std::forward<decltype(vars)>(vars))...);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using \c Get and \c \GetTuple
     *       member functions).
     */
    decltype(auto) Get1(const Concepts::Variable auto& var) const {
        using VariableType = std::remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::Size() == 1_idx) {
            return GetImpl(var).get();
        } else {
            return GetImpl(var);
        }
    }

    /**
     * @todo Remove (sub-variables should only be accessed using \c Get and \c \GetTuple
     *       member functions).
     */
    decltype(auto) Get1(const Concepts::Variable auto& var) {
        using VariableType = std::remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::Size() == 1_idx) {
            return GetImpl(var).get();
        } else {
            return GetImpl(var);
        }
    }

  protected:
    auto GetImpl(const Concepts::Variable auto& var) const {
        using VariableType = std::remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::IsQuaternion()) {
            return Eigen::Map<const Quaternion<_Scalar>>{_data + var.Index()};
        } else {
            constexpr auto size = VariableType::Size();
            if constexpr (size == 1_idx) {
                return std::cref(*(_data + var.Index()));
            } else if constexpr (size <= 32_idx) {
                return Eigen::Map<const Vector<_Scalar, static_cast<int>(size)>>{_data +
                                                                                 var.Index()};
            } else {
                return Eigen::Map<const VectorX<_Scalar>>{_data + var.Index(), size};
            }
        }
    }

    auto GetImpl(const Concepts::Variable auto& var)
        requires _EnableMutableMembers::value
    {
        using VariableType = std::remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::IsQuaternion()) {
            return Eigen::Map<Quaternion<_Scalar>>{const_cast<_Scalar*>(_data) + var.Index()};
        } else {
            constexpr auto size = VariableType::Size();
            if constexpr (size == 1_idx) {
                return std::ref(*(const_cast<_Scalar*>(_data) + var.Index()));
            } else if constexpr (size <= 32_idx) {
                return Eigen::Map<Vector<_Scalar, static_cast<int>(size)>>{
                    const_cast<_Scalar*>(_data) + var.Index()};
            } else {
                return Eigen::Map<VectorX<_Scalar>>{const_cast<_Scalar*>(_data) + var.Index(),
                                                    size};
            }
        }
    }

  private:
    template <typename _Sc, Concepts::Variable _Var>
    friend class VariableMap;

    const _Scalar* _data;
    _Variable _variable;
};

VariableLazyMap(const auto& underlying, const Concepts::Variable auto& var)
    -> VariableLazyMap<std::remove_cvref_t<decltype(*std::ranges::data(underlying))>,
                       std::remove_cvref_t<decltype(var)>,
                       hana::bool_<false>>;

VariableLazyMap(auto& underlying, const Concepts::Variable auto& var)
    -> VariableLazyMap<std::remove_reference_t<decltype(*std::ranges::data(underlying))>,
                       std::remove_cvref_t<decltype(var)>,
                       hana::bool_<true>>;

/**
 * @brief Create variable lazy map with a constant lvalue data container and a variable.
 *
 * This function creates a variable lazy map with a given constant lvalue data container, `underlying`,
 * and a Variable object representing the variable being mapped. The sizes of the data container
 * and the variable must match, and the variable must have an index of 0.
 *
 * @param[in] underlying Constant lvalue data container.
 * @param[in] var        Variable object representing the variable being mapped.
 * @return VariableLazyMap object with the specified data container and variable.
 *
 * @warning The sizes of the underlying data and the variable must match, and the variable's index
 *          must be 0. Otherwise, creating the variable lazy map results in an assertion failure.
 */
inline static auto MakeVariableLazyMap(const auto& underlying, const Concepts::Variable auto& var) {
    return VariableLazyMap{underlying, var};
}

/**
 * @brief Create variable lazy map with a mutable lvalue data container and a variable.
 *
 * This function creates a variable lazy map with a given mutable lvalue data container, `underlying`,
 * and a Variable object representing the variable being mapped. The sizes of the data container
 * and the variable must match, and the variable must have an index of 0.
 *
 * @param[in] underlying The mutable lvalue data container.
 * @param[in] var        The Variable object representing the variable being mapped.
 * @return VariableLazyMap object with the specified data container and Variable.
 *
 * @warning The sizes of the underlying data and the variable must match, and the variable's index
 *          must be 0. Otherwise, creating the VariableLazyMap will result in an assertion failure.
 */
inline static auto MakeVariableLazyMap(auto& underlying, const Concepts::Variable auto& var) {
    return VariableLazyMap{underlying, var};
}

}  // namespace Ungar

#endif /* _UNGAR__VARIABLE_LAZY_MAP_HPP_ */
