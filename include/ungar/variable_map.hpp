/******************************************************************************
 *
 * @file ungar/variable_map.hpp
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

#ifndef _UNGAR__VARIABLE_MAP_HPP_
#define _UNGAR__VARIABLE_MAP_HPP_

#include "ungar/utils/integral_map.hpp"
#include "ungar/variable_lazy_map.hpp"

namespace Ungar {

/**
 * @brief Class representing a mapping between a variable and an Eigen vector.
 *
 * @tparam _Scalar      Scalar type for the underlying vector.
 * @tparam _Variable    Variable type representing the variable being mapped.
 */
template <typename _Scalar, Concepts::Variable _Variable>
class VariableMap : private VectorX<_Scalar>,
                    private VariableLazyMap<_Scalar, _Variable, hana::true_> {
  private:
    using VectorType   = VectorX<_Scalar>;
    using LazyMapType  = VariableLazyMap<_Scalar, _Variable, hana::true_>;
    using VariableType = _Variable;

    // Helper function to calculate the implementation-defined size of a variable.
    template <Concepts::Variable _Var>
    static constexpr auto SizeCImpl(const _Var& var) {
        if constexpr (_Var::IsQuaternion()) {
            return Q_c;
        } else {
            return var.SizeC();
        }
    }

    static auto InitImpl(LazyMapType lazyMap,
                         const VariableType& var,
                         Concepts::HanaBool auto initConstMap) {
        add_const_if_t<LazyMapType, initConstMap>& lm{lazyMap};

        auto unique = var.Unique([](const auto& v) { return SizeCImpl(v); });
        auto map    = hana::unpack(unique, [](auto... sizes) {
            return hana::make_map(hana::make_pair(
                sizes,
                integral_map<decltype(lm.GetImpl(Variable{hana::string_c<>, sizes})),
                             static_cast<size_t>(VariableType::Size())>{})...);
        });
        var.ForEach([&](const auto& v) {
            static_assert(decltype(hana::contains(map, SizeCImpl(v)))::value);
            map[SizeCImpl(v)].emplace(v.Index(), lm.GetImpl(v));
        });
        return map;
    }

    LazyMapType& AsLazyMap() {
        return static_cast<LazyMapType&>(*this);
    }

    const LazyMapType& AsLazyMap() const {
        return static_cast<const LazyMapType&>(*this);
    }

  public:
    using ScalarType = _Scalar;

    VariableMap(hana::basic_type<_Scalar>, const VariableType& var)
        : VectorType{var.Size()},
          LazyMapType{Get(), var},
          _impl{InitImpl(AsLazyMap(), var, hana::false_c)},
          _constImpl{InitImpl(AsLazyMap(), var, hana::true_c)} {
        UNGAR_ASSERT(!var.Index());
    }

    /**
     * @brief Get vector representation of the variable map.
     *
     * The `VariableMap` class derives from `VectorX`, allowing direct access to the underlying
     * vector data using the `Get` member function. This function returns a reference to the
     * underlying vector, allowing access to its elements.
     *
     * @return Constant reference to the underlying vector.
     */
    decltype(auto) Get() const {
        return static_cast<const VectorType&>(*this);
    }

    /**
     * @brief Get vector representation of the variable map.
     *
     * The `VariableMap` class derives from `VectorX`, allowing direct access to the underlying
     * vector data using the `Get` member function. This function returns a reference to the
     * underlying vector, allowing access to its elements.
     *
     * @return Reference to the underlying vector.
     */
    decltype(auto) Get() {
        return static_cast<VectorType&>(*this);
    }

    /**
     * @brief Get constant reference to a sub-variable. See [1] for more details.
     *
     * @see   [1] Flavio De Vincenti and Stelian Coros. "Ungar -- A C++ Framework for
     *            Real-Time Optimal Control Using Template Metaprogramming." 2023 IEEE/RSJ
     *            International Conference on Intelligent Robots and Systems (IROS) (2023).
     */
    decltype(auto) Get(auto&&... args) const {
        return Get1(AsLazyMap()._variable(std::forward<decltype(args)>(args)...));
    }

    /**
     * @brief Get reference to a sub-variable.
     *
     * @see   VariableMap::Get.
     */
    decltype(auto) Get(auto&&... args) {
        return Get1(AsLazyMap()._variable(std::forward<decltype(args)>(args)...));
    }

    /**
     * @brief Get tuple of constant references to multiple sub-variables simultaneously.
     *
     * @see   VariableMap::Get.
     */
    decltype(auto) GetTuple(auto&&... vars) const {
        return std::forward_as_tuple(Get(std::forward<decltype(vars)>(vars))...);
    }

    /**
     * @brief Get tuple of references to multiple sub-variables simultaneously.
     *
     * @see   VariableMap::Get.
     */
    decltype(auto) GetTuple(auto&&... vars) {
        return std::forward_as_tuple(Get(std::forward<decltype(vars)>(vars))...);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using \c Get and \c \GetTuple
     *       member functions).
     */
    template <Concepts::Variable _Var>
    const auto& Get1(const _Var& var) const {
        if constexpr (decltype(hana::contains(_constImpl, SizeCImpl(var)))::value) {
            UNGAR_ASSERT(_constImpl[SizeCImpl(var)].contains(var.Index()));
            if constexpr (_Var::Size() == 1_idx) {
                return _constImpl[SizeCImpl(var)][var.Index()].get();
            } else {
                return _constImpl[SizeCImpl(var)][var.Index()];
            }
        } else {
            Unreachable();
        }
    }

    /**
     * @todo Remove (sub-variables should only be accessed using \c Get and \c \GetTuple
     *       member functions).
     */
    template <Concepts::Variable _Var>
    auto& Get1(const _Var& var) {
        if constexpr (decltype(hana::contains(_impl, SizeCImpl(var)))::value) {
            UNGAR_ASSERT(_impl[SizeCImpl(var)].contains(var.Index()));
            if constexpr (_Var::Size() == 1_idx) {
                return _impl[SizeCImpl(var)][var.Index()].get();
            } else {
                return _impl[SizeCImpl(var)][var.Index()];
            }
        } else {
            Unreachable();
        }
    }

  private:
    decltype(InitImpl(LazyMapType{}, VariableType{}, hana::false_c)) _impl;
    decltype(InitImpl(LazyMapType{}, VariableType{}, hana::true_c)) _constImpl;
};

template <typename _Scalar, Concepts::Variable _Variable>
VariableMap(const _Variable& var, hana::basic_type<_Scalar>) -> VariableMap<_Scalar, _Variable>;

/**
 * @brief Create `VariableMap` object with the specified variable and scalar type.
 *
 * This function creates a `VariableMap` object using the specified scalar type and the given
 * variable. The resulting `VariableMap` can be used to access and manipulate the values of the
 * variables using both vector-like and variable-like interfaces. The `VariableMap` provides
 * constant access to the variable values.
 *
 * @tparam _Scalar  Scalar type for the underlying vector in the `VariableMap`.
 * @param[in] var   Variable object to be used for creating the `VariableMap`.
 * @return VariableMap object that represents the provided variable with the specified
 *         scalar type.
 */
template <typename _Scalar>
inline static auto MakeVariableMap(const Concepts::Variable auto& var) {
    return VariableMap{hana::type_c<_Scalar>, var};
}

}  // namespace Ungar

#endif /* _UNGAR__VARIABLE_MAP_HPP_ */
