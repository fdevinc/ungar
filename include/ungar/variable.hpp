/******************************************************************************
 *
 * @file ungar/variable.hpp
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

#ifndef _UNGAR__VARIABLE_HPP_
#define _UNGAR__VARIABLE_HPP_

#include "ungar/utils/utils.hpp"

namespace Ungar {

template <Concepts::HanaString _N, Concepts::HanaMap _M, index_t _S>
struct Variable;

template <class _T>
struct is_variable : std::false_type {};
template <Concepts::HanaString _N, Concepts::HanaMap _M, index_t _S>
struct is_variable<Variable<_N, _M, _S>> : std::true_type {};
template <class _T>
constexpr bool is_variable_v = is_variable<_T>::value;

namespace Concepts {
template <typename _Variable>
concept Variable = is_variable_v<_Variable>;
}

template <Concepts::HanaIntegralConstant _Count, Concepts::Variable _Variable>
struct VariableProductExpr {
    constexpr VariableProductExpr(_Count cnt_, _Variable var_) : cnt{cnt_}, var{var_} {
    }

    _Count cnt;
    _Variable var;
};

template <Concepts::HanaIntegralConstant _Count, Concepts::Variable _Variable>
VariableProductExpr(_Count cnt_, _Variable var_) -> VariableProductExpr<_Count, _Variable>;

template <class _T>
struct is_variable_product_expr : std::false_type {};
template <Concepts::HanaIntegralConstant _Count, Concepts::Variable _Variable>
struct is_variable_product_expr<VariableProductExpr<_Count, _Variable>> : std::true_type {};
template <class _T>
constexpr bool is_variable_product_expr_v = is_variable_product_expr<_T>::value;

namespace Concepts {
template <typename _Variable>
concept VariableProductExpr = is_variable_product_expr_v<_Variable>;
}

/**
 * @brief A class representing a variable with a name, size, and optional sub-variables.
 *
 * The `Variable` class represents a variable that can be part of a variable hierarchy.
 * Each variable has a name, a size, and it can be either a "leaf" variable (with no sub-variables)
 * or a "branch" variable (with sub-variables). A "leaf" variable represents a scalar, a vector,
 * or a unit quaternion, while a "branch" variable represents a collection of sub-variables.
 *
 * @tparam _Name The name of the variable, represented as a Boost.Hana string.
 * @tparam _Map  The map containing sub-variables if the variable is a "branch" variable.
 * @tparam _SIZE The size of the variable if it is a "leaf" variable, 0 if it is a "branch"
 *               variable.
 *
 * @warning The size of a "branch" variable is always 0, regardless of the actual number of
 *          sub-variables it has. For "leaf" variables, the size may be 1 for scalars, an
 *          implementation-defined constant for unit quaternions, or any other positive number
 *           for vectors.
 */
template <Concepts::HanaString _Name,
          Concepts::HanaMap _Map = decltype(hana::make_map()),
          index_t _SIZE          = 0_idx>
class Variable {
  public:
    /**
     * @brief Create a variable with both size and index 0.
     */
    constexpr Variable() : _index{0_idx} {
    }

    /**
     * @brief Create a variable with a given name, and both size and index 0.
     *
     * @param[in] name Name of the variable with type \c boost::hana::string.
     */
    constexpr Variable(_Name /* name */) : _index{0_idx} {
    }

    /**
     * @brief Create a variable with a given name and size, and index 0.
     *
     * @param[in] name Name of the variable with type \c boost::hana::string.
     * @param[in] size Size of the variable as a Boost.Hana integral constant.
     */
    constexpr Variable(_Name /* name */, Concepts::HanaIntegralConstant auto /* size */)
        : _index{0_idx} {
    }

    /**
     * @brief Get the name of the variable as a \c boost::hana::string.
     *
     * @return Name of the variable.
     */
    static constexpr auto Name() {
        return _Name{};
    }

    /**
     * @brief Get the size of the variable. If it is a "leaf" variable, the size is
     *        1 for scalars, 4 for unit quaternions, or \a n for an n-dimensional
     *        vector. If it is a "branch" variable, the size is the sum of the sizes
     *        of all sub-variables.
     *
     * @return The size of the variable.
     *
     * @note  The returned size is not always identical to the \c _SIZE template
     *        parameter of the Variable class. Indeed, "branch" variables always
     *        have \c _SIZE equal to 0, while unit quaternions have \c _SIZE equal
     *        to an implementation-defined value.
     */
    static constexpr auto Size() {
        return SizeC().value;
    }

    /**
     * @brief Get the index of the variable within a variable hierarchy. The index of
     *        a variable is the sum of the sizes of all the preceding variables in the
     *        hierarchy.
     *
     * @return The index of the variable.
     */
    constexpr auto Index() const {
        return _index;
    }

    /**
     * @brief Query whether the variable is a "leaf", i.e., if it does not have any
     *        sub-variables.
     *
     * @return True if and only if the variable is a "leaf" variable.
     *
     * @note  Variables with 0 size are conventionally considered as "branch"
     *        variables even if they have no sub-variables.
     */
    static constexpr bool IsLeaf() {
        static_assert(hana::is_empty(hana::keys(DefaultMap())) || !_SIZE);
        return !!_SIZE;
    }

    /**
     * @brief Query whether the variable is a "branch", i.e., if it has at least one
     *        sub-variable.
     *
     * @return True if and only if the variable is a "leaf" variable.
     *
     * @note  Variables with 0 size are conventionally considered as "branch"
     *        variables even if they have no sub-variables.
     */
    static constexpr bool IsBranch() {
        return !_SIZE;
    }

    /**
     * @brief Query whether the variable represents a scalar quantity.
     *
     * @return True if and only if the variable has size 1.
     */
    static constexpr bool IsScalar() {
        return _SIZE == 1_idx;
    }

    /**
     * @brief Query whether the variable represents a vector quantity.
     *
     * @return True if and only if the variable is neither a scalar, nor
     *         a unit quaternion.
     */
    static constexpr bool IsVector() {
        return (!IsScalar()) && (!IsQuaternion());
    }

    /**
     * @brief Query whether the variable represents a unit quaternion.
     *
     * @return True if and only if the variable has \c _SIZE equal to
     *         the implementation-defined constant \c Q.
     */
    static constexpr bool IsQuaternion() {
        return _SIZE == Q;
    }

    /**
     * @brief Add a sub-variable to the variable hierarchy. The new sub-variable
     *        is always added at the end of the current hierarchy, so its index
     *        is equal to \c Index() + \c Size().
     *
     * @param[in] var Sub-variable to be added to the current variable hierarchy.
     * @return New "branch" variable with an additional sub-variable in the
     *         corresponding hierarchy.
     */
    constexpr auto operator<<=(Concepts::Variable auto var) const requires(IsBranch()) {
        auto map = hana::insert(
            _map, hana::make_pair(var.Name(), var.CloneWithIndexOffset(Index() + SizeC())));
        return Make(Name(), map, Index(), 0_c);
    }

    /**
     * @brief Add an array of identical sub-variables to the variable hierarchy.
     *        The new sub-variables are always added at the end of the current
     *        hierarchy, so their indices start at \c Index() + \c Size().
     *
     * @param[in] prodExpr Product expression containing the sub-variable to be
     *                     added and the number of its copies. A product expression
     *                     is the result of the multiplication of variable by an
     *                     integral constant.
     * @return New "branch" variable with an additional array of identical
     *         sub-variables in the corresponding hierarchy.
     */
    constexpr auto operator<<=(Concepts::VariableProductExpr auto prodExpr) const
        requires(IsBranch()) {
        auto array = hana::unpack(hana::make_range(0_c, prodExpr.cnt), [&](auto... is) {
            return std::array{prodExpr.var.CloneWithIndexOffset(Index() + SizeC() +
                                                                prodExpr.var.SizeC() * is)...};
        });
        auto map   = hana::insert(_map, hana::make_pair(prodExpr.var.Name(), array));
        return Make(Name(), map, Index(), 0_c);
    }

    /**
     * @brief Add multiple sub-variables and/or arrays thereof to the variable
     *        hierarchy. The new sub-variables are always added at the end of
     *        the current hierarchy, so their indices start at \c Index() +
     *        \c Size(). Also, the sub-variables are always appended at the end
     *        of the hierarchy in the same order of the input tuple.
     *
     * @param[in] tuple Tuple of sub-variables or product expressions to be
     *                  added to the current variable hierarchy.
     * @return New "branch" variable with an additional sub-variable and/or
     *         array of identical sub-variables for each element in the input
     *         tuple.
     */
    constexpr auto operator<<=(Concepts::HanaTuple auto tuple) const requires(IsBranch()) {
        auto lambda = [&](auto var, auto el) { return var <<= el; };
        return hana::fold(tuple, *this, lambda);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    template <Concepts::HanaString _N>
    constexpr const auto& Get(_N name) const
        requires(hana::contains(hana::keys(_Map{}), _N{}).value) {
        if constexpr (!is_std_array_v<std::remove_cvref_t<decltype(DefaultMap()[_N{}])>>) {
            return _map[name];
        } else {
            Unreachable();
        }
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    constexpr const auto& Get(Concepts::Variable auto var) const {
        return Get(var.Name());
    }

    /**
     * @brief Access sub-variable by its name.
     *
     * @tparam _NAME Name of the sub-variable to be accessed.
     * @return Constant reference to the sub-variable with name _NAME, or
     *         triggers an error if such a sub-variable does not exist.
     */
    template <fixed_string _NAME>
    constexpr const auto& At() const {
        return Get(string_c<_NAME>);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    template <Concepts::HanaString _N>
    constexpr const auto& Get(_N name, const std::integral auto i) const {
        if constexpr (is_std_array_v<std::remove_cvref_t<decltype(DefaultMap()[_N{}])>>) {
            return _map[name][static_cast<size_t>(i)];
        } else {
            Unreachable();
        }
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    template <Concepts::HanaString _N>
    constexpr const auto& Get(_N name, Concepts::HanaIntegralConstant auto i) const {
        return Get(name, i.value);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    constexpr const auto& Get(Concepts::Variable auto var, const std::integral auto i) const {
        return Get(var.Name(), i);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    constexpr const auto& Get(Concepts::Variable auto var,
                              Concepts::HanaIntegralConstant auto i) const {
        return Get(var.Name(), i.value);
    }

    /**
     * @brief Access sub-variable by its name and index in the array to
     *        which it belongs.
     *
     * @tparam _NAME Name of the sub-variable to be accessed.
     * @param[in] i  Index of the sub-variable in the array to which it
     *               belongs.
     * @return Constant reference to the sub-variable with name _NAME, or
     *         triggers an error if such a sub-variable does not exist.
     */
    template <fixed_string _NAME>
    constexpr const auto& At(const std::integral auto i) const {
        return Get(string_c<_NAME>, i);
    }

    /**
     * @brief Access sub-variable by its name and index in the array to
     *        which it belongs.
     *
     * @tparam _NAME Name of the sub-variable to be accessed.
     * @param[in] i  Index of the sub-variable in the array to which it
     *               belongs (given as a Boost.Hana integral constant).
     * @return Constant reference to the sub-variable with name _NAME, or
     *         triggers an error if such a sub-variable does not exist.
     */
    template <fixed_string _NAME>
    constexpr const auto& At(Concepts::HanaIntegralConstant auto i) const {
        return Get(string_c<_NAME>, i.value);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    constexpr const auto& Get(auto key, auto... args) const
        requires(hana::contains(hana::keys(_Map{}), decltype(key)::Name()).value) {
        return Get(key).Get(args...);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    constexpr const auto& Get(auto key, const std::integral auto i, auto... args) const {
        return Get(key, i).Get(args...);
    }

    /**
     * @todo Remove (sub-variables should only be accessed using the \c operator()
     *       or \c At member functions).
     */
    constexpr const auto& Get(auto key, Concepts::HanaIntegralConstant auto i, auto... args) const {
        return Get(key, i.value).Get(args...);
    }

    /**
     * @brief Access sub-variable. The input constitutes a sequence of sub-variables
     *        (and/or sub-variable and corresponding index if it appears in multiple
     *        copies) that unambiguously leads from the hierarchy root to the
     *        sub-variable of interest. For example, given some variables \c a0 and \c b0:
     *                  UNGAR_VARIABLE(a) <<= a0;
     *                  UNGAR_VARIABLE(b) <<= b0;
     *                  UNGAR_VARIABLE(c) <<= b0;
     *                  UNGAR_VARIABLE(d) <<= 4_c * a0;
     *                  UNGAR_VARIABLE(x) <<= (a, b, c, d);
     *                  x(a0);              // OK, access x -> a -> a0.
     *                  x(a0, 2);           // OK, access x -> d -> [_ _ a0 _].
     *                  x(b, b0);           // OK, access x -> b -> b0.
     *                  x(c, b0);           // OK, access x -> c -> b0.
     *                  // x(b0);           // Error, ambiguous access!
     *        Refer to [1] for more details.
     *
     * @param[in] var   First sub-variable in the input sequence.
     * @param[in]
     * @param[in] args  Remaining sub-variables and corresponding indices if they appear
     *                  in multiple copies.
     * @return Constant reference to the sub-variable of interest.
     *
     * @see   [1] Flavio De Vincenti and Stelian Coros. "Ungar -- A C++ Framework for
     *            Real-Time Optimal Control Using Template Metaprogramming." 2023 IEEE/RSJ
     *            International Conference on Intelligent Robots and Systems (IROS) (2023).
     */
    constexpr decltype(auto) operator()(Concepts::Variable auto var, auto... args) const {
        if constexpr (sizeof...(args)) {
            return BypassImpl(var.Name()).value().get()(args...);
        } else {
            return BypassImpl(var.Name()).value().get();
        }
    }

    /**
     * @brief Access sub-variable.
     *
     * @return Constant reference to the sub-variable of interest.
     *
     * @see   Variable::operator().
     */
    constexpr decltype(auto) operator()(Concepts::Variable auto var,
                                        const std::integral auto i,
                                        auto... args) const {
        if constexpr (sizeof...(args)) {
            return BypassImpl(var.Name(), i).value().get()(args...);
        } else {
            return BypassImpl(var.Name(), i).value().get();
        }
    }

    /**
     * @brief Access sub-variable.
     *
     * @return Constant reference to the sub-variable of interest.
     *
     * @see   Variable::operator().
     */
    constexpr decltype(auto) operator()(Concepts::Variable auto var,
                                        Concepts::HanaIntegralConstant auto i,
                                        auto... args) const {
        if constexpr (sizeof...(args)) {
            return BypassImpl(var.Name(), i.value).value().get()(args...);
        } else {
            return BypassImpl(var.Name(), i.value).value().get();
        }
    }

    /**
     * @brief Access sub-variable.
     *
     * @return Constant reference to the sub-variable of interest.
     *
     * @see   Variable::operator().
     */
    constexpr decltype(auto) operator()(Concepts::Variable auto var,
                                        const std::integral auto i1,
                                        const std::integral auto i2,
                                        auto... args) const {
        if constexpr (sizeof...(args)) {
            return BypassImpl(var.Name(), i1, i2).value().get()(args...);
        } else {
            return BypassImpl(var.Name(), i1, i2).value().get();
        }
    }

    /**
     * @brief Access sub-variable.
     *
     * @return Constant reference to the sub-variable of interest.
     *
     * @see   Variable::operator().
     */
    constexpr decltype(auto) operator()(Concepts::Variable auto var,
                                        const std::integral auto i1,
                                        const std::integral auto i2,
                                        const std::integral auto i3,
                                        auto... args) const {
        if constexpr (sizeof...(args)) {
            return BypassImpl(var.Name(), i1, i2, i3).value().get()(args...);
        } else {
            return BypassImpl(var.Name(), i1, i2, i3).value().get();
        }
    }

    /**
     * @brief Apply functor with state to a variable and all its sub-variables.
     *        The input functor must accept a state and a variable, and it must
     *        return the updated state.
     *
     * @tparam _State   Type of the initial state.
     * @tparam _F       Type of the functor.
     * @param[in] state Initial state.
     * @param[in] f     Functor.
     * @return Final state after the functor is called on all sub-variables in
     *         the hierarchy.
     */
    template <typename _State, std::invocable<_State, Variable> _F>
    auto Fold(_State&& state, _F&& f) const {
        return hana::fold(
            hana::values(_map),
            std::forward<_F>(f)(std::forward<_State>(state), *this),
            [&](auto&& s1, const auto& value) {
                if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                    return hana::fold(
                        value, std::forward<decltype(s1)>(s1), [&](auto&& s2, const auto& var) {
                            return var.Fold(std::forward<decltype(s2)>(s2), std::forward<_F>(f));
                        });
                } else {
                    return value.Fold(std::forward<decltype(s1)>(s1), std::forward<_F>(f));
                }
            });
    }

    /**
     * @brief Equality comparison operator for variables. Two variables are considered equal
     *        if they are of the same type.
     *
     * @param[in] var The variable to compare with.
     * @return Boolean integral constant equal to true if and only if the variables are of
     *         the same type.
     *
     * @note The comparison is based only on the types of the variables, not their indices.
     */
    constexpr auto operator==(Concepts::Variable auto var) const {
        return boost::hana::bool_c<std::same_as<Variable, decltype(var)>>;
    }

    /**
     * @brief Inequality comparison operator for variables. Two variables are considered
     *        unequal if they are not of the same type.
     *
     * @param[in] var The variable to compare with.
     * @return Boolean integral constant equal to true if and only if the variables are not
     *         of the same type.
     *
     * @note The comparison is based only on the types of the variables, not their indices.
     */
    constexpr auto operator!=(Concepts::Variable auto var) const {
        return boost::hana::bool_c<!std::same_as<Variable, decltype(var)>>;
    }

    /**
     * @brief Apply stateless functor to a variable and all its sub-variables.
     *        The input functor must accept a variable and its output is always
     *        discarded.
     *
     * @tparam _F       Type of the functor.
     * @param[in] f     Functor.
     */
    template <std::invocable<Variable> _F>
    constexpr void ForEach(_F&& f) const {
        std::forward<_F>(f)(*this);
        hana::for_each(hana::values(_map), [&](const auto& value) {
            if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                for (const auto& var : value) {
                    var.ForEach(std::forward<_F>(f));
                }
            } else {
                value.ForEach(std::forward<_F>(f));
            }
        });
    }

  private:
    template <Concepts::HanaString _N, Concepts::HanaMap _M, index_t _S>
    friend class Variable;
    template <typename _Scalar, Concepts::Variable _Variable>
    friend class VariableMap;

    static constexpr auto DefaultMap() {
        return Variable{}._map;
    }

    template <Concepts::HanaString _N, Concepts::HanaMap _M, Concepts::HanaIntegralConstant _S>
    static constexpr auto Make(_N name, _M map, const index_t index, _S /* size */) {
        return Variable<_N, _M, _S::value>{name, map, index};
    }

    constexpr Variable(_Name /* name */, _Map map, const index_t index) : _map{map}, _index{index} {
    }

    constexpr auto Clone() const {
        return Variable{*this};
    }

    constexpr auto CloneWithIndexOffset(const index_t offset) const {
        Variable var = Clone();
        var.ForEach([offset](auto& v) { v._index += offset; });
        return var;
    }

    constexpr auto Clone(auto&& /* dummy */) const {
        return Variable{*this};
    }

    static constexpr auto SizeC() {
        if constexpr (IsLeaf()) {
            if constexpr (IsQuaternion()) {
                return 4_c;
            } else {
                return hana::llong_c<_SIZE>;
            }
        } else {
            return hana::fold(hana::values(DefaultMap()), 0_c, [](auto size, auto value) {
                if constexpr (is_std_array_v<decltype(value)>) {
                    if constexpr (value.empty()) {
                        return size;
                    } else {
                        return size + hana::llong_c<static_cast<long long>(value.size())> *
                                          value.front().SizeC();
                    }
                } else {
                    return size + value.SizeC();
                }
            });
        }
    }

    template <std::invocable<Variable&> _F>
    constexpr void ForEach(_F&& f) {
        std::forward<_F>(f)(*this);

        if constexpr (IsLeaf()) {
            return;
        } else {
            hana::for_each(hana::keys(_map), [&](const auto& key) {
                auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                    for (auto& var : value) {
                        var.ForEach(std::forward<_F>(f));
                    }
                } else {
                    value.ForEach(std::forward<_F>(f));
                }
            });
        }
    }

    template <typename _State, std::invocable<_State, Variable> _F>
    auto FoldUnique(_State&& state, _F&& f) const {
        return hana::fold(hana::values(_map),
                          std::forward<_F>(f)(std::forward<_State>(state), *this),
                          [&](auto&& s1, const auto& value) {
                              if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                                  if (!value.empty()) {
                                      return value.front().FoldUnique(
                                          std::forward<decltype(s1)>(s1), std::forward<_F>(f));
                                  } else {
                                      Unreachable();
                                  }
                              } else {
                                  return value.FoldUnique(std::forward<decltype(s1)>(s1),
                                                          std::forward<_F>(f));
                              }
                          });
    }

    template <std::invocable<Variable> _F>
    constexpr void ForEachUnique(_F&& f) const {
        std::forward<_F>(f)(*this);
        hana::for_each(hana::values(_map), [&](const auto& value) {
            if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                if (!value.empty()) {
                    value.front().ForEach(std::forward<_F>(f));
                }
            } else {
                value.ForEach(std::forward<_F>(f));
            }
        });
    }

    static constexpr auto Union(Concepts::HanaSet auto s1,
                                Concepts::HanaSet auto s2,
                                Concepts::HanaSet auto... others) {
        return Union(hana::union_(s1, s2), others...);
    }

    static constexpr auto Union(Concepts::HanaSet auto set) {
        return set;
    }

    template <std::invocable<Variable> _Proj = hana::id_t>
    auto Unique(_Proj&& proj = hana::id) const {
        if constexpr (IsLeaf()) {
            return hana::make_set(std::forward<_Proj>(proj)(Variable{}));
        } else {
            return hana::unpack(hana::values(_map), [&](auto... values) {
                auto l = [&](auto v) {
                    if constexpr (is_std_array_v<std::remove_cvref_t<decltype(v)>>) {
                        if constexpr (!decltype(v){}.empty()) {
                            return v.front().Unique(std::forward<_Proj>(proj));
                        } else {
                            Unreachable();
                        }
                    } else {
                        return v.Unique(std::forward<_Proj>(proj));
                    }
                };
                return Union(hana::make_set(std::forward<_Proj>(proj)(Variable{})), l(values)...);
            });
        }
    }

    template <Concepts::HanaString _N>
    constexpr Concepts::HanaOptional auto BypassImpl(_N name) const {
        if constexpr (Name() == name) {
            return hana::just(std::cref(*this));
        } else if constexpr (IsLeaf()) {
            return hana::nothing;
        } else {
            auto candidates = hana::transform(hana::keys(_map), [&](const auto& key) {
                const auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                    return hana::nothing;
                } else {
                    return value.BypassImpl(name);
                }
            });
            static_assert(hana::count_if(candidates, hana::is_just) <= hana::size_c<1UL>,
                          "To prevent ambiguous bypasses, a variable must appear only once as a "
                          "sub-variable.");
            return hana::flatten(hana::find_if(candidates, hana::is_just));
        }
    }

    template <Concepts::HanaString _N>
    static constexpr Concepts::HanaBool auto IsParentVariableOf(_N name) {
        if constexpr (Name() == name) {
            return hana::true_c;
        } else if constexpr (IsLeaf()) {
            return hana::false_c;
        } else {
            auto candidates = hana::transform(hana::values(DefaultMap()), [=](auto value) {
                if constexpr (is_std_array_v<decltype(value)>) {
                    return hana::false_c;
                } else {
                    return value.IsParentVariableOf(name);
                }
            });
            static_assert(hana::count_if(candidates, hana::id) <= hana::size_c<1UL>,
                          "There is a unique sub-variable only if such a variable appears once.");
            return hana::is_just(hana::find(candidates, hana::true_c));
        }
    }

    constexpr Concepts::HanaOptional auto BypassImpl(Concepts::HanaString auto name,
                                                     const std::integral auto i) const {
        if constexpr (IsLeaf()) {
            return hana::nothing;
        } else {
            auto candidates = hana::transform(hana::keys(_map), [&](const auto& key) {
                const auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                    if constexpr (std::remove_cvref_t<decltype(value)>::value_type::
                                      IsParentVariableOf(std::remove_cvref_t<decltype(name)>{})) {
                        return value[static_cast<size_t>(i)].BypassImpl(name);
                    } else {
                        return hana::nothing;
                    }
                } else {
                    return value.BypassImpl(name, i);
                }
            });
            static_assert(hana::count_if(candidates, hana::is_just) <= hana::size_c<1UL>,
                          "To prevent ambiguous bypasses, a variable must appear only once as a "
                          "sub-variable.");
            return hana::flatten(hana::find_if(candidates, hana::is_just));
        }
    }

    constexpr Concepts::HanaOptional auto BypassImpl(Concepts::HanaString auto name,
                                                     const std::integral auto i1,
                                                     const std::integral auto i2) const {
        if constexpr (IsLeaf()) {
            return hana::nothing;
        } else {
            auto candidates = hana::transform(hana::keys(_map), [&](const auto& key) {
                const auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                    if constexpr (std::remove_cvref_t<decltype(value)>::value_type::
                                      IsParentVariableOf(std::remove_cvref_t<decltype(name)>{},
                                                         decltype(i2){})) {
                        return value[static_cast<size_t>(i1)].BypassImpl(name, i2);
                    } else {
                        return hana::nothing;
                    }
                } else {
                    return value.BypassImpl(name, i1, i2);
                }
            });
            static_assert(hana::count_if(candidates, hana::is_just) <= hana::size_c<1UL>,
                          "To prevent ambiguous bypasses, a variable must appear only once as a "
                          "sub-variable.");
            return hana::flatten(hana::find_if(candidates, hana::is_just));
        }
    }

    constexpr Concepts::HanaOptional auto BypassImpl(Concepts::HanaString auto name,
                                                     const std::integral auto i1,
                                                     const std::integral auto i2,
                                                     const std::integral auto i3) const {
        if constexpr (IsLeaf()) {
            return hana::nothing;
        } else {
            auto candidates = hana::transform(hana::keys(_map), [&](const auto& key) {
                const auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<std::remove_cvref_t<decltype(value)>>) {
                    if constexpr (std::remove_cvref_t<decltype(value)>::value_type::
                                      IsParentVariableOf(std::remove_cvref_t<decltype(name)>{},
                                                         decltype(i2){},
                                                         decltype(i3){})) {
                        return value[static_cast<size_t>(i1)].BypassImpl(name, i2, i3);
                    } else {
                        return hana::nothing;
                    }
                } else {
                    return value.BypassImpl(name, i1, i2, i3);
                }
            });
            static_assert(hana::count_if(candidates, hana::is_just) <= hana::size_c<1UL>,
                          "To prevent ambiguous bypasses, a variable must appear only once as a "
                          "sub-variable.");
            return hana::flatten(hana::find_if(candidates, hana::is_just));
        }
    }

    static constexpr Concepts::HanaBool auto IsParentVariableOf(Concepts::HanaString auto name,
                                                                const std::integral auto i,
                                                                const std::integral auto... is) {
        if constexpr (IsLeaf()) {
            return hana::false_c;
        } else {
            auto candidates = hana::transform(hana::values(DefaultMap()), [=](auto value) {
                if constexpr (is_std_array_v<decltype(value)>) {
                    if constexpr (std_array_empty_v<decltype(value)>) {
                        return hana::false_c;
                    } else {
                        return decltype(value)::value_type::IsParentVariableOf(name, is...);
                    }
                } else {
                    return value.IsParentVariableOf(name, i, is...);
                }
            });
            static_assert(hana::count_if(candidates, hana::id) <= hana::size_c<1UL>,
                          "There is a unique sub-variable only if such a variable appears once.");
            return hana::is_just(hana::find(candidates, hana::true_c));
        }
    }

    _Name _name;
    _Map _map;

    index_t _index;
};

template <Concepts::HanaString _Name>
Variable(_Name name) -> Variable<_Name, decltype(hana::make_map()), 0_idx>;

template <Concepts::HanaString _Name, Concepts::HanaIntegralConstant _SIZE>
Variable(_Name name, _SIZE size) -> Variable<_Name, decltype(hana::make_map()), _SIZE::value>;

static_assert(Q != std::numeric_limits<index_t>::max() &&
                  Q_c.value != static_cast<long long>(std::numeric_limits<index_t>::max()),
              "The special value denoting the size of a unit quaternion must be different from the "
              "maximum value that index_t integers can take.");

/**
 * @brief Create variable with a specified name and, if it is a "leaf" variable, a specified size.
 *
 * @tparam _NAME Name of the variable.
 * @tparam _SIZE Size of the variable. If not provided, the created variable will be a "branch".
 * @return Variable object with the specified name and size (if provided).
 *
 * @note The returned Variable is a constexpr object.
 *
 * @warning If the _SIZE template parameter is specified, it must be a constant value known at compile time.
 *          Using a non-constant value will lead to a compilation error.
 */
template <fixed_string _NAME, index_t _SIZE = std::numeric_limits<index_t>::max()>
inline constexpr auto var_c = []() {
    if constexpr (_SIZE == std::numeric_limits<index_t>::max()) {
        return Variable{string_c<_NAME>};
    } else {
        return Variable{string_c<_NAME>, hana::llong_c<static_cast<long long>(_SIZE)>};
    }
}();

/**
 * @brief Compute product expression as the multiplication of a variable by an
 *        integral constant.
 *
 * @param[in] cnt   Integral constant denoting the number of times a variable
 *                  should be replicated.
 * @param[in] var   Variable to be multiplied.
 * @return Product expression encoding an array of identical variables.
 */
template <Concepts::HanaString _N, Concepts::HanaMap _M, index_t _S>
constexpr auto operator*(Concepts::HanaIntegralConstant auto cnt, Variable<_N, _M, _S> var) {
    return VariableProductExpr{cnt, var};
}

/**
 * @brief Concatenate two variables into a Boost.Hana tuple.
 *
 * @param[in] var1 First variable to concatenate.
 * @param[in] var2 Second variable to concatenate.
 * @return Boost.Hana tuple containing both Variables in the order they are provided.
 */
constexpr auto operator,(Concepts::Variable auto var1, Concepts::Variable auto var2) {
    return hana::make_tuple(var1, var2);
}

/**
 * @brief Append variable to a Boost.Hana tuple.
 *
 * @param[in] tuple Tuple to which the variable is appended.
 * @param[in] var   Variable to append.
 * @return Boost.Hana tuple with the variable added at the end.
 */
constexpr auto operator,(Concepts::HanaTuple auto tuple, Concepts::Variable auto var) {
    return hana::append(tuple, var);
}

/**
 * @brief Prepend variable to a Boost.Hana tuple.
 *
 * @param[in] var   Variable to prepend.
 * @param[in] tuple Boost.Hana tuple to which the variable is prepended.
 * @return Boost.Hana tuple with the variable added at the beginning.
 */
constexpr auto operator,(Concepts::Variable auto var, Concepts::HanaTuple auto tuple) {
    return hana::prepend(tuple, var);
}

/**
 * @brief Concatenate product expression and variable into a Boost.Hana tuple.
 *
 * @param[in] prodExpr Product expression to concatenate.
 * @param[in] var      Variable to concatenate.
 * @return Boost.Hana tuple containing both the product expression and variable in
 *         the order they are provided.
 */
constexpr auto operator,(Concepts::VariableProductExpr auto prodExpr, Concepts::Variable auto var) {
    return hana::make_tuple(prodExpr, var);
}

/**
 * @brief Concatenate variable and product expression into a Boost.Hana tuple.
 *
 * @param[in] var      Variable to concatenate.
 * @param[in] prodExpr Product expression to concatenate.
 * @return Boost.Hana tuple containing both the variable and product expression in
 *         the order they are provided.
 */
constexpr auto operator,(Concepts::Variable auto var, Concepts::VariableProductExpr auto prodExpr) {
    return hana::make_tuple(var, prodExpr);
}

/**
 * @brief Concatenate two product expressions into a Boost.Hana tuple.
 *
 * @param[in] prodExpr1 First product expression to concatenate.
 * @param[in] prodExpr2 Second product expression to concatenate.
 * @return Boost.Hana tuple containing both product expressions in the order
 *         they are provided.
 */
constexpr auto operator,(Concepts::VariableProductExpr auto prodExpr1,
                         Concepts::VariableProductExpr auto prodExpr2) {
    return hana::make_tuple(prodExpr1, prodExpr2);
}

/**
 * @brief Appends product expression to a Boost.Hana tuple.
 *
 * @param[in] tuple    Boost.Hana tuple to which the product expression is appended.
 * @param[in] prodExpr Product expression to append.
 * @return Boost.Hana tuple with the product expression added at the end.
 */
constexpr auto operator,(Concepts::HanaTuple auto tuple,
                         Concepts::VariableProductExpr auto prodExpr) {
    return hana::append(tuple, prodExpr);
}

/**
 * @brief Prepends product expression to a Boost.Hana tuple.
 *
 * @param[in] prodExpr Product expression to prepend.
 * @param[in] tuple    Boost.Hana tuple to which the product expression is prepended.
 * @return Boost.Hana tuple with the product expression added at the beginning.
 */
constexpr auto operator,(Concepts::VariableProductExpr auto prodExpr,
                         Concepts::HanaTuple auto tuple) {
    return hana::prepend(tuple, prodExpr);
}

#define UNGAR_LEAF_VARIABLE(name, size) constexpr auto name = ::Ungar::var_c<#name, size>
#define UNGAR_BRANCH_VARIABLE(name) constexpr auto name = ::Ungar::var_c<#name>
#define UNGAR_VARIABLE(name, ...) \
    constexpr auto name = ::Ungar::var_c<#name __VA_OPT__(, __VA_ARGS__)>

}  // namespace Ungar

#endif /* _UNGAR__VARIABLE_HPP_ */
