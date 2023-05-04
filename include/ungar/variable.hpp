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

template <Concepts::HanaString _N, Concepts::HanaMap _M, Concepts::HanaOptional _S>
struct Variable;

template <class _T>
struct is_variable : std::false_type {};
template <Concepts::HanaString _N, Concepts::HanaMap _M, Concepts::HanaOptional _S>
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

template <Concepts::HanaString _Name,
          Concepts::HanaMap _Map       = decltype(hana::make_map()),
          Concepts::HanaOptional _Size = decltype(hana::nothing)>
class Variable {
  public:
    constexpr Variable() : _index{0_idx} {
    }

    constexpr Variable(_Name /* name */) : _index{0_idx} {
    }

    constexpr Variable(_Name /* name */, Concepts::HanaIntegralConstant auto /* size */)
        : _index{0_idx} {
    }

    static constexpr auto Name() {
        return _Name{};
    }

    static constexpr auto Size() {
        return SizeC().value;
    }

    constexpr auto Index() const {
        return _index;
    }

    static constexpr bool IsLeaf() {
        static_assert(hana::is_empty(hana::keys(DefaultMap())) || !hana::is_just(_Size{}));
        return hana::is_just(_Size{});
    }

    static constexpr bool IsBranch() {
        return hana::is_nothing(_Size{});
    }

    static constexpr bool IsScalar() {
        return (_Size{} == hana::make_optional(1_c)).value;
    }

    static constexpr bool IsVector() {
        return (!IsScalar()) && (!IsQuaternion());
    }

    static constexpr bool IsQuaternion() {
        return (_Size{} == hana::make_optional(Q_c)).value;
    }

    constexpr auto operator<<=(Concepts::Variable auto var) const requires(IsBranch()) {
        auto map = hana::insert(
            _map, hana::make_pair(var.Name(), var.CloneWithIndexOffset(Index() + SizeC())));
        return Make(Name(), map, Index(), hana::nothing);
    }

    constexpr auto operator<<=(Concepts::HanaTuple auto tuple) const requires(IsBranch()) {
        auto lambda = [&](auto var, auto el) { return var <<= el; };
        return hana::fold(tuple, *this, lambda);
    }

    constexpr auto operator<<=(Concepts::VariableProductExpr auto prodExpr) const
        requires(IsBranch()) {
        auto array = hana::unpack(hana::make_range(0_c, prodExpr.cnt), [&](auto... is) {
            return std::array{prodExpr.var.CloneWithIndexOffset(Index() + SizeC() +
                                                                prodExpr.var.SizeC() * is)...};
        });
        auto map   = hana::insert(_map, hana::make_pair(prodExpr.var.Name(), array));
        return Make(Name(), map, Index(), hana::nothing);
    }

    /**
     * @todo Improve or add requirements.
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
     * @todo Improve or add requirements.
     */
    constexpr const auto& Get(Concepts::Variable auto var) const {
        return Get(var.Name());
    }

    /**
     * @todo Improve or add requirements.
     */
    template <fixed_string _NAME>
    constexpr const auto& At() const {
        return Get(string_c<_NAME>);
    }

    template <Concepts::HanaString _N>
    constexpr const auto& Get(_N name, const std::integral auto i) const {
        if constexpr (is_std_array_v<std::remove_cvref_t<decltype(DefaultMap()[_N{}])>>) {
            return _map[name][static_cast<size_t>(i)];
        } else {
            Unreachable();
        }
    }

    template <Concepts::HanaString _N>
    constexpr const auto& Get(_N name, Concepts::HanaIntegralConstant auto i) const {
        return Get(name, i.value);
    }

    constexpr const auto& Get(Concepts::Variable auto var, const std::integral auto i) const {
        return Get(var.Name(), i);
    }

    constexpr const auto& Get(Concepts::Variable auto var,
                              Concepts::HanaIntegralConstant auto i) const {
        return Get(var.Name(), i.value);
    }

    template <fixed_string _NAME>
    constexpr const auto& At(const std::integral auto i) const {
        return Get(string_c<_NAME>, i);
    }

    template <fixed_string _NAME>
    constexpr const auto& At(Concepts::HanaIntegralConstant auto i) const {
        return Get(string_c<_NAME>, i.value);
    }

    constexpr const auto& Get(auto key, auto... args) const
        requires(hana::contains(hana::keys(_Map{}), decltype(key)::Name()).value) {
        return Get(key).Get(args...);
    }

    constexpr const auto& Get(auto key, const std::integral auto i, auto... args) const {
        return Get(key, i).Get(args...);
    }

    constexpr const auto& Get(auto key, Concepts::HanaIntegralConstant auto i, auto... args) const {
        return Get(key, i.value).Get(args...);
    }

    /**
     * @todo Improve or add requirements.
     */
    constexpr decltype(auto) operator()(Concepts::Variable auto var, auto... args) const {
        if constexpr (sizeof...(args)) {
            return BypassImpl(var.Name()).value().get()(args...);
        } else {
            return BypassImpl(var.Name()).value().get();
        }
    }

    constexpr decltype(auto) operator()(Concepts::Variable auto var,
                                        const std::integral auto i,
                                        auto... args) const {
        if constexpr (sizeof...(args)) {
            return BypassImpl(var.Name(), i).value().get()(args...);
        } else {
            return BypassImpl(var.Name(), i).value().get();
        }
    }

    constexpr decltype(auto) operator()(Concepts::Variable auto var,
                                        Concepts::HanaIntegralConstant auto i,
                                        auto... args) const {
        if constexpr (sizeof...(args)) {
            return BypassImpl(var.Name(), i.value).value().get()(args...);
        } else {
            return BypassImpl(var.Name(), i.value).value().get();
        }
    }

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
     * @todo Improve or add requirements.
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

    constexpr auto operator==(Concepts::Variable auto var) const {
        return boost::hana::bool_c<std::same_as<Variable, decltype(var)>>;
    }

    constexpr auto operator!=(Concepts::Variable auto var) const {
        return boost::hana::bool_c<!std::same_as<Variable, decltype(var)>>;
    }

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
    template <Concepts::HanaString _N, Concepts::HanaMap _M, Concepts::HanaOptional _S>
    friend class Variable;
    template <typename _Scalar, Concepts::Variable _Variable>
    friend class VariableMap;

    static constexpr auto DefaultMap() {
        return Variable{}._map;
    }

    template <Concepts::HanaString _N, Concepts::HanaMap _M, Concepts::HanaOptional _S>
    static constexpr auto Make(_N name, _M map, const index_t index, _S size) {
        return Variable<_N, _M, _S>{name, map, index, size};
    }

    constexpr Variable(_Name /* name */, _Map map, const index_t index, _Size /* size */)
        : _map{map}, _index{index} {
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
                return _Size{}.value();
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
    _Size _size;

    index_t _index;
};

template <Concepts::HanaString _Name>
Variable(_Name name) -> Variable<_Name, decltype(hana::make_map()), decltype(hana::nothing)>;

template <Concepts::HanaString _Name, Concepts::HanaIntegralConstant _Size>
Variable(_Name name, _Size size)
    -> Variable<_Name, decltype(hana::make_map()), decltype(hana::just(size))>;

static_assert(Q != std::numeric_limits<index_t>::max() &&
                  Q_c.value != static_cast<long long>(std::numeric_limits<index_t>::max()),
              "The special value denoting the size of a unit quaternion must be different from the "
              "maximum value that index_t integers can take.");
template <fixed_string _NAME, index_t _SIZE = std::numeric_limits<index_t>::max()>
inline constexpr auto var_c = []() {
    if constexpr (_SIZE == std::numeric_limits<index_t>::max()) {
        return Variable{string_c<_NAME>};
    } else {
        return Variable{string_c<_NAME>, hana::llong_c<static_cast<long long>(_SIZE)>};
    }
}();

template <Concepts::HanaString _N, Concepts::HanaMap _M, Concepts::HanaOptional _S>
constexpr auto operator*(Concepts::HanaIntegralConstant auto cnt, Variable<_N, _M, _S> var) {
    return VariableProductExpr{cnt, var};
}

constexpr auto operator,(Concepts::Variable auto var1, Concepts::Variable auto var2) {
    return hana::make_tuple(var1, var2);
}

constexpr auto operator,(Concepts::HanaTuple auto tuple, Concepts::Variable auto var) {
    return hana::append(tuple, var);
}

constexpr auto operator,(Concepts::Variable auto var, Concepts::HanaTuple auto tuple) {
    return hana::prepend(tuple, var);
}

constexpr auto operator,(Concepts::VariableProductExpr auto prodExpr, Concepts::Variable auto var) {
    return hana::make_tuple(prodExpr, var);
}

constexpr auto operator,(Concepts::Variable auto var, Concepts::VariableProductExpr auto prodExpr) {
    return hana::make_tuple(var, prodExpr);
}

constexpr auto operator,(Concepts::VariableProductExpr auto prodExpr1,
                         Concepts::VariableProductExpr auto prodExpr2) {
    return hana::make_tuple(prodExpr1, prodExpr2);
}

constexpr auto operator,(Concepts::HanaTuple auto tuple,
                         Concepts::VariableProductExpr auto prodExpr) {
    return hana::append(tuple, prodExpr);
}

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
