/******************************************************************************
 *
 * @file ungar/utils/integral_map.hpp
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

#ifndef _UNGAR__UTILS__INTEGRAL_MAP_HPP_
#define _UNGAR__UTILS__INTEGRAL_MAP_HPP_

#include "ungar/data_types.hpp"

namespace Ungar {

template <typename _T, size_t _MAX_SIZE>
class integral_map {
  public:
    using array_type = std::array<std::optional<_T>, _MAX_SIZE>;
    using value_type = _T;

    constexpr integral_map() : _array{} {
    }

    constexpr integral_map(const integral_map& other) : _array{other._array} {
    }

    constexpr integral_map(integral_map&& other) : _array{std::move(other._array)} {
    }

    constexpr integral_map& operator=(const integral_map& other) {
        _array = other._array;
        return *this;
    }

    constexpr integral_map& operator=(integral_map&& other) {
        _array = std::move(other._array);
        return *this;
    }

    constexpr size_t size() const {
        size_t sz = 0UL;
        for (const auto& el : _array) {
            sz += static_cast<size_t>(el.has_value());
        };
        return sz;
    }

    constexpr const value_type& at(const std::integral auto key) const {
        return _array.at(static_cast<size_t>(key)).value();
    }

    constexpr const value_type& operator[](const std::integral auto key) const {
        return *_array[static_cast<size_t>(key)];
    }

    constexpr value_type& at(const std::integral auto key) {
        return _array.at(static_cast<size_t>(key)).value();
    }

    constexpr value_type& operator[](const std::integral auto key) {
        return *_array[static_cast<size_t>(key)];
    }

    template <typename... _Args>
    constexpr void emplace(const std::integral auto key, _Args&&... args) {
        _array[static_cast<size_t>(key)].emplace(std::forward<_Args>(args)...);
    }

    template <typename... _Args>
    constexpr bool try_emplace(const std::integral auto key, _Args&&... args) {
        if (contains(key)) {
            return false;
        } else {
            _array[static_cast<size_t>(key)].emplace(std::forward<_Args>(args)...);
            return true;
        }
    }

    template <class _Value>
    constexpr void insert_or_assign(const std::integral auto key, _Value&& value) {
        _array[static_cast<size_t>(key)] = std::forward<_Value>(value);
    }

    template <class _Value>
    constexpr bool insert(const std::integral auto key, _Value&& value) {
        if (contains(key)) {
            return false;
        } else {
            _array[static_cast<size_t>(key)].emplace(std::forward<_Value>(value));
            return true;
        }
    }

    constexpr bool contains(const std::integral auto key) const {
        return _array[static_cast<size_t>(key)].has_value();
    }

    constexpr bool empty() const {
        return !size();
    }

  private:
    array_type _array;
};

}  // namespace Ungar

#endif /* _UNGAR__UTILS__INTEGRAL_MAP_HPP_ */
