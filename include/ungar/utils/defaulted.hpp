/******************************************************************************
 *
 * @file ungar/utils/defaulted.hpp
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

#ifndef _UNGAR__UTILS__DEFAULTED_HPP_
#define _UNGAR__UTILS__DEFAULTED_HPP_

#include <optional>

namespace Ungar {
namespace Internal {

template <typename _T>
struct default_helper {
    using value_type = _T;

    constexpr default_helper(_T v) : value{std::move(v)} {
    }

    _T value;
};

}  // namespace Internal

template <Internal::default_helper _DEFAULT>
class defaulted : private std::optional<typename decltype(_DEFAULT)::value_type> {
  public:
    using value_type = typename decltype(_DEFAULT)::value_type;
    using std::optional<value_type>::optional;

    constexpr value_type value() const& {
        return value_or(_DEFAULT.value);
    }

    constexpr value_type value() && {
        return value_or(_DEFAULT.value);
    }

    constexpr operator value_type() const noexcept {
        return value();
    }

  private:
    using std::optional<value_type>::value_or;
};

inline constexpr auto default_value = std::nullopt;

}  // namespace Ungar

#endif /* _UNGAR__UTILS__DEFAULTED_HPP_ */
