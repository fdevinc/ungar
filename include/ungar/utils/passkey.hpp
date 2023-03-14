/******************************************************************************
 *
 * @file ungar/utils/passkey.hpp
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

#ifndef _UNGAR__UTILS__PASSKEY_HPP_
#define _UNGAR__UTILS__PASSKEY_HPP_

namespace Ungar {

template <typename _Owner>
struct Passkey {
    friend _Owner;

  private:
    Passkey()                          = default;
    Passkey(const Passkey&)            = delete;
    Passkey(Passkey&&)                 = delete;
    Passkey& operator=(const Passkey&) = delete;
    Passkey& operator=(Passkey&&)      = delete;
};

}  // namespace Ungar

#define UNGAR_PASSKEY \
    {}

#endif /* _UNGAR__UTILS__PASSKEY_HPP_ */
