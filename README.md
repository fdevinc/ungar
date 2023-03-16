# Ungar

> Simplifying optimal control with metaprogramming

## Overview
```cpp
#include "ungar/variable_map.hpp"
#include "ungar/autodiff/function.hpp"
using namespace Ungar;

int main() {
    /***************                     QUADROTOR MODEL                      ***************/
    /***************     Define numeric invariants as integral constants.     ***************/
    constexpr auto N          = 30_c;  // Discrete time horizon.
    constexpr auto NUM_ROTORS = 4_c;

    /***************                 Define "leaf" variables.                 ***************/
    // Positions are 3-dimensional vectors, orientations are unit quaternions,
    // rotor speeds are scalars, etc.
    constexpr auto position         = var_c<"position", 3>;          // := p
    constexpr auto orientation      = var_c<"orientation", Q>;       // := q
    constexpr auto linear_velocity  = var_c<"linear_velocity", 3>;   // := pDot
    constexpr auto angular_velocity = var_c<"angular_velocity", 3>;  // := omega
    constexpr auto rotor_speed      = var_c<"rotor_speed", 1>;       // := r

    /***************                Define "branch" variables.                ***************/
    // States are stacked poses and velocities, while inputs are stacked rotor
    // speeds: one for each rotor.
    constexpr auto x = var_c<"x"> <<=
        (position, orientation, linear_velocity, angular_velocity);  // x := [p q pDot omega]
    constexpr auto u = var_c<"u"> <<= NUM_ROTORS * rotor_speed;      // u := [r0 r1 r2 r3]
    constexpr auto X = var_c<"X"> <<= (N + 1_c) * x;                 // X := [x0 x1 ... xN]
    constexpr auto U = var_c<"U"> <<= N * u;                         // U := [u0 u1 ... uN-1]
    constexpr auto variables = var_c<"variables"> <<= (X, U);

    /***************            Instantiate and use variable maps.            ***************/
    // Access information about variables at compile time.
    static_assert(x.Size() == 13);
    static_assert(variables(x, 0).Index() == 0);         // [{x0} x1  x2  ...  xN  u0  ... ]
    static_assert(variables(x, 1).Index() == 13);        // [ x0 {x1} x2  ...  xN  u0  ... ]
    static_assert(variables(x, 2).Index() == 26);        // [ x0  x1 {x2} ...  xN  u0  ... ]
    static_assert(variables(u, 0).Index() == X.Size());  // [ x0  x1  x2  ...  xN {u0} ... ]

    // Create maps over contiguous arrays of data.
    auto vars = MakeVariableMap<real_t>(variables);  // [ X   u0    u1   ...  uN-1 ]
    vars.Get(u, 0).setZero();                        // [ X  0000   u1   ...  uN-1 ]
    vars.Get(u, 1).setOnes();                        // [ X  0000  1111  ...  uN-1 ]
    vars.Get(u, N - 1).setLinSpaced(2.0, 8.0);       // [ X  0000  1111  ...  2468 ]

    static_assert(std::same_as<decltype(vars.Get(rotor_speed, 0, 0)), real_t&>);
    static_assert(std::same_as<decltype(vars.Get(position, 0)), Eigen::Map<Vector3r>&>);
    static_assert(std::same_as<decltype(vars.Get(orientation, 0)), Eigen::Map<Quaternionr>&>);
    static_assert(std::same_as<decltype(vars.Get(x, 0)), Eigen::Map<Vector<real_t, 13>>&>);
}
```

## Project Organization
The project is organized in the following subdirectories:
- The [`example`](example) directory contains various examples showing the basic functionalities of the library.
- The [`external`](external) directory contains all the third-party libraries. [Eigen][] and
  [Boost.Hana][] are the only required dependencies: if their targets are not found, Ungar will use a bundled version
  of each. All remaining dependencies are optional and bring logging, automatic differentiation, and nonlinear optimization functionalities to the
  library. If enabled, they are downloaded using [CMake][].
- The [`include`](include) directory contains the library itself. The optional modules are located in the [`io`](include/ungar/io),
  [`autodiff`](include/ungar/autodiff), and [`optimization`](include/ungar/optimization) subfolders.
- The [`test`](test) directory contains the source code of all the unit tests.

## Development Status
Ungar is a project meant for long-term maintenance. While the core API of the library is stable, the optional modules will likely see breaking changes
over time. The current implementations will be enhanced to achieve faster compile times. Finally, new algorithms and types will be introduced to
make the creation of optimal control problems as straightforward as possible.

## License
Please see [LICENSE.md](LICENSE.md).

<!-- Links -->
[Eigen]: https://eigen.tuxfamily.org/index.php?title=Main_Page
[Boost.Hana]: https://github.com/boostorg/hana
[CMake]: http://www.cmake.org
