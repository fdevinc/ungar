# Ungar

> Simplifying optimal control with metaprogramming
## Overview
```cpp
#include "ungar/variable_map.hpp"
using namespace Ungar;

int main() {
    /***************                     QUADROTOR MODEL                      ***************/
    /***************     Define numeric invariants as integral constants.     ***************/
    constexpr auto N = 30_c;  // Discrete time horizon.
    constexpr auto NUM_ROTORS = 4_c;

    /***************                 Define "leaf" variables.                 ***************/
    // Positions are 3-dimensional vectors, orientations are unit quaternions,
    // rotor speeds are scalars, etc.
    UNGAR_VARIABLE(position, 3);          // := p
    UNGAR_VARIABLE(orientation, Q);       // := q
    UNGAR_VARIABLE(linear_velocity, 3);   // := pDot
    UNGAR_VARIABLE(angular_velocity, 3);  // := omega
    UNGAR_VARIABLE(rotor_speed, 1);       // := r

    /***************                Define "branch" variables.                ***************/
    // States are stacked poses and velocities, while inputs are stacked rotor
    // speeds: one for each rotor.
    UNGAR_VARIABLE(x) <<=
        (position, orientation, linear_velocity, angular_velocity); // x := [p q pDot omega]
    UNGAR_VARIABLE(u) <<= NUM_ROTORS * rotor_speed;                 // u := [r0 r1 r2 r3]
    UNGAR_VARIABLE(X) <<= (N + 1_c) * x;                            // X := [x0 x1 ... xN]
    UNGAR_VARIABLE(U) <<= N * u;                                    // U := [u0 u1 ... uN-1]
    UNGAR_VARIABLE(variables) <<= (X, U);

    /***************            Instantiate and use variable maps.            ***************/
    // Access information about variables at compile time.
    static_assert(x.Size() == 13);
    static_assert(variables(x, 0).Index() == 0);         // [{x0} x1  x2  ...  xN  u0  ... ]
    static_assert(variables(x, 1).Index() == 13);        // [ x0 {x1} x2  ...  xN  u0  ... ]
    static_assert(variables(x, 2).Index() == 26);        // [ x0  x1 {x2} ...  xN  u0  ... ]
    static_assert(variables(u, 0).Index() == X.Size());  // [ x0  x1  x2  ...  xN {u0} ... ]

    // Create maps over contiguous arrays of data.
    auto vars = MakeVariableMap<real_t>(variables);          // [ X   u0    u1   ...  uN-1 ]
    vars.Get(u, 0).setZero();                                // [ X  0000   u1   ...  uN-1 ]
    vars.Get(u, 1).setOnes();                                // [ X  0000  1111  ...  uN-1 ]
    vars.Get(u, N - 1).setLinSpaced(2.0, 8.0);               // [ X  0000  1111  ...  2468 ]

    static_assert(std::same_as<decltype(vars.Get(rotor_speed, 0, 0)), real_t&>);
    static_assert(std::same_as<decltype(vars.Get(position, 0)), Eigen::Map<Vector3r>&>);
    static_assert(std::same_as<decltype(vars.Get(orientation, 0)), Eigen::Map<Quaternionr>&>);
    static_assert(std::same_as<decltype(vars.Get(x, 0)), Eigen::Map<Vector<real_t, 13>>&>);
}
```

## Using Ungar
> Coming soon...

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
