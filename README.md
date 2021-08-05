# MiniCombust
MiniCombust is a mini-app for investigating simulations of combustion in Gas Turbines. MiniCombust uses a Langrangian-Eulerian (particle & FVM flow field solver) approach and incorporates basic models for heat transfer, turbulence, boundary conditions, convection schemes, fuel, soot, spray atomisation and emissions based purely on public-domain work. It is designed for investigating combustion simulations at Exascale, and scales to tens of thousands of cores.

## Design aims:
We'll aim to
* have simple but realistic physics based on public-domain knowledge (no proprietary information)
* provide sample scenarios with known correct answers, with meshes, configurations and run instructions provided. These will help when porting MiniCombust to other languages and architectures.
* provide a canonical implementation in standards-compliant C++17. But MiniCombust’s focus is on the _concepts_ which allow it to scale well rather than language features. Porting to other languages is encouraged.
* focus on achieving good performance at _extreme scale_. While this does not preclude single-node optimisations, optimisations shouldn't detract from the clarity of the implementation. Nothing in the design should make it difficult to use shared-memory intra-node parallelism (e.g. OpenMPI, SYCL) or to target accelerators
* support the ability to model different load-balancing strategies
* make the code should easy to modify, with the focus on _code comprehension_ and _tinkerability_ rather than over-emphasising software engineering concerns such as patterns, DRY, templates etc. The canonical implementation should be written primarily for humans rather than compilers.
* make the code easy to build, with tested support for a wide range of compilers and architectures
* provide living documentation describing the physics and the design choices of the applicatin. We'll adopt a literate programming spirit, and view these docs as equally important as the code, 
* stick to core concerns, and well-tested libraries for auxiliary functions where possible instead of rolling our own. However, we'll also strive to keep the number of dependencies to a minimum, and ensure that the build system manages these well (specifying specific compatible versions etc.). 
* use unit and integration tests that strive for clarity of purpose, forming living documentation and a safety net for regression rather than satisfying arbitrary code coverage.

## Living Documentation
Our documentation lives with the code in this repository, and we use IPython notebooks that describe the application design, physics, references
and runnable snippets that demonstrate concepts.

You can see the docs [here](docs/).