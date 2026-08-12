# Introduction
Hoshi is a hybrid framework for Anomalistics, Metascience, Frontier Science, Epistemic Engineering, and Post-Normal Science. Hoshi is engineered for real-time, multi-scale simulations/games. The following report guides the physics implementation:

[Hidden Assumptions in Fundamental Physics: A Dialectical Audit](FundamentalAssumptions.md)

Rust is used for the Core Layer (for advanced performance) while Julia is used for the Orchestration Layer (for domain logic). The Core Orchestration Bridge (COB) uses Coefficient32, which is a strict cross-vendor version of IEEE 754 floating-point numbers, for hardware-agnostic determinism. COB shaders avoid built-in transcendental functions (sin, cos) for strict bit-exactness across GPU vendors by using precise polynomial approximations (like REMEZ). Matrix isomorphism provides validation logic for the geometric algebra code.