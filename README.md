# Hoshi
Hoshi is a hybrid framework for real-time, multi-scale simulations and games. It targets: Anomalistics, Metascience, Frontier Science, Epistemological Engineering, and Post-Normal Science.

Rust is used for the Core Performance Layer while Julia is used for the Orchestration and Logic Layer.

P32E2, which is a 32-bit posit number format configuration with an exponent size (es) of 2 bits, is used instead of IEEE 754 floating-point numbers for hardware-agnostic determinism. The shaders avoid built-in transcendental functions (sin, cos) for strict bit-exactness across GPU vendors by using precise polynomial approximations (like REMEZ). Matrix isomorphism provides validation logic for the geometric algebra code.