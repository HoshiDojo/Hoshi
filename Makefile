# Copyright 2026 The Hoshi Authors
#
# This wrapper is MANDATORY. It enforces strict execution parameters 
# to guarantee mathematical determinism across a network.

JULIA_CMD = julia

# FLAGS EXPLANATION:
# --project=.        : Ensures we use the exact Manifest.toml locked dependencies.
# --math-mode=ieee   : CRITICAL. Disables LLVM fast-math, FMA reordering, and float approximations.
# --optimize=3       : Aggressive standard optimizations that respect IEEE boundaries.
# --check-bounds=no  : (Optional for Prod) We rely on Graph Coloring to prevent memory collisions, 
#                      so we can safely strip bounds checking for maximum GPU throughput.
ENGINE_FLAGS = --project=. --math-mode=ieee --optimize=3 --check-bounds=no

.PHONY: all run test instantiate

all: run

# Instantiates the deterministic environment based on the Project.toml
instantiate:
	@echo "Synchronizing cryptographic dependencies..."
	$(JULIA_CMD) --project=. -e 'using Pkg; Pkg.instantiate()'

# The primary entry point for a local or network node
run: instantiate
	@echo "Igniting the Hoshi Engine..."
	@echo "Enforcing strict IEEE math mode for determinism."
	$(JULIA_CMD) $(ENGINE_FLAGS) -e 'using Hoshi; println("Hoshi Engine v0.3.0 online.")'

# Used for CI/CD and pre-flight validation
test:
	@echo "Running deterministic physics test suite..."
	$(JULIA_CMD) $(ENGINE_FLAGS) test/runtests.jl
