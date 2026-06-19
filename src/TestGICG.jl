#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
using Test

export run_simd_period_test, run_field_arithmetic_test

"""
    run_field_arithmetic_test()

Standalone regression test for the field-arithmetic primitives, independent
of the SIMD/KernelAbstractions plumbing. These expected values were
cross-checked against an independent big-integer implementation, so a
passing result here means the modulus, Montgomery constants, mont_mul, and
mont_inverse are all numerically correct -- not just "internally consistent"
the way a uniqueness-only check could be fooled by a wrong-but-self-consistent
modulus.
"""
function run_field_arithmetic_test()
    # Round-trip through Montgomery form should be the identity.
    for i in (UInt64(1), UInt64(5), UInt64(0x2a), UInt64(0xdeadbeef))
        x = (i, UInt64(0), UInt64(0), UInt64(0))
        @test from_montgomery(to_montgomery(x)) == x
    end

    # inverse(5) mod r, computed independently.
    inv5_expected = (0xcccccccc33333334, 0x323e959b66656a65, 0x51ef819e6c2de803, 0x458e97984c2b4b2b)
    inv5_got = from_montgomery(mont_inverse(to_montgomery((UInt64(5), UInt64(0), UInt64(0), UInt64(0)))))
    @test inv5_got == inv5_expected

    # Pole case: x = r - 1 means x + delta â‰¡ 0 mod r, so the transition rule
    # must fall back to exactly b = 2, not propagate an undefined division.
    r_minus_1 = (0xffffffff00000000, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48)
    pole_buf = [to_montgomery(r_minus_1)]
    mitsuse!(KernelAbstractions.CPU(), pole_buf, 1)
    @test from_montgomery(pole_buf[1]) == (0x2, 0x0, 0x0, 0x0)

    # A normal, non-pole step: x = 3, independently verified expected output.
    step3_expected = (0x3fffffff40000004, 0xfece3b023ffec4ff, 0x266b620607396203, 0x56f23d7e5f361df6)
    buf = [to_montgomery((UInt64(3), UInt64(0), UInt64(0), UInt64(0)))]
    mitsuse!(KernelAbstractions.CPU(), buf, 1)
    @test from_montgomery(buf[1]) == step3_expected

    println("Field arithmetic verified against independent reference values.")
end

function run_simd_period_test(total_states::Int)
    # 1. Target CPU execution backend explicitly to leverage predictable SIMD execution
    backend = KernelAbstractions.CPU()

    # 2. Allocate the state buffer directly managed by the KernelAbstractions backend
    states_buffer = KernelAbstractions.allocate(backend, NTuple{4, UInt64}, total_states)

    # 3. Initialize state paths sequentially on the host CPU.
    #    States live in Montgomery form throughout the kernel pipeline, so
    #    each plain integer must be encoded with to_montgomery before upload.
    host_states = Vector{NTuple{4, UInt64}}(undef, total_states)
    for i in 1:total_states
        host_states[i] = to_montgomery((UInt64(i), UInt64(0), UInt64(0), UInt64(0)))
    end

    # KernelAbstractions.allocate(CPU(), ...) returns a plain Array, so a
    # plain copyto! is all that's needed here -- no backend-aware overload
    # exists (or is needed) for the CPU case.
    copyto!(states_buffer, host_states)

    # 4. Run a single step mutation across all SIMD lanes simultaneously
    mitsuse!(backend, states_buffer, 1)

    # 5. Allocate an empty host array to hold the mutated output results
    results_mont = Vector{NTuple{4, UInt64}}(undef, total_states)

    copyto!(results_mont, states_buffer)

    # 6. Decode back out of Montgomery form before inspecting the values
    results = from_montgomery.(results_mont)

    # 7. Statistical Check: Ensure 100% uniqueness across the processed vector lane
    unique_results = unique(results)
    @test length(unique_results) == total_states
    println("Verification Complete: All SIMD vector lanes evolved without collision traps.")
end
