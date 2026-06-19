#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
using KernelAbstractions
using StaticArrays

export add_carry, mul_add_carry, add_mod256, mont_mul, mont_inverse,
       to_montgomery, from_montgomery, mitsuse!

# ============================================================================
# Field constants (BLS12-381 scalar field Fr)
# ============================================================================

const R_LIMBS = (0xffffffff00000001, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48)
const INV_R_N_64 = 0xfffffffeffffffff
const R_MOD_R = (0x00000001fffffffe, 0x5884b7fa00034802, 0x998c4fefecbc4ff5, 0x1824b159acc5056f)
const R2_MOD_R = (0xc999e990f3f29c6d, 0x2b6cedcb87925c23, 0x05d314967254398f, 0x0748d9d99f59ff11)
const R_MINUS_2 = (0xfffffffeffffffff, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48)

const ZERO_LIMBS = (0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000)
const ONE_LIMBS  = (0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000)

const COEFF_A     = 0x5
const COEFF_B     = 0x2
const COEFF_DELTA = 0x1

const COEFF_A_MONT     = (0x0000000afffffff5, 0x66d9f3df00120c0b, 0xcc83b7a7960bb7c5, 0x04c9cf6d363b9de5)
const COEFF_B_MONT     = (0x00000003fffffffc, 0xb1096ff400069004, 0x33189fdfd9789fea, 0x304962b3598a0adf)
const COEFF_DELTA_MONT = R_MOD_R

# ============================================================================
# Branchless 64-bit primitives
# ============================================================================

@inline function add_carry(a::UInt64, b::UInt64, carry::UInt64)
    res = UInt128(a) + UInt128(b) + UInt128(carry)
    return UInt64(res & typemax(UInt64)), UInt64(res >> 64)
end

@inline function mul_add_carry(a::UInt64, b::UInt64, c::UInt64, carry::UInt64)
    wide = UInt128(a) * UInt128(b) + UInt128(c) + UInt128(carry)
    return UInt64(wide & typemax(UInt64)), UInt64(wide >> 64)
end

@inline function sub_borrow(a::UInt64, b::UInt64, borrow::UInt64)
    diff1 = a - b
    borrow1 = ifelse(a < b, UInt64(1), UInt64(0))
    diff2 = diff1 - borrow
    borrow2 = ifelse(diff1 < borrow, UInt64(1), UInt64(0))
    return diff2, (borrow1 | borrow2)
end

@inline function sub256(a::NTuple{4, UInt64}, b::NTuple{4, UInt64})
    d1, br = sub_borrow(a[1], b[1], UInt64(0))
    d2, br = sub_borrow(a[2], b[2], br)
    d3, br = sub_borrow(a[3], b[3], br)
    d4, br = sub_borrow(a[4], b[4], br)
    return (d1, d2, d3, d4), br
end

@inline function cond_sub_mod(x::NTuple{4, UInt64})
    diff, borrow = sub256(x, R_LIMBS)
    return ifelse(borrow == 0x1, x, diff)
end

@inline function add_mod256(a::NTuple{4, UInt64}, b::NTuple{4, UInt64})
    s1, c = add_carry(a[1], b[1], UInt64(0))
    s2, c = add_carry(a[2], b[2], c)
    s3, c = add_carry(a[3], b[3], c)
    s4, c = add_carry(a[4], b[4], c)
    return cond_sub_mod((s1, s2, s3, s4))
end

# ============================================================================
# Montgomery multiplication
# ============================================================================

# We use 6 explicitly shadowed scalar variables (t1..t6) representing the
# 384-bit accumulator. The LLVM compiler will map these natively to registers.
@inline function mont_mul(a::NTuple{4, UInt64}, b::NTuple{4, UInt64})
    n  = R_LIMBS
    np = INV_R_N_64

    t1 = t2 = t3 = t4 = t5 = t6 = UInt64(0)

    for i in 1:4
        ai = a[i]
        
        # Multiply step
        lo, c = mul_add_carry(ai, b[1], t1, UInt64(0)); t1 = lo
        lo, c = mul_add_carry(ai, b[2], t2, c); t2 = lo
        lo, c = mul_add_carry(ai, b[3], t3, c); t3 = lo
        lo, c = mul_add_carry(ai, b[4], t4, c); t4 = lo
        lo, c = add_carry(t5, c, UInt64(0)); t5 = lo
        t6 = c

        # Reduce step
        m = t1 * np
        _, c = mul_add_carry(m, n[1], t1, UInt64(0)) # t1 is consumed here
        lo, c = mul_add_carry(m, n[2], t2, c); t1 = lo
        lo, c = mul_add_carry(m, n[3], t3, c); t2 = lo
        lo, c = mul_add_carry(m, n[4], t4, c); t3 = lo
        lo, c = add_carry(t5, c, UInt64(0)); t4 = lo
        t5 = t6 + c
        t6 = UInt64(0)
    end

    L = (t1, t2, t3, t4)
    diff, borrow = sub256(L, n)
    keep_as_is = (t5 == UInt64(0)) & (borrow == UInt64(1))
    return ifelse(keep_as_is, L, diff)
end

@inline to_montgomery(x::NTuple{4, UInt64}) = mont_mul(x, R2_MOD_R)
@inline from_montgomery(x::NTuple{4, UInt64}) = mont_mul(x, ONE_LIMBS)

@inline function mont_inverse(x_mont::NTuple{4, UInt64})
    res = R_MOD_R   
    base = x_mont
    for limb_idx in 1:4
        limb = R_MINUS_2[limb_idx]
        for bit in 0:63
            if (limb & (UInt64(1) << bit)) != 0
                res = mont_mul(res, base)
            end
            base = mont_mul(base, base)
        end
    end
    return res
end

# ============================================================================
# GICG kernel
# ============================================================================

@kernel function gicg_transition_kernel!(states)
    idx = @index(Global, Linear)

    x_curr = states[idx]
    y_shifted = add_mod256(x_curr, COEFF_DELTA_MONT)
    y_inv = mont_inverse(y_shifted)

    # CRITICAL POLE RESOLUTION
    is_pole = iszero(y_shifted[1] | y_shifted[2] | y_shifted[3] | y_shifted[4])
    y_safe_inv = ifelse(is_pole, ZERO_LIMBS, y_inv)

    term_a = mont_mul(COEFF_A_MONT, y_safe_inv)
    x_next = add_mod256(term_a, COEFF_B_MONT)

    states[idx] = x_next
end

function mitsuse!(backend, state_vectors, steps::Int)
    kernel = gicg_transition_kernel!(backend)

    for step in 1:steps
        kernel(state_vectors, ndrange=length(state_vectors))
        KernelAbstractions.synchronize(backend)
    end
end
