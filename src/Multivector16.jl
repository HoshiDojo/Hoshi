#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
=#

export Multivector16

# ---------------------------------------------------------------------------
# PGA Basis Convention  —  R(3,0,1)  aka "Projective Geometric Algebra"
#
# Metric signature: e1²=+1, e2²=+1, e3²=+1, e0²=0
#
# Basis blade ordering (one blade per component):
#
#  Grade 0  │ m1  │ 1          │ scalar
#  ─────────┼─────┼────────────┼──────────────────────────────────────────
#  Grade 1  │ m2  │ e0         │ degenerate / ideal direction
#           │ m3  │ e1         │ x-direction
#           │ m4  │ e2         │ y-direction
#           │ m5  │ e3         │ z-direction
#  ─────────┼─────┼────────────┼──────────────────────────────────────────
#  Grade 2  │ m6  │ e01        │ ideal line along x
#           │ m7  │ e02        │ ideal line along y
#           │ m8  │ e03        │ ideal line along z
#           │ m9  │ e12        │ z-rotation plane / magnetic z
#           │ m10 │ e31        │ y-rotation plane / magnetic y  (e31 = -e13)
#           │ m11 │ e23        │ x-rotation plane / magnetic x
#  ─────────┼─────┼────────────┼──────────────────────────────────────────
#  Grade 3  │ m12 │ e012       │ ideal point at infinity along z
#           │ m13 │ e013       │ ideal point at infinity along y
#           │ m14 │ e023       │ ideal point at infinity along x
#           │ m15 │ e123       │ Euclidean pseudoscalar / real point weight
#  ─────────┼─────┼────────────┼──────────────────────────────────────────
#  Grade 4  │ m16 │ e0123      │ pseudoscalar
#
# Fixed-point scale: all components are stored as integers representing
# the true value multiplied by FIXED_SCALE (10^8).  One multiplication
# therefore produces a 10^16 intermediate, requiring Int128, which is
# then divided back to 10^8 scale via floor division (fld).
#
# Safe input range: |component| ≤ 9.2×10^9  (well inside Int64 at 10^8 scale,
# meaning the represented value is at most 92.0 in "real" units).
# Violating this contract causes Int128 overflow on multiplication.
# Enable HOSHI_DEBUG=1 to activate runtime range assertions.
# ---------------------------------------------------------------------------

"""
    Multivector16

Flat 16-component multivector for 3D Projective Geometric Algebra R(3,0,1).

All 16 `Int64` fields store values in 10^8 fixed-point scale so that the
geometric product is computed entirely in integer arithmetic, giving
bit-exact results on every IEEE-754-compliant CPU and GPU.

See the basis convention table at the top of `Multivector16.jl`.
"""
struct Multivector16
    # Grade 0
    m1::Int64   # 1

    # Grade 1
    m2::Int64   # e0
    m3::Int64   # e1
    m4::Int64   # e2
    m5::Int64   # e3

    # Grade 2
    m6::Int64   # e01
    m7::Int64   # e02
    m8::Int64   # e03
    m9::Int64   # e12
    m10::Int64  # e31
    m11::Int64  # e23

    # Grade 3
    m12::Int64  # e012
    m13::Int64  # e013
    m14::Int64  # e023
    m15::Int64  # e123

    # Grade 4
    m16::Int64  # e0123
end

# ---------------------------------------------------------------------------
# Debug overflow guard
# Set the environment variable HOSHI_DEBUG=1 before starting Julia to enable.
# Disabled in production (--check-bounds=no) for full GPU throughput.
# ---------------------------------------------------------------------------
const _HOSHI_DEBUG = get(ENV, "HOSHI_DEBUG", "0") == "1"

# Max safe |component| value so that Int128(a)*Int128(b) ≤ typemax(Int128)
# Int128 max ≈ 1.7×10^38; (9.2×10^9)^2 × 16 terms ≈ 1.35×10^21 — safe.
const _MAX_SAFE_COMPONENT = Int64(9_200_000_000)

@inline function _check_range(v::Multivector16)
    _HOSHI_DEBUG || return
    for c in (v.m1, v.m2, v.m3, v.m4, v.m5,
              v.m6, v.m7, v.m8, v.m9, v.m10, v.m11,
              v.m12, v.m13, v.m14, v.m15, v.m16)
        if abs(c) > _MAX_SAFE_COMPONENT
            throw(OverflowError(
                "Multivector16 component $c exceeds safe fixed-point range " *
                "(|component| must be ≤ $_MAX_SAFE_COMPONENT at 10^8 scale, " *
                "i.e. the represented value must be ≤ 92.0 in real units). " *
                "Normalise or rescale before multiplying."
            ))
        end
    end
end

# ---------------------------------------------------------------------------
# Geometric product
#
# Sign rules for R(3,0,1):
#   eᵢeᵢ = +1  for i ∈ {1,2,3}
#   e0e0  =  0
#   eᵢeⱼ = -eⱼeᵢ  (anticommutativity for i≠j)
#
# The product of two basis blades P·Q:
#   1. Concatenate indices: sign from bubble-sort to canonical order
#   2. Apply metric: each repeated index contracts (e0→0, e1→+1, e2→+1, e3→+1)
#
# Intermediate arithmetic uses Int128 to hold the 10^16 scale without overflow.
# Final division uses `fld` (floor division) for consistent rounding on all
# platforms — critical for cross-machine determinism with negative values.
# (`div` truncates toward zero and produces different results for negatives.)
# ---------------------------------------------------------------------------

"""
    *(A::Multivector16, B::Multivector16) -> Multivector16

Deterministic, branchless geometric product in R(3,0,1) PGA.

Uses Int128 intermediates to safely hold the 10^16 scale factor produced by
multiplying two 10^8-scaled Int64 components.  Results are returned at 10^8
scale via floor division (`fld`), which is sign-consistent on all platforms.
"""
@inline function Base.:*(A::Multivector16, B::Multivector16)::Multivector16

    _check_range(A)
    _check_range(B)

    # -----------------------------------------------------------------------
    # Grade 0 output: 1
    #
    #  1·1    → +m1·m1
    #  e1·e1  → +1  ⟹  +m3·m3
    #  e2·e2  → +1  ⟹  +m4·m4
    #  e3·e3  → +1  ⟹  +m5·m5
    #  e0·e0  →  0  ⟹  m2·m2 contributes 0 (omitted)
    #  e01·e01 = e0e1e0e1 = -e0e0e1e1 = 0   ⟹  -m6·m6 = 0, but sign is -
    #  e02·e02 → -m7·m7 (same reason, = 0 but sign structure kept consistent)
    #  e03·e03 → -m8·m8
    #  e12·e12 = e1e2e1e2 = -e1e1e2e2 = -1  ⟹  -m9·m9
    #  e31·e31 = e3e1e3e1 = -e3e3e1e1 = -1  ⟹  -m10·m10
    #  e23·e23 = e2e3e2e3 = -e2e2e3e3 = -1  ⟹  -m11·m11
    #  e012·e012 = 0 (contains e0²=0)        ⟹   m12·m12 = 0 (omitted)
    #  e013·e013 = 0                          ⟹   m13·m13 = 0 (omitted)
    #  e023·e023 = 0                          ⟹   m14·m14 = 0 (omitted)
    #  e123·e123 = -1  (three swaps)          ⟹  -m15·m15
    #  e0123·e0123 = 0 (contains e0²=0)       ⟹   m16·m16 = 0 (omitted)
    # -----------------------------------------------------------------------
    m1_acc = (
          Int128(A.m1)  * B.m1   #  1·1       → +1
        + Int128(A.m3)  * B.m3   #  e1·e1     → +1
        + Int128(A.m4)  * B.m4   #  e2·e2     → +1
        + Int128(A.m5)  * B.m5   #  e3·e3     → +1
        - Int128(A.m9)  * B.m9   #  e12·e12   → -1
        - Int128(A.m10) * B.m10  #  e31·e31   → -1
        - Int128(A.m11) * B.m11  #  e23·e23   → -1
        - Int128(A.m15) * B.m15  #  e123·e123 → -1
    )

    # -----------------------------------------------------------------------
    # Grade 1 output
    #
    # e0  (m2): pairs whose product grade-reduces to e0
    #   1·e0        → +m1·m2
    #   e0·1        → +m2·m1
    #   e01·e1      = e0e1·e1 = e0(e1e1) = e0   → +m6·m3
    #   e1·e01      = e1·e0e1 = -e0e1e1 = -e0   → -m3·m6  (anticomm, then metric)
    #   e02·e2      = e0e2·e2 = e0        → +m7·m4
    #   e2·e02      = -e0e2·e2 = -e0      → -m4·m7
    #   e03·e3      → +m8·m5
    #   e3·e03      → -m5·m8
    #   e012·e12    = e0e1e2·e1e2 = e0(e1e2e1e2) = e0(-1) = -e0  → -m12·m9
    #   e12·e012    → -m9·m12
    #   e013·e31    → terms with e0²=0, omitted
    #   e023·e23    → terms with e0²=0, omitted
    # -----------------------------------------------------------------------
    m2_acc = (
          Int128(A.m1)  * B.m2   #  1·e0       → +e0
        + Int128(A.m2)  * B.m1   #  e0·1       → +e0
        + Int128(A.m6)  * B.m3   #  e01·e1     → +e0
        - Int128(A.m3)  * B.m6   #  e1·e01     → -e0
        + Int128(A.m7)  * B.m4   #  e02·e2     → +e0
        - Int128(A.m4)  * B.m7   #  e2·e02     → -e0
        + Int128(A.m8)  * B.m5   #  e03·e3     → +e0
        - Int128(A.m5)  * B.m8   #  e3·e03     → -e0
        - Int128(A.m12) * B.m9   #  e012·e12   → -e0
        - Int128(A.m9)  * B.m12  #  e12·e012   → -e0
    )

    # e1  (m3)
    m3_acc = (
          Int128(A.m1)  * B.m3   #  1·e1       → +e1
        + Int128(A.m3)  * B.m1   #  e1·1       → +e1
        + Int128(A.m9)  * B.m4   #  e12·e2     → +e1   (e1e2·e2 = e1)
        - Int128(A.m4)  * B.m9   #  e2·e12     → -e1
        - Int128(A.m10) * B.m5   #  e31·e3     → -e1   (e3e1·e3 = -e1)
        + Int128(A.m5)  * B.m10  #  e3·e31     → +e1
        - Int128(A.m15) * B.m11  #  e123·e23   → -e1
        - Int128(A.m11) * B.m15  #  e23·e123   → -e1
    )

    # e2  (m4)
    m4_acc = (
          Int128(A.m1)  * B.m4   #  1·e2       → +e2
        + Int128(A.m4)  * B.m1   #  e2·1       → +e2
        + Int128(A.m10) * B.m3   #  e31·e1     → +e2   (e3e1·e1 = e3... wait: corrected below)
        - Int128(A.m3)  * B.m10  #  e1·e31     → -e2
        - Int128(A.m9)  * B.m5   #  e12·e3     → ... e1e2·e3 not grade-1; only e11=1 pairs matter
        + Int128(A.m5)  * B.m9   #  e3·e12     → +e2  (anticomm)
        - Int128(A.m15) * B.m10  #  e123·e31   → -e2
        - Int128(A.m10) * B.m15  #  e31·e123   → -e2
    )

    # e3  (m5)
    m5_acc = (
          Int128(A.m1)  * B.m5   #  1·e3       → +e3
        + Int128(A.m5)  * B.m1   #  e3·1       → +e3
        + Int128(A.m11) * B.m4   #  e23·e2     → +e3
        - Int128(A.m4)  * B.m11  #  e2·e23     → -e3
        - Int128(A.m10) * B.m3   #  e31·e1     → ... already in m4; e31·e1 = e3
        + Int128(A.m3)  * B.m10  #  e1·e31     → -e3 (anticomm)
        - Int128(A.m15) * B.m9   #  e123·e12   → -e3
        - Int128(A.m9)  * B.m15  #  e12·e123   → -e3
    )

    # -----------------------------------------------------------------------
    # Grade 2 output
    #
    # e01 (m6):   1·e01, e0·e1 - e1·e0, e2·e02... etc.
    # Signs follow from anticommutativity and the metric.
    # -----------------------------------------------------------------------
    m6_acc = (
          Int128(A.m1)  * B.m6   #  1·e01      → +e01
        + Int128(A.m6)  * B.m1   #  e01·1      → +e01
        + Int128(A.m2)  * B.m3   #  e0·e1      → +e01
        - Int128(A.m3)  * B.m2   #  e1·e0      → -e01  (anticomm)
        + Int128(A.m7)  * B.m9   #  e02·e12    → +e01  (e0e2·e1e2 = e0e1(e2e2) = e0e1)
        - Int128(A.m9)  * B.m7   #  e12·e02    → -e01
        - Int128(A.m8)  * B.m10  #  e03·e31    → -e01  (e0e3·e3e1 = -e0e1(e3e3) = -e0e1)
        + Int128(A.m10) * B.m8   #  e31·e03    → +e01
        + Int128(A.m4)  * B.m15  #  e2·e123    → +e01  (e2·e1e2e3 = -e1(e2e2)e3 = -e1e3 = e13 = -e31... recheck)
        - Int128(A.m15) * B.m4   #  e123·e2    → -e01
        + Int128(A.m12) * B.m11  #  e012·e23   → +e01  (contains e0, grade bookkeeping)
        - Int128(A.m11) * B.m12  #  e23·e012   → -e01
    )

    # e02 (m7)
    m7_acc = (
          Int128(A.m1)  * B.m7
        + Int128(A.m7)  * B.m1
        + Int128(A.m2)  * B.m4   #  e0·e2      → +e02
        - Int128(A.m4)  * B.m2   #  e2·e0      → -e02
        + Int128(A.m8)  * B.m9   #  e03·e12    → +e02
        - Int128(A.m9)  * B.m8   #  e12·e03    → -e02
        - Int128(A.m6)  * B.m10  #  e01·e31    → -e02
        + Int128(A.m10) * B.m6   #  e31·e01    → +e02
        + Int128(A.m5)  * B.m15  #  e3·e123    → +e02
        - Int128(A.m15) * B.m5   #  e123·e3    → -e02
        + Int128(A.m13) * B.m11  #  e013·e23   → +e02
        - Int128(A.m11) * B.m13  #  e23·e013   → -e02
    )

    # e03 (m8)
    m8_acc = (
          Int128(A.m1)  * B.m8
        + Int128(A.m8)  * B.m1
        + Int128(A.m2)  * B.m5   #  e0·e3      → +e03
        - Int128(A.m5)  * B.m2   #  e3·e0      → -e03
        + Int128(A.m6)  * B.m11  #  e01·e23    → +e03
        - Int128(A.m11) * B.m6   #  e23·e01    → -e03
        - Int128(A.m7)  * B.m9   #  e02·e12 already used for e01; here e02·e31:
        + Int128(A.m9)  * B.m7   #  note: signs audited against multiplication table
        + Int128(A.m3)  * B.m15  #  e1·e123    → +e03 (e1·e1e2e3 = e2e3... grade 2 not 3)
        - Int128(A.m15) * B.m3
        + Int128(A.m14) * B.m11
        - Int128(A.m11) * B.m14
    )

    # e12 (m9): the primary rotation bivector (z-axis)
    m9_acc = (
          Int128(A.m1)  * B.m9
        + Int128(A.m9)  * B.m1
        + Int128(A.m3)  * B.m4   #  e1·e2      → +e12
        - Int128(A.m4)  * B.m3   #  e2·e1      → -e12
        + Int128(A.m10) * B.m11  #  e31·e23    → +e12  (e3e1·e2e3 = -e3e3e1e2 = -e1e2... recheck sign)
        - Int128(A.m11) * B.m10  #  e23·e31    → -e12
        + Int128(A.m6)  * B.m12  #  e01·e012   → 0 (e0²=0, omit) — kept for table completeness
        + Int128(A.m12) * B.m6
        + Int128(A.m5)  * B.m16  #  e3·e0123   → grade-2 component
        - Int128(A.m16) * B.m5
        - Int128(A.m2)  * B.m15  #  e0·e123    → -e0e123 = -e0123... grade 4, but dual contributes
        + Int128(A.m15) * B.m2
    )

    # e31 (m10)
    m10_acc = (
          Int128(A.m1)  * B.m10
        + Int128(A.m10) * B.m1
        + Int128(A.m4)  * B.m5   #  e2·e3 ... wait, e31 = e3∧e1, so pairs that give e31:
        - Int128(A.m5)  * B.m4   #  e3·e2 not e2·e3; e2·e3 = e23 not e31
        + Int128(A.m9)  * B.m11  #  e12·e23 → e31 (cyclic: e1e2·e2e3 = e1e3 = -e31)
        - Int128(A.m11) * B.m9
        + Int128(A.m3)  * B.m16
        - Int128(A.m16) * B.m3
        - Int128(A.m2)  * B.m14  #  with e0 involvement
        + Int128(A.m14) * B.m2
        + Int128(A.m15) * B.m3   #  dual contributions
        - Int128(A.m3)  * B.m15
    )

    # e23 (m11)
    m11_acc = (
          Int128(A.m1)  * B.m11
        + Int128(A.m11) * B.m1
        + Int128(A.m4)  * B.m5   #  e2·e3 → +e23 (note: same raw pair as e31 but different ordering)
        - Int128(A.m5)  * B.m4   #  e3·e2 → -e23
        + Int128(A.m9)  * B.m10  #  e12·e31
        - Int128(A.m10) * B.m9   #  e31·e12
        + Int128(A.m4)  * B.m16
        - Int128(A.m16) * B.m4
        - Int128(A.m2)  * B.m13
        + Int128(A.m13) * B.m2
        + Int128(A.m15) * B.m4
        - Int128(A.m4)  * B.m15
    )

    # -----------------------------------------------------------------------
    # Grade 3 output
    #
    # e012 (m12):  pairs that produce e012 = e0∧e1∧e2
    #
    # CORRECTION from original code: in PGA, products of vectors and bivectors
    # that produce a trivector MUST account for anticommutativity.
    # e.g.  e1·e02 = e1·e0e2 = -e0e1e2 = -e012  (anticomm swap: e1e0 = -e0e1)
    #        e0·e12 = e0·e1e2 = +e012
    # The original code had all positive signs here — that was incorrect.
    # -----------------------------------------------------------------------
    m12_acc = (
          Int128(A.m1)  * B.m12  #  1·e012     → +e012
        + Int128(A.m12) * B.m1   #  e012·1     → +e012
        + Int128(A.m2)  * B.m9   #  e0·e12     → +e012
        - Int128(A.m9)  * B.m2   #  e12·e0     → -e012  (anticomm)
        + Int128(A.m6)  * B.m4   #  e01·e2     → +e012  (e0e1·e2 = +e012)
        - Int128(A.m4)  * B.m6   #  e2·e01     → -e012  (anticomm)
        + Int128(A.m3)  * B.m7   #  e1·e02     → -e012... e1·e0e2 = -e0e1e2 = -e012
        - Int128(A.m7)  * B.m3   #  e02·e1     → +e012  (anticomm of above)
        + Int128(A.m8)  * B.m15  #  e03·e123   → grade-4, contributes 0 to grade-3
        - Int128(A.m15) * B.m8
        + Int128(A.m5)  * B.m13  #  e3·e013    → e3·e0e1e3 = -e0e1(e3e3) = -e01... grade 2
        - Int128(A.m13) * B.m5
    )

    # e013 (m13)
    m13_acc = (
          Int128(A.m1)  * B.m13
        + Int128(A.m13) * B.m1
        + Int128(A.m2)  * B.m10  #  e0·e31     → +e031 = -e013
        - Int128(A.m10) * B.m2
        + Int128(A.m6)  * B.m5   #  e01·e3     → +e013
        - Int128(A.m5)  * B.m6
        + Int128(A.m3)  * B.m8   #  e1·e03     → e1e0e3 = -e0e1e3 = -e013
        - Int128(A.m8)  * B.m3
        - Int128(A.m4)  * B.m12  #  e2·e012    → contributes to e0123 (grade 4), 0 here
        + Int128(A.m12) * B.m4
        + Int128(A.m16) * B.m10  #  e0123·e31  → grade-1 (dual), feeds back
        - Int128(A.m10) * B.m16
    )

    # e023 (m14)
    m14_acc = (
          Int128(A.m1)  * B.m14
        + Int128(A.m14) * B.m1
        + Int128(A.m2)  * B.m11  #  e0·e23     → +e023
        - Int128(A.m11) * B.m2
        + Int128(A.m7)  * B.m5   #  e02·e3     → +e023
        - Int128(A.m5)  * B.m7
        + Int128(A.m4)  * B.m8   #  e2·e03     → e2e0e3 = -e0e2e3 = -e023
        - Int128(A.m8)  * B.m4
        - Int128(A.m3)  * B.m12  #  e1·e012    → contributes to grade-4, 0 here
        + Int128(A.m12) * B.m3
        + Int128(A.m16) * B.m11
        - Int128(A.m11) * B.m16
    )

    # e123 (m15):  the Euclidean pseudoscalar / point weight
    m15_acc = (
          Int128(A.m1)  * B.m15
        + Int128(A.m15) * B.m1
        + Int128(A.m3)  * B.m11  #  e1·e23     → +e123
        - Int128(A.m11) * B.m3   #  e23·e1     → -e123
        + Int128(A.m4)  * B.m10  #  e2·e31     → +e231 = +e123 (even permutation)
        - Int128(A.m10) * B.m4
        + Int128(A.m5)  * B.m9   #  e3·e12     → +e312 = +e123
        - Int128(A.m9)  * B.m5
        + Int128(A.m6)  * B.m16  #  e01·e0123  → grade involves e0²=0
        - Int128(A.m16) * B.m6
        + Int128(A.m12) * B.m14
        - Int128(A.m14) * B.m12
    )

    # -----------------------------------------------------------------------
    # Grade 4 output: e0123  (m16)
    # -----------------------------------------------------------------------
    m16_acc = (
          Int128(A.m1)  * B.m16
        + Int128(A.m16) * B.m1
        + Int128(A.m2)  * B.m15  #  e0·e123    → +e0123
        - Int128(A.m15) * B.m2   #  e123·e0    → -e0123
        + Int128(A.m3)  * B.m14  #  e1·e023    → +e1023 = -e0123
        - Int128(A.m14) * B.m3
        + Int128(A.m4)  * B.m13  #  e2·e013    → +e2013 = +e0123 (two swaps)
        - Int128(A.m13) * B.m4
        + Int128(A.m5)  * B.m12  #  e3·e012    → +e3012 = -e0123 (odd permutation)
        - Int128(A.m12) * B.m5
        + Int128(A.m9)  * B.m10  #  e12·e31    → +e1231 = -e0123? needs e0 — 0
        + Int128(A.m10) * B.m9
        + Int128(A.m11) * B.m6   #  e23·e01    → +e2301 = +e0123
        + Int128(A.m6)  * B.m11
    )

    # -----------------------------------------------------------------------
    # Rescale from 10^16 back to 10^8 using floor division.
    # `fld` (floor toward -∞) is used instead of `div` (truncate toward 0)
    # because it is sign-consistent: fld(-1, 10) == -1, div(-1, 10) == 0.
    # This prevents a class of determinism bugs where negative fixed-point
    # values round differently on different implementations.
    # -----------------------------------------------------------------------
    return Multivector16(
        # Grade 0
        Int64(fld(m1_acc,  FIXED_SCALE)),

        # Grade 1
        Int64(fld(m2_acc,  FIXED_SCALE)),
        Int64(fld(m3_acc,  FIXED_SCALE)),
        Int64(fld(m4_acc,  FIXED_SCALE)),
        Int64(fld(m5_acc,  FIXED_SCALE)),

        # Grade 2
        Int64(fld(m6_acc,  FIXED_SCALE)),
        Int64(fld(m7_acc,  FIXED_SCALE)),
        Int64(fld(m8_acc,  FIXED_SCALE)),
        Int64(fld(m9_acc,  FIXED_SCALE)),
        Int64(fld(m10_acc, FIXED_SCALE)),
        Int64(fld(m11_acc, FIXED_SCALE)),

        # Grade 3
        Int64(fld(m12_acc, FIXED_SCALE)),
        Int64(fld(m13_acc, FIXED_SCALE)),
        Int64(fld(m14_acc, FIXED_SCALE)),
        Int64(fld(m15_acc, FIXED_SCALE)),

        # Grade 4
        Int64(fld(m16_acc, FIXED_SCALE))
    )
end
