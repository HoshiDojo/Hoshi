#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export Multivector16
"""
Multivector16

The foundational geometric entity for the Hoshi engine. Represents a flat 16-component multivector for 3D Projective/Conformal Geometric Algebra. Uses strict 10^8 fixed-point integer scaling to eliminate floating-point drift.
"""
struct Multivector16
    # Grade 0 (Scalar)
    m1::Int64  

    # Grade 1 (Vectors)
    m2::Int64  
    m3::Int64  
    m4::Int64  
    m5::Int64  

    # Grade 2 (Bivectors)
    m6::Int64  
    m7::Int64  
    m8::Int64  
    m9::Int64  
    m10::Int64 
    m11::Int64 

    # Grade 3 (Trivectors)
    m12::Int64 
    m13::Int64 
    m14::Int64 
    m15::Int64 

    # Grade 4 (Pseudoscalar / Quadvector)
    m16::Int64 
end

"""
*(A::Multivector16, B::Multivector16)

Executes the deterministic, branchless geometric product of two 16-component multivectors. Utilizes 128-bit intermediate hardware registers to catch the 10^16 scale factor multiplication, safely downscaling back to Int64 without overflow.
"""
@inline function Base.:*(A::Multivector16, B::Multivector16)::Multivector16
    
    # Grade 0
    # In PGA (R_3,0,1), the degenerate basis e0 squares to 0, while e1, e2, e3 square to 1.
    # Bivectors square to -1, which results in subtraction.
    m1_acc = (
        Int128(A.m1) * B.m1 + 
        Int128(A.m2) * B.m2 + 
        Int128(A.m3) * B.m3 + 
        Int128(A.m4) * B.m4 - 
        Int128(A.m6) * B.m6 - 
        Int128(A.m7) * B.m7 - 
        Int128(A.m8) * B.m8 - 
        Int128(A.m12) * B.m12
    )

    # Grade 1
    # Combines vector-scalar products and bivector-vector contractions
    m2_acc = (
        Int128(A.m1) * B.m2 + 
        Int128(A.m2) * B.m1 + 
        Int128(A.m6) * B.m3 - 
        Int128(A.m3) * B.m6 +
        Int128(A.m4) * B.m8 -
        Int128(A.m8) * B.m4 -
        Int128(A.m12) * B.m7 -
        Int128(A.m7) * B.m12
    )
    m3_acc = (
        Int128(A.m1) * B.m3 + 
        Int128(A.m3) * B.m1 - 
        Int128(A.m6) * B.m2 + 
        Int128(A.m2) * B.m6 +
        Int128(A.m7) * B.m4 -
        Int128(A.m4) * B.m7 -
        Int128(A.m12) * B.m8 -
        Int128(A.m8) * B.m12
    )
    m4_acc = (
        Int128(A.m1) * B.m4 + 
        Int128(A.m4) * B.m1 - 
        Int128(A.m7) * B.m3 + 
        Int128(A.m3) * B.m7 +
        Int128(A.m8) * B.m2 -
        Int128(A.m2) * B.m8 -
        Int128(A.m12) * B.m6 -
        Int128(A.m6) * B.m12
    )
    m5_acc = (
        Int128(A.m1) * B.m5 + 
        Int128(A.m5) * B.m1 + 
        Int128(A.m9) * B.m2 - 
        Int128(A.m2) * B.m9 +
        Int128(A.m10) * B.m3 - 
        Int128(A.m3) * B.m10 +
        Int128(A.m11) * B.m4 - 
        Int128(A.m4) * B.m11 -
        Int128(A.m13) * B.m7 - 
        Int128(A.m7) * B.m13 -
        Int128(A.m14) * B.m8 - 
        Int128(A.m8) * B.m14 -
        Int128(A.m15) * B.m6 - 
        Int128(A.m6) * B.m15 -
        Int128(A.m16) * B.m12 +
        Int128(A.m12) * B.m16
    )

    # Grade 2
    # Captures rotations, magnetic fields, and angular momentum seamlessly.
    m6_acc = (
        Int128(A.m1) * B.m6 + 
        Int128(A.m6) * B.m1 + 
        Int128(A.m2) * B.m3 - 
        Int128(A.m3) * B.m2 -
        Int128(A.m7) * B.m8 +
        Int128(A.m8) * B.m7 +
        Int128(A.m4) * B.m12 +
        Int128(A.m12) * B.m4
    )
    m7_acc = (
        Int128(A.m1) * B.m7 + 
        Int128(A.m7) * B.m1 + 
        Int128(A.m3) * B.m4 - 
        Int128(A.m4) * B.m3 +
        Int128(A.m6) * B.m8 -
        Int128(A.m8) * B.m6 +
        Int128(A.m2) * B.m12 +
        Int128(A.m12) * B.m2
    )
    m8_acc = (
        Int128(A.m1) * B.m8 + 
        Int128(A.m8) * B.m1 + 
        Int128(A.m4) * B.m2 - 
        Int128(A.m2) * B.m4 -
        Int128(A.m6) * B.m7 +
        Int128(A.m7) * B.m6 +
        Int128(A.m3) * B.m12 +
        Int128(A.m12) * B.m3
    )
    m9_acc = (
        Int128(A.m1) * B.m9 + 
        Int128(A.m9) * B.m1 + 
        Int128(A.m5) * B.m2 - 
        Int128(A.m2) * B.m5 +
        Int128(A.m6) * B.m10 - 
        Int128(A.m10) * B.m6 -
        Int128(A.m8) * B.m11 + 
        Int128(A.m11) * B.m8 +
        Int128(A.m3) * B.m15 + 
        Int128(A.m15) * B.m3 -
        Int128(A.m4) * B.m14 - 
        Int128(A.m14) * B.m4 +
        Int128(A.m12) * B.m13 - 
        Int128(A.m13) * B.m12 -
        Int128(A.m16) * B.m7 - 
        Int128(A.m7) * B.m16
    )
    m10_acc = (
        Int128(A.m1) * B.m10 + 
        Int128(A.m10) * B.m1 + 
        Int128(A.m5) * B.m3 - 
        Int128(A.m3) * B.m5 +
        Int128(A.m7) * B.m11 - 
        Int128(A.m11) * B.m7 -
        Int128(A.m6) * B.m9 + 
        Int128(A.m9) * B.m6 +
        Int128(A.m4) * B.m13 + 
        Int128(A.m13) * B.m4 -
        Int128(A.m2) * B.m15 - 
        Int128(A.m15) * B.m2 +
        Int128(A.m12) * B.m14 - 
        Int128(A.m14) * B.m12 -
        Int128(A.m16) * B.m8 - 
        Int128(A.m8) * B.m16
    )
    m11_acc = (
        Int128(A.m1) * B.m11 + 
        Int128(A.m11) * B.m1 + 
        Int128(A.m5) * B.m4 - 
        Int128(A.m4) * B.m5 +
        Int128(A.m8) * B.m9 - 
        Int128(A.m9) * B.m8 -
        Int128(A.m7) * B.m10 + 
        Int128(A.m10) * B.m7 +
        Int128(A.m2) * B.m14 + 
        Int128(A.m14) * B.m2 -
        Int128(A.m3) * B.m13 - 
        Int128(A.m13) * B.m3 +
        Int128(A.m12) * B.m15 - 
        Int128(A.m15) * B.m12 -
        Int128(A.m16) * B.m6 - 
        Int128(A.m6) * B.m16
    )

    # Grade 3
    m12_acc = (
        Int128(A.m1) * B.m12 + 
        Int128(A.m12) * B.m1 + 
        Int128(A.m2) * B.m7 + 
        Int128(A.m7) * B.m2 +
        Int128(A.m3) * B.m8 +
        Int128(A.m8) * B.m3 +
        Int128(A.m4) * B.m6 +
        Int128(A.m6) * B.m4
    )
    m13_acc = (
        Int128(A.m1) * B.m13 + 
        Int128(A.m13) * B.m1 + 
        Int128(A.m5) * B.m7 + 
        Int128(A.m7) * B.m5 -
        Int128(A.m3) * B.m11 - 
        Int128(A.m11) * B.m3 +
        Int128(A.m4) * B.m10 + 
        Int128(A.m10) * B.m4 -
        Int128(A.m2) * B.m16 + 
        Int128(A.m16) * B.m2 +
        Int128(A.m9) * B.m12 - 
        Int128(A.m12) * B.m9 +
        Int128(A.m6) * B.m14 - 
        Int128(A.m14) * B.m6 -
        Int128(A.m8) * B.m15 + 
        Int128(A.m15) * B.m8
    )
    m14_acc = (
        Int128(A.m1) * B.m14 + 
        Int128(A.m14) * B.m1 + 
        Int128(A.m5) * B.m8 + 
        Int128(A.m8) * B.m5 -
        Int128(A.m4) * B.m9 - 
        Int128(A.m9) * B.m4 +
        Int128(A.m2) * B.m11 + 
        Int128(A.m11) * B.m2 -
        Int128(A.m3) * B.m16 + 
        Int128(A.m16) * B.m3 +
        Int128(A.m10) * B.m12 - 
        Int128(A.m12) * B.m10 +
        Int128(A.m7) * B.m15 - 
        Int128(A.m15) * B.m7 -
        Int128(A.m6) * B.m13 + 
        Int128(A.m13) * B.m6
    )
    m15_acc = (
        Int128(A.m1) * B.m15 + 
        Int128(A.m15) * B.m1 + 
        Int128(A.m5) * B.m6 + 
        Int128(A.m6) * B.m5 -
        Int128(A.m2) * B.m10 - 
        Int128(A.m10) * B.m2 +
        Int128(A.m3) * B.m9 + 
        Int128(A.m9) * B.m3 -
        Int128(A.m4) * B.m16 + 
        Int128(A.m16) * B.m4 +
        Int128(A.m11) * B.m12 - 
        Int128(A.m12) * B.m11 +
        Int128(A.m8) * B.m13 - 
        Int128(A.m13) * B.m8 -
        Int128(A.m7) * B.m14 + 
        Int128(A.m14) * B.m7
    )

    # Grade 4
    m16_acc = (
        Int128(A.m1) * B.m16 + 
        Int128(A.m16) * B.m1 + 
        Int128(A.m5) * B.m12 - 
        Int128(A.m12) * B.m5 -
        Int128(A.m2) * B.m13 + 
        Int128(A.m13) * B.m2 -
        Int128(A.m3) * B.m14 + 
        Int128(A.m14) * B.m3 -
        Int128(A.m4) * B.m15 + 
        Int128(A.m15) * B.m4 +
        Int128(A.m9) * B.m7 + 
        Int128(A.m7) * B.m9 +
        Int128(A.m10) * B.m8 + 
        Int128(A.m8) * B.m10 +
        Int128(A.m11) * B.m6 + 
        Int128(A.m6) * B.m11
    )

    return Multivector16(
        # Grade 0
        Int64(div(m1_acc, FIXED_SCALE)),
        
        # Grade 1
        Int64(div(m2_acc, FIXED_SCALE)),
        Int64(div(m3_acc, FIXED_SCALE)),
        Int64(div(m4_acc, FIXED_SCALE)),
        Int64(div(m5_acc, FIXED_SCALE)),
        
        # Grade 2
        Int64(div(m6_acc, FIXED_SCALE)),
        Int64(div(m7_acc, FIXED_SCALE)),
        Int64(div(m8_acc, FIXED_SCALE)),
        Int64(div(m9_acc, FIXED_SCALE)),
        Int64(div(m10_acc, FIXED_SCALE)),
        Int64(div(m11_acc, FIXED_SCALE)),
        
        # Grade 3
        Int64(div(m12_acc, FIXED_SCALE)),
        Int64(div(m13_acc, FIXED_SCALE)),
        Int64(div(m14_acc, FIXED_SCALE)),
        Int64(div(m15_acc, FIXED_SCALE)),
        
        # Grade 4
        Int64(div(m16_acc, FIXED_SCALE))
    )
end
