#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export calculate_morton_3d, get_morton_from_multivector

# To pack 3 coordinates into a 64-bit integer, we are limited to 21 bits per coordinate.
# 21 bits gives us a grid resolution of 2,097,152 units per axis.
const MORTON_BITS = 21
const MORTON_MASK = 0x00000000001fffff # 21 bits

"""
    expand_bits(v::UInt64)

Expands a 21-bit integer by inserting two zeros after each bit.
This is the core bitwise magic required for 3D Z-Order Curve interleaving.
"""
@inline function expand_bits(v::UInt64)::UInt64
    v &= MORTON_MASK
    v = (v | (v << 32)) & 0x001f00000000ffff
    v = (v | (v << 16)) & 0x001f0000ff0000ff
    v = (v | (v <<  8)) & 0x100f00f00f00f00f
    v = (v | (v <<  4)) & 0x10c30c30c30c30c3
    v = (v | (v <<  2)) & 0x1249249249249249
    return v
end

"""
    calculate_morton_3d(x::UInt64, y::UInt64, z::UInt64)

Interleaves the bits of 3 independent axes into a single 1D Morton Code.
"""
@inline function calculate_morton_3d(x::UInt64, y::UInt64, z::UInt64)::UInt64
    xx = expand_bits(x)
    yy = expand_bits(y)
    zz = expand_bits(z)
    
    # Shift and combine: x at bit 0, y at bit 1, z at bit 2
    return xx | (yy << 1) | (zz << 2)
end

"""
    get_morton_from_multivector(pos::Multivector16, universe_bounds::Int64)

Extracts the x, y, and z translation components from our fixed-point multivector.
Offsets the values by the universe bounds to ensure all values are positive 
before casting to UInt64 for bitwise operations.
"""
@inline function get_morton_from_multivector(pos::Multivector16, universe_bounds::Int64)::UInt64
    # Assuming Grade 1 vectors (m2, m3, m4) map to x, y, z translations in our specific GA mapping.
    # We must divide by FIXED_SCALE if we want the Morton grid to represent whole engine units,
    # or keep the scale to track sub-unit microscopic collisions. 
    # For this implementation, we will hash the scaled values for maximum precision.
    
    # Offset by bounds to guarantee positive integers for bit manipulation
    x_pos = UInt64(max(0, pos.m2 + universe_bounds))
    y_pos = UInt64(max(0, pos.m3 + universe_bounds))
    z_pos = UInt64(max(0, pos.m4 + universe_bounds))
    
    return calculate_morton_3d(x_pos, y_pos, z_pos)
end
