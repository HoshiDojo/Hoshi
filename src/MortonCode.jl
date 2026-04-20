#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export calculate_morton_3d, get_morton_from_multivector, sort_arena_by_morton!

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

# ---------------------------------------------------------------------------
# BIT-TWIDDLING MAGIC
# Expands a 21-bit integer by inserting two zeros after every bit.
# e.g., 00000000000000000000000000000111 -> 00000000000000000000000001001001
# ---------------------------------------------------------------------------
@inline function expand_bits_21(v::UInt64)
    v = (v | (v << 32)) & 0x001f00000000ffff
    v = (v | (v << 16)) & 0x001f0000ff0000ff
    v = (v | (v <<  8)) & 0x010f00f00f00f00f
    v = (v | (v <<  4)) & 0x10c30c30c30c30c3
    v = (v | (v <<  2)) & 0x1249249249249249
    return v
end

@inline function morton_3d(x::UInt64, y::UInt64, z::UInt64)
    # Interleave the bits: X takes bits 0,3,6... Y takes 1,4,7... Z takes 2,5,8...
    return expand_bits_21(x) | (expand_bits_21(y) << 1) | (expand_bits_21(z) << 2)
end

# ---------------------------------------------------------------------------
# COORDINATE NORMALIZATION
# ---------------------------------------------------------------------------
"""
Translates a signed 64-bit fixed-point coordinate into a strictly positive 21-bit grid index.
"""
@inline function pos_to_grid(p::Int64)
    # Shift domain so 0 is center. Assumes playable space fits in +/- 2^40 fixed-point units.
    domain_offset = p + (Int64(1) << 40)
    
    # Shift down to compress the extreme fixed-point resolution into a 21-bit space.
    # We don't need sub-millimeter precision for the Broad Phase sort, just spatial grouping.
    grid_val = domain_offset >> 20
    
    # Clamp safely to 21-bit unsigned max (2,097,151)
    return UInt64(max(0, min(grid_val, 0x1FFFFF)))
end

"""
    sort_arena_by_morton!(arena::MemoryArena, scratch_perm::Vector{Int})

Calculates Z-Order curves for all active entities and sorts the Struct of Arrays (SoA)
in place. This guarantees spatial locality in linear memory, optimizing hardware cache 
hits for the Graph Coloring sliding window and Barnes-Hut tree building.
"""
function sort_arena_by_morton!(arena::MemoryArena, scratch_perm::Vector{Int})
    active_count = arena.active_count
    if active_count == 0
        return
    end

    # In a true zero-allocation engine, this array must be pre-allocated in the outer loop.
    morton_codes = Vector{UInt64}(undef, active_count)
    
    # 1. Compute Morton codes for all entities
    @inbounds for i in 1:active_count
        if arena.is_active[i]
            pos = arena.positions[i]
            # m2, m3, m4 map to X, Y, Z in the GA Multivector
            x_grid = pos_to_grid(pos.m2)
            y_grid = pos_to_grid(pos.m3)
            z_grid = pos_to_grid(pos.m4)
            morton_codes[i] = morton_3d(x_grid, y_grid, z_grid)
        else
            # Push inactive entities to the very end of the sorted arrays
            morton_codes[i] = typemax(UInt64)
        end
    end

    # 2. Generate the sorting permutation
    # Base.sortperm creates an array of indices. We write to the pre-allocated scratch_perm
    # to avoid triggering the GC.
    sortperm!(scratch_perm, morton_codes)

    # 3. Apply the permutation (In-place SoA shuffle)
    # We re-align the entire struct of arrays based on the new spatial Z-curve sequence.
    arena.positions[1:active_count] .= arena.positions[scratch_perm]
    arena.forces[1:active_count] .= arena.forces[scratch_perm]
    arena.is_active[1:active_count] .= arena.is_active[scratch_perm]
    
    # If the colors array has been allocated we sort it too.
    if isdefined(arena, :colors)
        arena.colors[1:active_count] .= arena.colors[scratch_perm]
    end
end
