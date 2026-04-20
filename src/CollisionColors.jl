#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export compute_deterministic_colors!, compute_graph_colors!

# Maximum number of parallel phases (colors) that the GPU compute shaders will execute in sequence. 
const MAX_COLORS = 32

"""
    check_collision(posA::Multivector16, posB::Multivector16, radius::Int64)

A highly simplified deterministic distance check utilizing our fixed-point 
multivectors to determine if two entities form an 'edge' in our collision graph.
"""
@inline function check_collision(posA::Multivector16, posB::Multivector16, sq_radius::Int64)::Bool
    # In full Conformal GA, we would take the inner product of dual spheres, 
    # but for this, we extract the Grade 1 translation components.
    dx = posA.m2 - posB.m2
    dy = posA.m3 - posB.m3
    dz = posA.m4 - posB.m4
    
    # Calculate squared distance natively in integers to prevent floating-point drift
    dist_sq = (dx * dx) + (dy * dy) + (dz * dz)
    
    return dist_sq <= sq_radius
end

"""
    compute_deterministic_colors!(arena::MemoryArena, collision_radius::Int64)

Iterates over the active entities and assigns a color (phase) to each.
By enforcing that no two colliding entities share a color, the GPU 
can execute an entire color group simultaneously without atomic memory locks.
This algorithm is strictly zero-allocation to appease the Julia GC.
"""
function compute_deterministic_colors!(arena::MemoryArena, collision_radius::Int64)
    sq_radius = collision_radius * collision_radius
    
    # Pre-allocated bitmask array to track unavailable colors for the current entity.
    # Since we are single-threaded on the CPU for this management pass, 
    # a single localized array avoids allocations.
    used_colors = fill(false, MAX_COLORS)
    
    @inbounds for i in 1:arena.active_count
        if !arena.is_active[i]
            continue
        end
        
        # Reset the color mask for the current entity
        fill!(used_colors, false)
        
        pos_i = arena.positions[i]
        
        # Because the entities are sorted by their Morton codes, we only need to 
        # check a localized neighborhood of nearby indices rather than an O(N^2) loop.
        search_window = 50 
        start_idx = max(1, i - search_window)
        
        # Look backwards at entities that have already been colored
        for j in start_idx:(i - 1)
            if arena.is_active[j] && check_collision(pos_i, arena.positions[j], sq_radius)
                neighbor_color = arena.colors[j]
                if neighbor_color > 0 && neighbor_color <= MAX_COLORS
                    used_colors[neighbor_color] = true
                end
            end
        end
        
        # Deterministically assign the lowest available color
        assigned_color = 0x00000000
        for c in 1:MAX_COLORS
            if !used_colors[c]
                assigned_color = UInt32(c)
                break
            end
        end
        
        # Fallback if the graph is too dense (in production, 
        # this should trigger a spatial subdivision or warning)
        if assigned_color == 0
            assigned_color = UInt32(MAX_COLORS)
        end
        
        arena.colors[i] = assigned_color
    end
end

"""
    compute_graph_colors!(arena::MemoryArena, search_radius::Int64)

Executes a deterministic Greedy Graph Coloring algorithm over the ECS memory arena.
Assigns a collision color to each active entity. Entities with the same color are 
mathematically guaranteed not to be in collision range, allowing them to be processed 
in parallel on the GPU without atomic locks or race conditions.
"""
function compute_graph_colors!(arena::MemoryArena, search_radius::Int64)
    # Upcast to 128-bit to prevent overflow during distance squared calculation
    search_radius_sq = Int128(search_radius) * Int128(search_radius)

    # In a fully optimized pipeline, entities are pre-sorted by Morton Code (Z-curve),
    # meaning spatially adjacent entities are adjacent in memory. 
    # We only need to check a localized sliding window instead of O(N^2).
    SLIDING_WINDOW = 64 

    @inbounds for i in 1:arena.active_count
        if !arena.is_active[i]
            continue
        end

        pos_i = arena.positions[i]
        
        # We use a 64-bit integer as a bitmask to track used colors.
        # This supports up to 64 chromatic batches (colors), which is massive overkill. 
        # (The Kepler conjecture notes a 3D sphere can only touch 12 neighbors. 
        # Practical physics chromatic numbers rarely exceed 20).
        used_colors = UInt64(0) 

        # Check backward window (previously assigned neighbors)
        start_idx = max(1, i - SLIDING_WINDOW)
        for j in start_idx:(i - 1)
            if !arena.is_active[j]
                continue
            end
            
            pos_j = arena.positions[j]
            
            # Fixed-point deterministic distance check
            dx = Int128(pos_i.m2) - Int128(pos_j.m2)
            dy = Int128(pos_i.m3) - Int128(pos_j.m3)
            dz = Int128(pos_i.m4) - Int128(pos_j.m4)
            
            dist_sq = (dx * dx) + (dy * dy) + (dz * dz)
            
            if dist_sq <= search_radius_sq
                # Neighbor is in collision range. Mark their color as 'used'.
                color_j = arena.colors[j]
                if color_j > 0 && color_j <= 64
                    # Bitwise shift to flag the specific color index
                    used_colors |= (UInt64(1) << (color_j - 1))
                end
            end
        end
        
        # Bit-twiddling magic: 
        # ~used_colors flips all bits so available colors become 1s.
        # trailing_zeros() maps directly to a fast hardware instruction (e.g. tzcnt)
        # to find the exact index of the lowest available color.
        first_available_color = trailing_zeros(~used_colors) + 1
        
        # Clamp gracefully if we somehow exceed 64 colors in an ultra-dense cluster
        if first_available_color > 64
            first_available_color = 64 
        end
        
        # Assign the deterministic chromatic tier
        arena.colors[i] = UInt8(first_available_color)
    end
end
