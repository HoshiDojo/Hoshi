#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export MemoryArena, init_arena, spawn_entity!, destroy_entity!

# For the sake of the engine's initial architecture, we define a maximum capacity.
const MAX_ENTITIES = 100_000

"""
    MemoryArena

The core SoA ECS structure. All sim state is laid out in flat, contiguous 
1D Arrays of fixed-point multivector components to ensure memory requests 
from the GPU warps are perfectly coalesced. 
"""
mutable struct MemoryArena
    # Entity lifecycle tracking
    active_count::Int64
    is_active::Vector{Bool}

    # Core physical components (1D arrays of Multivector16)
    # Positions represent translation in Conformal GA
    positions::Vector{Multivector16}
    
    # Velocities represent rotors/translators
    velocities::Vector{Multivector16}
    
    # Represents force vectors acting upon the entity
    forces::Vector{Multivector16}
    
    # 1D Array for Morton Codes (Spatial Hashing)
    morton_codes::Vector{UInt64}
    
    # Graph Coloring for deterministic lock-free physics execution
    colors::Vector{UInt32}
end

"""
    init_arena()

Pre-allocates the entire engine memory footprint into custom Memory Arenas.
This must only be called during the loading screen to prevent 10-100ms GC pauses 
during the main simulation event loop.
"""
function init_arena()::MemoryArena
    # Instantiate zero-state Multivector16 for initialization
    empty_mv = Multivector16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    return MemoryArena(
        0,                                         # active_count
        fill(false, MAX_ENTITIES),                 # is_active
        fill(empty_mv, MAX_ENTITIES),              # positions
        fill(empty_mv, MAX_ENTITIES),              # velocities
        fill(empty_mv, MAX_ENTITIES),              # forces
        fill(zero(UInt64), MAX_ENTITIES),          # morton_codes
        fill(zero(UInt32), MAX_ENTITIES)           # colors
    )
end

"""
    spawn_entity!(arena::MemoryArena, pos::Multivector16, vel::Multivector16)

Zero-allocation entity spawning. Overwrites existing memory at the 
next available index instead of triggering the Julia GC.
"""
function spawn_entity!(arena::MemoryArena, pos::Multivector16, vel::Multivector16)::Int64
    if arena.active_count >= MAX_ENTITIES
        # In network cases, exceeding the capacity should be handled gracefully,
        # but for absolute strictness, we enforce the hard limit.
        error("Memory Arena Overflow: Cannot spawn entity, MAX_ENTITIES reached.")
    end
    
    # Increment our active pointer
    arena.active_count += 1
    idx = arena.active_count
    
    # Overwrite the contiguous 1D array data natively
    @inbounds arena.is_active[idx] = true
    @inbounds arena.positions[idx] = pos
    @inbounds arena.velocities[idx] = vel
    
    # Reset forces and structural data for the recycled slot
    @inbounds arena.forces[idx] = Multivector16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    @inbounds arena.morton_codes[idx] = 0
    @inbounds arena.colors[idx] = 0
    
    return idx
end

"""
    destroy_entity!(arena::MemoryArena, idx::Int64)

Executes zero-allocation entity destruction using the swap-and-pop technique. 
[span_2](start_span)[span_3](start_span)To ensure memory requests from the GPU warps are perfectly coalesced[span_2](end_span)[span_3](end_span),
we cannot leave holes in our active memory. We swap the destroyed entity's 
data with the last active entity in the array, then shrink the active boundary.
"""
function destroy_entity!(arena::MemoryArena, idx::Int64)
    # Prevent operations on already dead entities
    if !arena.is_active[idx]
        return
    end
    
    last_idx = arena.active_count
    
    # If the entity being destroyed is NOT the last active entity,
    # we copy the last active entity's data into the target index's slot.
    if idx != last_idx
        @inbounds arena.positions[idx] = arena.positions[last_idx]
        @inbounds arena.velocities[idx] = arena.velocities[last_idx]
        @inbounds arena.forces[idx] = arena.forces[last_idx]
        @inbounds arena.morton_codes[idx] = arena.morton_codes[last_idx]
        @inbounds arena.colors[idx] = arena.colors[last_idx]
        # We don't need to touch `is_active` for `idx` because it remains true
        # (it's now occupied by the previously last active entity).
    end
    
    # Scrub the last slot (which has now been moved, or is the one being destroyed).
    # While the active_count boundary mathematically prevents the engine from reading 
    # this data, zeroing it out prevents visual/state artifacts during debugging.
    empty_mv = Multivector16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    @inbounds arena.positions[last_idx] = empty_mv
    @inbounds arena.velocities[last_idx] = empty_mv
    @inbounds arena.forces[last_idx] = empty_mv
    @inbounds arena.morton_codes[last_idx] = 0
    @inbounds arena.colors[last_idx] = 0
    @inbounds arena.is_active[last_idx] = false
    
    # Shrink the active boundary
    arena.active_count -= 1
end
