#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export update_spatial_hashes!

"""
    update_spatial_hashes!(arena::MemoryArena, universe_bounds::Int64)

The main event loop hook for micro-scale spatial hashing. 
It operates strictly within the `active_count` boundary of the ECS.
Because all entity data is flattened into contiguous 1D memory arrays, 
these reads and writes perfectly coalesce and execute with zero allocations.
"""
function update_spatial_hashes!(arena::MemoryArena, universe_bounds::Int64)
    # The swap-and-pop logic ensures there are no holes in our arrays,
    # meaning everything from index 1 to active_count is a valid, living entity.
    
    # We use @inbounds to disable bounds checking, and @simd to allow the CPU 
    # to vectorize the loop, preparing the data efficiently before it hits the GPU.
    @inbounds @simd for i in 1:arena.active_count
        # We perform a safety check, though theoretically redundant due to swap-and-pop
        if arena.is_active[i]
            # Extract the fixed-point 16-component multivector
            pos = arena.positions[i]
            
            # Interleave the 3D coordinates into a 1D Morton code and write it back
            arena.morton_codes[i] = get_morton_from_multivector(pos, universe_bounds)
        end
    end
    
    # Note: In the complete pipeline, this updated `arena.morton_codes` array 
    # is what gets synced via Vulkan Sparse Buffers to the GPU VRAM, 
    # where the SMs will execute a highly parallel Radix Sort.
end
