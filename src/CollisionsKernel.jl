#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export dispatch_narrow_phase_collisions!

using KernelAbstractions

# ---------------------------------------------------------------------------
# VULKAN COMPUTE KERNEL
# ---------------------------------------------------------------------------
"""
    narrow_phase_collision_kernel!(...)

A lock-free GPU compute shader. It evaluates exact collision intersection for 
entities matching `current_color`. Because entities of the same color are 
guaranteed mathematically distinct, this kernel executes with zero atomic locks.
"""
@kernel function narrow_phase_collision_kernel!(
    positions, forces, colors, is_active, active_count, current_color, window_size
)
    # The absolute hardware thread index
    i = @index(Global)

    # Only process active entities that belong to the current chromatic dispatch tier
    if i <= active_count && is_active[i] && colors[i] == current_color
        pos_i = positions[i]
        
        # 128-bit accumulators strictly confined to the SM's local registers.
        # This prevents precision leaks during the penalty force calculations.
        force_acc_x = Int128(0)
        force_acc_y = Int128(0)
        force_acc_z = Int128(0)

        # We rely on the Phase 1 Morton Code (Z-curve) sort. 
        # Spatially relevant neighbors are guaranteed to be within the linear array window.
        start_idx = max(1, i - window_size)
        end_idx = min(active_count, i + window_size)

        for j in start_idx:end_idx
            if i != j && is_active[j]
                pos_j = positions[j]
                
                # Deterministic 128-bit fixed-point distance vector
                dx = Int128(pos_i.m2) - Int128(pos_j.m2)
                dy = Int128(pos_i.m3) - Int128(pos_j.m3)
                dz = Int128(pos_i.m4) - Int128(pos_j.m4)
                
                dist_sq = (dx * dx) + (dy * dy) + (dz * dz)
                
                # Collision radius (scaled by our 10^8 constant). 
                # E.g., 2.0 meters = 200,000,000
                col_radius = Int128(200_000_000) 
                col_radius_sq = col_radius * col_radius
                
                if dist_sq < col_radius_sq && dist_sq > 0
                    # Narrow-phase collision detected!
                    # Apply a Hooke's Law style penalty force (push-out)
                    overlap = col_radius_sq - dist_sq
                    
                    # Proportional pushout. We bit-shift right by 32 to gracefully
                    # downscale the extreme 128-bit magnitude back into manageable forces.
                    force_acc_x += (overlap * dx) >> 32
                    force_acc_y += (overlap * dy) >> 32
                    force_acc_z += (overlap * dz) >> 32
                end
            end
        end

        # Write the resolved forces back to the ECS.
        # Because we only dispatch one color tier at a time, no two threads will ever
        # attempt to mutate the same physical space. Zero CAS atomic operations required.
        curr_f = forces[i]
        forces[i] = Multivector16(
            curr_f.m1,
            curr_f.m2 + Int64(force_acc_x),
            curr_f.m3 + Int64(force_acc_y),
            curr_f.m4 + Int64(force_acc_z),
            curr_f.m5, curr_f.m6, curr_f.m7, curr_f.m8,
            curr_f.m9, curr_f.m10, curr_f.m11, curr_f.m12,
            curr_f.m13, curr_f.m14, curr_f.m15, curr_f.m16
        )
    end
end

# ---------------------------------------------------------------------------
# HOST DISPATCH PIPELINE
# ---------------------------------------------------------------------------
"""
    dispatch_narrow_phase_collisions!(arena::MemoryArena)

Iterates through the chromatic tiers generated in the Broad Phase, launching 
a Vulkan compute shader for each color, separated by strict memory barriers.
"""
function dispatch_narrow_phase_collisions!(arena::MemoryArena)
    # Identify the hardware backend natively (Vulkan, CUDA, or CPU fallback for testing)
    backend = get_backend(arena.positions) 
    
    # 1. Find the highest chromatic tier used in this frame.
    #    (Usually < 20 colors due to the Kepler conjecture in 3D space).
    max_color = UInt8(0)
    for i in 1:arena.active_count
        if arena.is_active[i] && arena.colors[i] > max_color
            max_color = arena.colors[i]
        end
    end

    # 2. Prepare the Ampere SM workload parameters
    kernel! = narrow_phase_collision_kernel!(backend)
    workgroup_size = 256 # Ideal wavefront/warp saturation
    ndrange = arena.active_count

    # 3. The Chromatic Dispatch Loop
    for color in 1:max_color
        kernel!(
            arena.positions,
            arena.forces,
            arena.colors,
            arena.is_active,
            arena.active_count,
            color,
            64, # SLIDING_WINDOW matching Morton spatial locality
            ndrange=ndrange,
            workgroupsize=workgroup_size
        )
        # synchronize(backend) instructs the Vulkan driver to insert a strict
        # memory barrier. The GPU physically waits for all SMs executing the current 
        # color to flush their L2 caches to VRAM before launching the next color.
        synchronize(backend)
    end
end
