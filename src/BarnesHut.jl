#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export compute_gravity_step!

# The Barnes-Hut threshold (theta squared). 
# If (node_width^2 / distance^2) < THETA_SQ, we treat the SVDAG branch as a single center of mass.
const THETA_SQ = 0.25 

# Gravitational constant in fixed-point space. Tunable for engine/gameplay feel.
const GRAVITY_CONSTANT = 100_000_000

"""
    compute_gravity_step!(arena::MemoryArena, svdag::SVDAGArena)

Executes the macro-scale astrophysics pass. Exact math is local; distant matter 
is approximated as a single multivector to achieve O(N log N) complexity.
Maintains pure determinism via 128-bit local register accumulation.
"""
function compute_gravity_step!(arena::MemoryArena, svdag::SVDAGArena)
    # Pre-allocate bounded stacks for tree traversal to appease the Julia GC.
    # A depth of 1024 prevents stack overflows even in impossibly dense astrophysical octrees,
    # ensuring zero heap allocations inside the main physics loop.
    MAX_DEPTH = 1024
    traversal_stack = fill(UInt32(0), MAX_DEPTH)
    width_stack = fill(Int64(0), MAX_DEPTH)
    
    @inbounds for i in 1:arena.active_count
        if !arena.is_active[i]
            continue
        end
        
        pos = arena.positions[i]
        
        # 128-bit accumulators kept in local SM registers to prevent Hamiltonian energy loss.
        # These catch the massive fixed-point multiplication scales before down-converting.
        force_acc_x = Int128(0)
        force_acc_y = Int128(0)
        force_acc_z = Int128(0)
        
        # Start traversal from the SVDAG root (Index 1)
        stack_ptr = 1
        traversal_stack[stack_ptr] = UInt32(1)
        width_stack[stack_ptr] = 1_000_000_000 * FIXED_SCALE # Initial macro-sector width
        
        while stack_ptr > 0
            # Pop the current node off the zero-allocation stack
            current_node_idx = traversal_stack[stack_ptr]
            current_width = width_stack[stack_ptr]
            stack_ptr -= 1
            
            node = svdag.nodes[current_node_idx]
            
            com_x = svdag.com_x[current_node_idx]
            com_y = svdag.com_y[current_node_idx]
            com_z = svdag.com_z[current_node_idx]
            node_mass = svdag.node_mass[current_node_idx]
            
            # Calculate 128-bit distance vector to prevent macro-scale coordinate overflow.
            # (pos - com) gives a vector pointing FROM the CoM TO the entity.
            dx = Int128(pos.m2) - Int128(com_x)
            dy = Int128(pos.m3) - Int128(com_y)
            dz = Int128(pos.m4) - Int128(com_z)
            
            # Squared distance calculated natively in 128-bit integers (scaled by 10^16)
            dist_sq = (dx * dx) + (dy * dy) + (dz * dz)
            
            # Prevent self-interaction and division by zero
            if dist_sq == 0
                continue
            end
            
            # Check the Barnes-Hut threshold: (width^2 / dist_sq) < THETA_SQ
            width_sq = Int128(current_width) * Int128(current_width)
            is_distant = (width_sq * 100) < (dist_sq * 25) # 25 is THETA_SQ * 100
            
            # If the node is distant enough OR it is a dense matter leaf node, calculate force
            if is_distant || node.leaf_mask != 0x00
                # Distance 'r' via deterministic integer square root.
                # Since dist_sq is 10^16 scaled, isqrt(10^16) = 10^8. r is perfectly scaled!
                r = isqrt(dist_sq)
                
                # To avoid dropping precision, we compute the force entirely in Int128 registers.
                # F = (G * m1 * m2) / r^2. Vector form: F_vec = (G * m1 * m2 * r_vec) / r^3
                # Numerator: G * mass * FIXED_SCALE ensures (10^8 * 10^8 * 10^8) = 10^24 magnitude.
                num_base = Int128(GRAVITY_CONSTANT) * node_mass * FIXED_SCALE
                
                # Denominator: r^3 = dist_sq * r (10^16 * 10^8 = 10^24 magnitude).
                denom = dist_sq * r
                
                # Since dx, dy, dz point AWAY from the CoM, we subtract them from the accumulator
                # so gravity pulls the entity TOWARDS the CoM.
                # The division natively drops the scale back to 10^8 (10^32 / 10^24 = 10^8).
                force_acc_x -= (num_base * dx) ÷ denom
                force_acc_y -= (num_base * dy) ÷ denom
                force_acc_z -= (num_base * dz) ÷ denom
            else
                # Node is too close and has children. Push children to the stack.
                half_width = current_width >> 1
                for octant in 0:7
                    child_idx = get_child_index(node, octant)
                    if child_idx != 0
                        stack_ptr += 1
                        traversal_stack[stack_ptr] = child_idx
                        width_stack[stack_ptr] = half_width
                    end
                end
            end
        end
        
        # Downscale the 128-bit register accumulations and apply to the 64-bit ECS force array.
        # We reconstruct the Multivector16 entirely on the stack to maintain zero-allocation 
        # while preserving the magnetic/rotational grades untouched.
        curr_f = arena.forces[i]
        arena.forces[i] = Multivector16(
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
