#=
Copyright 2026 The Hoshi Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
=#
export SVDAGNode, SVDAGArena, init_svdag_arena, get_child_index, compute_upward_pass!

const MAX_SVDAG_NODES = 10_000_000

"""
    SVDAGNode

A brutally compressed 48-bit (padded to 64-bit) structure representing a volume of space.
To traverse the DAG, we use bitwise operations on the masks to find the contiguous 
block of children in our pre-allocated array.
"""
struct SVDAGNode
    # 8 bits representing the 8 octants of 3D space. 1 = occupied, 0 = vacuum.
    child_mask::UInt8   
    
    # 8 bits to denote if a child is a leaf (dense matter) or another branch.
    leaf_mask::UInt8    
    
    # Unused padding to align memory for the Ampere GPU.
    _padding::UInt16    
    
    # 32-bit index pointing to where the contiguous block of children begins in the Arena.
    # O(1) lookup: child_index = child_base_idx + count_ones(child_mask before this octant)
    child_base_idx::UInt32 
end

"""
    SVDAGArena

The pre-allocated memory topological map of the universe.
All spatial queries (gravity, fluids, raytracing) read from this contiguous array.
"""
mutable struct SVDAGArena
    active_node_count::UInt32
    
    # The flattened DAG. Index 1 is always the root node of the universe.
    nodes::Vector{SVDAGNode}
    
    # For dense matter leaves, we track material properties (mass, state)
    # mapped to the same indices as the leaf nodes.
    leaf_materials::Vector{UInt32} 

    # Parallel Arrays for Barnes-Hut
    node_mass::Vector{Int64}
    com_x::Vector{Int64}
    com_y::Vector{Int64}
    com_z::Vector{Int64}
end

"""
    init_svdag_arena()

Pre-allocates the universe's spatial grid. Called only during the loading screen 
to keep the Julia GC completely asleep during the physics loop.
"""
function init_svdag_arena()::SVDAGArena
    empty_node = SVDAGNode(0x00, 0x00, 0x0000, 0x00000000)
    
    return SVDAGArena(
        1, # Start with 1 active node (the root)
        fill(empty_node, MAX_SVDAG_NODES),
        fill(0x00000000, MAX_SVDAG_NODES)
    )
end

"""
    get_child_index(node::SVDAGNode, octant::Int)

Given a branch node and an octant (0-7), calculates the exact flat array index 
of the child node without branching logic.
"""
@inline function get_child_index(node::SVDAGNode, octant::Int)::UInt32
    # If the octant is empty, return 0 (Vacuum)
    if (node.child_mask & (0x01 << octant)) == 0
        return 0 
    end
    
    # Count the number of active children *before* the requested octant
    # to find the exact offset within the contiguous child block.
    mask_before = (0x01 << octant) - 1
    active_before = count_ones(node.child_mask & mask_before)
    
    return node.child_base_idx + active_before
end

"""
    compute_upward_pass!(svdag::SVDAGArena)

Executes an O(N) bottom-up traversal of the SVDAG. Calculates the total mass
and the weighted Center of Mass (CoM) for every branch node based on its children.
This functions identically to a geometric Merkle tree, rolling up physical state 
towards the root to feed the Barnes-Hut algorithm.
"""
function compute_upward_pass!(svdag::SVDAGArena)
    # Iterate backwards. Because nodes allocate children sequentially, 
    # a reverse iteration guarantees we process all children before their parent.
    @inbounds for i in svdag.active_node_count:-1:1
        node = svdag.nodes[i]
        
        # If it's a dense matter leaf node, its mass and CoM are already 
        # statically calculated from the raw material volumes. We skip it.
        if node.leaf_mask != 0x00
            continue
        end

        # For branch nodes, we aggregate from the octant children.
        total_mass = Int64(0)
        
        # We use 128-bit accumulators to prevent overflow during the weighted 
        # spatial multiplication, avoiding precision leaks.
        weighted_x = Int128(0)
        weighted_y = Int128(0)
        weighted_z = Int128(0)

        for octant in 0:7
            child_idx = get_child_index(node, octant)
            if child_idx != 0
                c_mass = svdag.node_mass[child_idx]
                total_mass += c_mass
                
                # CoM is an average weighted by mass
                weighted_x += Int128(c_mass) * svdag.com_x[child_idx]
                weighted_y += Int128(c_mass) * svdag.com_y[child_idx]
                weighted_z += Int128(c_mass) * svdag.com_z[child_idx]
            end
        end

        # Assign the aggregated mass to this branch
        svdag.node_mass[i] = total_mass
        
        # Divide by total mass to find the exact Center of Mass for the branch.
        # This gives us perfect fixed-point precision for Barnes-Hut.
        if total_mass > 0
            svdag.com_x[i] = Int64(weighted_x ÷ total_mass)
            svdag.com_y[i] = Int64(weighted_y ÷ total_mass)
            svdag.com_z[i] = Int64(weighted_z ÷ total_mass)
        else
            svdag.com_x[i] = 0
            svdag.com_y[i] = 0
            svdag.com_z[i] = 0
        end
    end
end
