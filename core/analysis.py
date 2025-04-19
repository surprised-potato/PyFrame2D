# core/analysis.py

import numpy as np
import math
from typing import TYPE_CHECKING, Optional # Avoid circular imports for type hinting

# Use TYPE_CHECKING block for imports needed only for type hints
if TYPE_CHECKING:
    from .model import StructuralModel, NodalLoad, MemberLoad, MemberPointLoad, MemberUDLoad

# Import necessary functions from model and elements
from .model import Load, NodalLoad, MemberLoad # Need these at runtime for isinstance
from .elements import local_stiffness_matrix, transformation_matrix, fixed_end_forces
from numpy.linalg import LinAlgError

def assemble_global_stiffness(model: 'StructuralModel') -> np.ndarray:
    """
    Assembles the global stiffness matrix [K] for the structural model.

    Iterates through each member, calculates its global stiffness matrix,
    and adds its contributions to the appropriate locations in the overall
    global matrix based on the model's DOF mapping.

    Args:
        model: The StructuralModel object containing nodes, members, etc.

    Returns:
        A NumPy array representing the global stiffness matrix [K]
        (size: num_active_dofs x num_active_dofs).

    Raises:
        RuntimeError: If the DOF map hasn't been generated or is invalid.
    """
    # Get DOF mapping information from the model
    try:
        dof_map, _, num_active_dofs = model.get_dof_map()
    except RuntimeError as e:
        raise RuntimeError(f"Could not assemble stiffness matrix: {e}") from e

    if num_active_dofs == 0:
        print("Warning: Model has no active degrees of freedom. Returning empty matrix.")
        return np.zeros((0, 0))

    # Initialize the global stiffness matrix with zeros
    K_global = np.zeros((num_active_dofs, num_active_dofs))

    # Define the order of local DOFs for mapping
    local_dof_indices = ['dx', 'dy', 'rz'] # Per node

    # Iterate through each member in the model
    for member in model.members.values():
        # Get member properties needed for stiffness calculation
        E = member.E
        A = member.A
        I = member.I
        L = member.length
        angle = member.angle

        # Calculate the 6x6 local stiffness matrix
        k_local = local_stiffness_matrix(E, A, I, L)

        # Calculate the 6x6 transformation matrix
        T = transformation_matrix(angle)

        # Calculate the 6x6 element stiffness matrix in global coordinates
        # k_global_elem = T.T @ k_local @ T
        # Efficient calculation: T is block diagonal, T.T = T^-1 for rotation part
        # Can optimize later if needed, use standard formula for clarity first.
        k_global_elem = T.T @ k_local @ T

        # Get the global DOF indices for this member's nodes
        n_i = member.start_node.id
        n_j = member.end_node.id
        node_ids = [n_i, n_j]
        global_indices = [] # Stores the 6 global indices for this member
        for node_id in node_ids:
            for dof_name in local_dof_indices:
                global_idx = dof_map.get((node_id, dof_name), -1) # Get index from map (-1 if constrained)
                global_indices.append(global_idx)

        # Add the element's global stiffness contributions to the main K matrix
        for r in range(6): # Row index in k_global_elem (0 to 5)
            glob_r = global_indices[r] # Corresponding global index
            if glob_r != -1: # Only consider active DOFs (not constrained)
                for c in range(6): # Column index in k_global_elem (0 to 5)
                    glob_c = global_indices[c] # Corresponding global index
                    if glob_c != -1: # Only consider active DOFs
                        K_global[glob_r, glob_c] += k_global_elem[r, c]

    return K_global


def assemble_global_loads(model: 'StructuralModel') -> np.ndarray:
    """
    Assembles the global load vector {F} for the structural model.

    Includes contributions from NodalLoads and the equivalent nodal forces
    derived from MemberLoads (calculated from Fixed-End Forces).

    Args:
        model: The StructuralModel object containing nodes, members, loads, etc.

    Returns:
        A NumPy array representing the global load vector {F}
        (size: num_active_dofs x 1).

    Raises:
        RuntimeError: If the DOF map hasn't been generated or is invalid.
        KeyError: If a member load references a member not found in the model
                  (should ideally be caught by model.validate earlier).
    """
    # Get DOF mapping information from the model
    try:
        dof_map, _, num_active_dofs = model.get_dof_map()
    except RuntimeError as e:
        raise RuntimeError(f"Could not assemble load vector: {e}") from e

    if num_active_dofs == 0:
        print("Warning: Model has no active degrees of freedom. Returning empty vector.")
        return np.zeros((0, 1))

    # Initialize the global load vector with zeros
    F_global = np.zeros((num_active_dofs, 1))

    # Define the order of local DOFs for mapping
    local_dof_indices = ['dx', 'dy', 'rz'] # Per node

    # Iterate through each load in the model
    for load in model.loads.values():
        if isinstance(load, NodalLoad):
            node_id = load.node_id
            load_components = {'dx': load.fx, 'dy': load.fy, 'rz': load.mz}

            for dof_name, load_value in load_components.items():
                if load_value != 0.0: # Only process non-zero loads
                    global_idx = dof_map.get((node_id, dof_name), -1)
                    if global_idx != -1: # Check if DOF is active
                        F_global[global_idx, 0] += load_value

        elif isinstance(load, MemberLoad):
            try:
                member = model.get_member(load.member_id)
            except KeyError:
                # This indicates an inconsistent model state if add_load checks were skipped
                print(f"Warning: Member {load.member_id} referenced by Load {load.id} not found. Skipping load.")
                continue

            L = member.length
            angle = member.angle

            try:
                # Calculate Fixed-End Forces in local coordinates
                fef_local = fixed_end_forces(load, L) # Returns 6x1 vector
            except (NotImplementedError, ValueError) as e:
                 print(f"Warning: Could not calculate FEF for Load {load.id} on Member {load.member_id}: {e}. Skipping load.")
                 continue

            # Transform local FEFs to global equivalent nodal forces
            T = transformation_matrix(angle)
            fef_global_equiv = T.T @ fef_local # 6x1 vector in global coords

            # Get the global DOF indices for this member's nodes
            n_i = member.start_node.id
            n_j = member.end_node.id
            node_ids = [n_i, n_j]
            global_indices = [] # Stores the 6 global indices for this member
            for node_id in node_ids:
                for dof_name in local_dof_indices:
                    global_idx = dof_map.get((node_id, dof_name), -1)
                    global_indices.append(global_idx)

            # Add the equivalent nodal forces to the global load vector F
            for r in range(6):
                glob_idx = global_indices[r]
                if glob_idx != -1: # Only add to active DOFs
                    F_global[glob_idx, 0] += fef_global_equiv[r]

        else:
             # Handle other potential Load base types if needed
             print(f"Warning: Load type {type(load)} not currently handled in load assembly. Skipping Load ID {load.id}.")

    return F_global

def solve_system(K_global_active: np.ndarray, F_global_active: np.ndarray) -> np.ndarray:
    """
    Solves the linear system of equations [K]{U} = {F} for the nodal displacements {U}.

    Uses the reduced global stiffness matrix and load vector corresponding
    only to the active (unconstrained) degrees of freedom.

    Args:
        K_global_active: The reduced global stiffness matrix (num_active_dofs x num_active_dofs).
        F_global_active: The reduced global load vector (num_active_dofs x 1).

    Returns:
        A NumPy array representing the displacement vector {U_active} for active DOFs
        (size: num_active_dofs x 1).

    Raises:
        LinAlgError: If the stiffness matrix K_global_active is singular (indicating
                     an unstable structure or mechanism).
        ValueError: If K and F dimensions are incompatible.
    """
    num_dofs = K_global_active.shape[0]
    if num_dofs == 0:
        # No active DOFs, displacement is empty
        return np.zeros((0, 1))

    if K_global_active.shape != (num_dofs, num_dofs):
        raise ValueError(f"Stiffness matrix K must be square ({num_dofs}x{num_dofs}), but got {K_global_active.shape}")
    if F_global_active.shape != (num_dofs, 1):
         raise ValueError(f"Load vector F must have shape ({num_dofs}x1), but got {F_global_active.shape}")

    try:
        # Use numpy's linear algebra solver
        U_active = np.linalg.solve(K_global_active, F_global_active)
        return U_active
    except LinAlgError as e:
        # Re-raise with a more informative message (or handle differently)
        raise LinAlgError("The global stiffness matrix is singular. "
                          "The structure may be unstable or contain a mechanism.") from e


def reconstruct_full_displacement(model: 'StructuralModel', U_active: np.ndarray) -> np.ndarray:
    """
    Reconstructs the full global displacement vector, including zeros for constrained DOFs.

    Args:
        model: The StructuralModel object containing the DOF map.
        U_active: The displacement vector containing only the active DOFs,
                  as returned by solve_system (num_active_dofs x 1).

    Returns:
        A NumPy array representing the full displacement vector {U_full}
        (size: total_num_dofs x 1). The order corresponds to iterating through
        sorted node IDs, taking DOFs [dx, dy, rz] for each node.

    Raises:
        RuntimeError: If the DOF map hasn't been generated or is invalid.
        ValueError: If the size of U_active does not match the number of active DOFs
                    expected by the model's DOF map.
    """
    try:
        dof_map, _, num_active_dofs_map = model.get_dof_map()
    except RuntimeError as e:
        raise RuntimeError(f"Could not reconstruct displacements: {e}") from e

    if U_active.shape != (num_active_dofs_map, 1):
        raise ValueError(f"Input displacement vector U_active shape {U_active.shape} "
                         f"does not match expected active DOFs ({num_active_dofs_map}, 1).")

    num_nodes = len(model.nodes)
    total_dofs = num_nodes * 3 # dx, dy, rz per node
    U_full = np.zeros((total_dofs, 1))

    # Create mapping from node ID to a sequential index (0 to num_nodes-1)
    sorted_node_ids = sorted(model.nodes.keys())
    node_id_to_seq_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}

    # Map active displacements back to the full vector
    for (node_id, dof_name), active_dof_index in dof_map.items():
        if active_dof_index >= 0: # If it's an active DOF index
            try:
                node_seq_idx = node_id_to_seq_idx[node_id]
            except KeyError:
                # Should not happen if dof_map is consistent with model.nodes
                print(f"Warning: Node ID {node_id} found in DOF map but not in sorted node list. Skipping.")
                continue

            # Determine the index offset for the DOF type
            if dof_name == 'dx':
                dof_offset = 0
            elif dof_name == 'dy':
                dof_offset = 1
            elif dof_name == 'rz':
                dof_offset = 2
            else:
                # Should not happen
                print(f"Warning: Unknown DOF name '{dof_name}' in DOF map. Skipping.")
                continue

            # Calculate the index in the full displacement vector
            full_vector_index = node_seq_idx * 3 + dof_offset

            # Assign the calculated displacement
            U_full[full_vector_index, 0] = U_active[active_dof_index, 0]
        # else: DOF is constrained (index is -1), U_full entry remains 0.0

    return U_full


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Use the same portal frame example from previous step
    from .model import StructuralModel, Node, Material, RectangularProfile, Member, Support, NodalLoad, MemberUDLoad

    print("--- Solving the simple portal frame example ---")
    model = StructuralModel(name="Portal Frame")
    # ... (Recreate the portal frame model as in the previous __main__ block) ...
    H = 4.0; W = 6.0; E = 200e9; A_col=0.004; I_col=1.333e-5; A_beam=0.045; I_beam=3.375e-4 # More realistic Section values needed
    n1 = Node(1, 0, 0); n2 = Node(2, 0, H); n3 = Node(3, W, H); n4 = Node(4, W, 0)
    model.add_node(n1); model.add_node(n2); model.add_node(n3); model.add_node(n4)
    mat = Material(1, "Steel", E)
    # Using MockSection for simplicity here, assuming desired A/I
    sec_col = MockSection(102,"Col",A_col,I_col) # Need MockSection defined above or imported
    sec_beam = MockSection(101,"Beam",A_beam,I_beam)
    model.add_material(mat); model.add_section(sec_beam); model.add_section(sec_col)
    mem1 = Member(1, n1, n2, mat, sec_col); mem2 = Member(2, n2, n3, mat, sec_beam); mem3 = Member(3, n4, n3, mat, sec_col)
    model.add_member(mem1); model.add_member(mem2); model.add_member(mem3)
    model.add_support(Support.fixed(n1.id)); model.add_support(Support.fixed(n4.id))
    load1 = MemberUDLoad(id=1, member_id=mem2.id, wy=-10e3)
    load2 = NodalLoad(id=2, node_id=n2.id, fx=50e3)
    model.add_load(load1); model.add_load(load2)
    # --- End model recreation ---

    try:
        # 1. Assemble
        K = assemble_global_stiffness(model)
        F = assemble_global_loads(model)
        print(f"Assembled K ({K.shape}), F ({F.shape})")

        # 2. Solve
        U_active = solve_system(K, F)
        print(f"\nSolved Active Displacements U_active ({U_active.shape}):")
        print(U_active) # Print the raw active displacements

        # 3. Reconstruct Full Displacements
        U_full = reconstruct_full_displacement(model, U_active)
        print(f"\nReconstructed Full Displacements U_full ({U_full.shape}):")
        # print(np.round(U_full, 6)) # Print rounded full displacements

        # Print displacements node by node for clarity
        print("\nNodal Displacements (dx, dy, rz):")
        dof_map, _, _ = model.get_dof_map()
        sorted_node_ids = sorted(model.nodes.keys())
        node_id_to_seq_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}

        for node_id in sorted_node_ids:
            seq_idx = node_id_to_seq_idx[node_id]
            dx = U_full[seq_idx * 3 + 0, 0]
            dy = U_full[seq_idx * 3 + 1, 0]
            rz = U_full[seq_idx * 3 + 2, 0]
            print(f"  Node {node_id}: dx={dx:.4e} m, dy={dy:.4e} m, rz={math.degrees(rz):.4f} deg")


    except LinAlgError as e:
        print(f"\nError solving system: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
