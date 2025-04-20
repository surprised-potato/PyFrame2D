# core/analysis.py

import numpy as np
import math
from typing import TYPE_CHECKING, Optional # Avoid circular imports for type hinting
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Use TYPE_CHECKING block for imports needed only for type hints
if TYPE_CHECKING:
    from .model import StructuralModel, NodalLoad, MemberLoad, MemberPointLoad, MemberUDLoad

# Import necessary functions from model and elements
from .model import (StructuralModel, Load, NodalLoad, MemberLoad,
                    MemberPointLoad, MemberUDLoad)

from .elements import local_stiffness_matrix, transformation_matrix, fixed_end_forces
from numpy.linalg import LinAlgError

@dataclass
class AnalysisResults:
    """
    Stores the results of a structural analysis.

    Attributes:
        status (str): Analysis completion status ("Success", "Singular Matrix", "Error").
        message (Optional[str]): Additional information or error message.
        model_name (str): Name of the analyzed model.
        nodal_displacements (Optional[Dict[int, Tuple[float, float, float]]]):
            Dictionary mapping node ID to its global displacements (dx, dy, rz).
            Units: meters for dx/dy, radians for rz. None if analysis failed.
        support_reactions (Optional[Dict[int, Tuple[float, float, float]]]):
            Dictionary mapping supported node ID to global reaction forces/moment (Rx, Ry, Mz).
            Units: Newtons for Rx/Ry, Newton-meters for Mz. None if analysis failed.
        member_end_forces (Optional[Dict[int, np.ndarray]]):
            Dictionary mapping member ID to its 6x1 local end force vector
            [Px_i, Py_i, Mz_i, Px_j, Py_j, Mz_j].
            Units: N, N, Nm, N, N, Nm. None if analysis failed.
        member_afd_data (Optional[Dict[int, np.ndarray]]):
            Axial Force Diagram data. Maps member ID to Nx2 array [x, P(x)].
            Units: m, N. None if not calculated or analysis failed.
        member_sfd_data (Optional[Dict[int, np.ndarray]]):
            Shear Force Diagram data. Maps member ID to Nx2 array [x, V(x)].
            Units: m, N. None if not calculated or analysis failed.
        member_bmd_data (Optional[Dict[int, np.ndarray]]):
            Bending Moment Diagram data. Maps member ID to Nx2 array [x, M(x)].
            Units: m, Nm. None if not calculated or analysis failed.
    """
    status: str
    message: Optional[str] = None
    model_name: str = "Untitled"
    nodal_displacements: Optional[Dict[int, Tuple[float, float, float]]] = None
    support_reactions: Optional[Dict[int, Tuple[float, float, float]]] = None
    member_end_forces: Optional[Dict[int, np.ndarray]] = None
    member_afd_data: Optional[Dict[int, np.ndarray]] = field(default=None, repr=False) # Don't include bulky data in repr
    member_sfd_data: Optional[Dict[int, np.ndarray]] = field(default=None, repr=False)
    member_bmd_data: Optional[Dict[int, np.ndarray]] = field(default=None, repr=False)

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


def calculate_member_forces(model: 'StructuralModel', U_full: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Calculates the local end forces for each member after analysis.

    Member end forces = [k_local]{u_local} - {fef_local_restraining}

    Args:
        model: The StructuralModel object.
        U_full: The full global displacement vector (including constrained DOFs).

    Returns:
        A dictionary mapping member ID to its 6x1 local end force vector
        [Px_i, Py_i, Mz_i, Px_j, Py_j, Mz_j].

    Raises:
        RuntimeError: If DOF map is invalid.
        KeyError: If a member references a node not found in the model's sorted list
                  (indicates inconsistency).
    """
    member_end_forces_dict: Dict[int, np.ndarray] = {}

    try:
        # Needed for mapping global U_full indices to nodes
        sorted_node_ids = sorted(model.nodes.keys())
        node_id_to_seq_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}
    except Exception as e:
         raise RuntimeError(f"Error creating node index mapping: {e}")

    local_dof_indices = ['dx', 'dy', 'rz']

    # Get all loads applied to each member
    loads_on_member: Dict[int, List[MemberLoad]] = {}
    for load in model.loads.values():
        if isinstance(load, MemberLoad):
            if load.member_id not in loads_on_member:
                loads_on_member[load.member_id] = []
            loads_on_member[load.member_id].append(load)

    # Iterate through members
    for mem_id, member in model.members.items():
        # 1. Get Member Properties
        E, A, I, L = member.E, member.A, member.I, member.length
        angle = member.angle

        # 2. Extract Global Displacements for Member Nodes
        u_global_elem = np.zeros((6, 1))
        node_indices = [member.start_node.id, member.end_node.id]
        for i, node_id in enumerate(node_indices): # i=0 for start, i=1 for end
            try:
                node_seq_idx = node_id_to_seq_idx[node_id]
            except KeyError:
                 raise KeyError(f"Node {node_id} for Member {mem_id} not found in model's sorted node list.")

            for j, dof_name in enumerate(local_dof_indices): # j=0 for dx, 1 for dy, 2 for rz
                full_vector_index = node_seq_idx * 3 + j
                u_global_elem[i * 3 + j, 0] = U_full[full_vector_index, 0]

        # 3. Transform Global Displacements to Local
        T = transformation_matrix(angle)
        u_local_elem = T @ u_global_elem # 6x1 local displacements

        # 4. Calculate Local Forces due to Displacements
        k_local = local_stiffness_matrix(E, A, I, L)
        f_disp_local = k_local @ u_local_elem # 6x1 local forces from displacements

        # 5. Calculate Total Fixed-End Forces (Restraining) for this Member
        fef_local_total_restraining = np.zeros(6)
        member_loads = loads_on_member.get(mem_id, [])
        if member_loads:
            for load in member_loads:
                try:
                    fef_local_single = fixed_end_forces(load, L) # Get restraining FEFs
                    fef_local_total_restraining += fef_local_single
                except (NotImplementedError, ValueError) as e:
                     print(f"Warning: Could not calculate FEF for Load {load.id} on Member {mem_id} during force calculation: {e}. Skipping load's contribution.")

        # 6. Calculate Final Member End Forces
        # End forces = forces from displacements - restraining forces from loads
        f_total_local = f_disp_local.flatten() - fef_local_total_restraining # Use flatten for broadcasting if needed, ensure shape is consistent (6,) or (6,1)
        member_end_forces_dict[mem_id] = f_total_local.reshape(6, 1) # Ensure 6x1 shape

    return member_end_forces_dict


def calculate_reactions(model: 'StructuralModel', member_end_forces_dict: Dict[int, np.ndarray]) -> Dict[int, Tuple[float, float, float]]:
    """
    Calculates support reactions based on member end forces and external nodal loads.

    Reaction = Sum(Transformed Member End Forces at Node) - External Nodal Load at Node

    Args:
        model: The StructuralModel object.
        member_end_forces_dict: Dictionary mapping member ID to 6x1 local end force vector.

    Returns:
        A dictionary mapping supported node ID to global reactions (Rx, Ry, Mz).

    Raises:
        RuntimeError: If DOF map is invalid.
        KeyError: If issues arise mapping DOFs/nodes.
    """
    reactions: Dict[int, np.ndarray] = {} # Store reactions as Rx, Ry, Mz arrays

    try:
        dof_map, constrained_dofs, _ = model.get_dof_map()
    except RuntimeError as e:
        raise RuntimeError(f"Could not calculate reactions: {e}") from e

    # Identify nodes with constraints
    constrained_node_ids = {node_id for node_id, dof_name in constrained_dofs}
    if not constrained_node_ids:
        return {} # No supports, no reactions

    # Initialize reaction vectors for supported nodes
    for node_id in constrained_node_ids:
        reactions[node_id] = np.zeros(3) # Rx, Ry, Mz order

    local_dof_indices = ['dx', 'dy', 'rz']

    # 1. Sum contributions from member end forces
    for mem_id, member in model.members.items():
        angle = member.angle
        T = transformation_matrix(angle)
        f_local = member_end_forces_dict.get(mem_id)

        if f_local is None:
            print(f"Warning: End forces for Member {mem_id} not found during reaction calculation. Skipping.")
            continue
        if f_local.shape != (6, 1):
             print(f"Warning: End forces for Member {mem_id} have unexpected shape {f_local.shape}. Skipping.")
             continue

        # Transform local end forces to global
        f_global = T.T @ f_local # 6x1 global end forces

        # Add contributions to nodes
        node_ids = [member.start_node.id, member.end_node.id]
        for i, node_id in enumerate(node_ids): # i=0 for start, i=1 for end
            if node_id in reactions:
                # Extract the 3 global forces/moment for this node end from f_global
                node_forces_global = f_global[i * 3 : (i + 1) * 3].flatten() # Fx, Fy, Mz
                reactions[node_id] -= node_forces_global

    # 2. Subtract external nodal loads applied at supported nodes
    for load in model.loads.values():
        if isinstance(load, NodalLoad):
            node_id = load.node_id
            if node_id in reactions:
                 # Subtract applied loads from the sum of internal forces
                 reactions[node_id][0] -= load.fx # Rx contribution
                 reactions[node_id][1] -= load.fy # Ry contribution
                 reactions[node_id][2] -= load.mz # Mz contribution

    # 3. Filter results to only include constrained DOFs' reactions & format output
    final_reactions: Dict[int, Tuple[float, float, float]] = {}
    for node_id, r_vec in reactions.items():
        rx, ry, mz = 0.0, 0.0, 0.0
        if (node_id, 'dx') in constrained_dofs: rx = r_vec[0]
        if (node_id, 'dy') in constrained_dofs: ry = r_vec[1]
        if (node_id, 'rz') in constrained_dofs: mz = r_vec[2]
        # Only add node to final dict if at least one reaction is non-zero (or if support exists)
        support = model.get_support(node_id)
        if support and (support.dx or support.dy or support.rz):
             final_reactions[node_id] = (rx, ry, mz)

    return final_reactions


def calculate_diagram_data(model: 'StructuralModel',
                           member_end_forces_dict: Dict[int, np.ndarray],
                           num_points: int = 11) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Calculates Axial Force (AFD), Shear Force (SFD), and Bending Moment (BMD)
    diagram data points for each member.

    Uses FEA sign conventions:
    - P(x): Positive for tension.
    - V(x): Positive shear corresponds to positive face rotation (upward on right face of segment).
    - M(x): Positive moment causes tension on the bottom fiber (sagging, CCW moment on right face).

    Args:
        model: The StructuralModel object.
        member_end_forces_dict: Dictionary mapping member ID to 6x1 local end force vector.
        num_points: The number of points along each member to calculate diagram data
                    (including start and end points). Must be >= 2.

    Returns:
        A tuple containing three dictionaries (afd_data, sfd_data, bmd_data):
        - afd_data: Maps member ID to Nx2 NumPy array [x, P(x)].
        - sfd_data: Maps member ID to Nx2 NumPy array [x, V(x)].
        - bmd_data: Maps member ID to Nx2 NumPy array [x, M(x)].
    """
    if num_points < 2:
        print("Warning: num_points for diagrams must be at least 2. Setting to 2.")
        num_points = 2

    afd_data_dict: Dict[int, np.ndarray] = {}
    sfd_data_dict: Dict[int, np.ndarray] = {}
    bmd_data_dict: Dict[int, np.ndarray] = {}

    # Get all loads applied to each member
    loads_on_member: Dict[int, List[MemberLoad]] = {}
    for load in model.loads.values():
        if isinstance(load, MemberLoad):
            member_id = load.member_id
            if member_id not in loads_on_member:
                loads_on_member[member_id] = []
            loads_on_member[member_id].append(load)

    # Iterate through members
    for mem_id, member in model.members.items():
        L = member.length
        f_local = member_end_forces_dict.get(mem_id)

        if f_local is None:
            print(f"Warning: End forces for Member {mem_id} not found during diagram calculation. Skipping.")
            continue
        if f_local.shape != (6, 1):
             print(f"Warning: End forces for Member {mem_id} have unexpected shape {f_local.shape}. Skipping.")
             continue

        # Extract local end forces (FEA convention: acting on member ends)
        Px_i, Py_i, Mz_i = f_local[0,0], f_local[1,0], f_local[2,0]
        Px_j, Py_j, Mz_j = f_local[3,0], f_local[4,0], f_local[5,0] # Note: Px_j is force at end j

        # Get loads specific to this member
        member_loads = loads_on_member.get(mem_id, [])
        point_loads = [load for load in member_loads if isinstance(load, MemberPointLoad)]
        udl_loads = [load for load in member_loads if isinstance(load, MemberUDLoad)]
        # Sort point loads by position
        point_loads.sort(key=lambda ld: ld.position)

        # Calculate diagram values at discrete points
        x_coords = np.linspace(0, L, num_points)
        P_vals = np.zeros(num_points)
        V_vals = np.zeros(num_points)
        M_vals = np.zeros(num_points)

        for i, x in enumerate(x_coords):
            # --- Axial Force P(x) ---
            # P(x) = -Px_i - sum(wx*x) - sum(Px @ pos < x)
            p_val = -Px_i # Start with reaction force at i (-ve of force in member)
            for load in udl_loads:
                 p_val -= load.wx * x # Effect of axial UDL
            for pload in point_loads:
                if pload.position <= x + 1e-9: # Include load if at or before x
                    p_val -= pload.px
            P_vals[i] = p_val

            # --- Shear Force V(x) ---
            # V(x) = -Py_i - sum(wy*x) - sum(Py @ pos < x)
            v_val = -Py_i # Start with reaction force at i (-ve of force in member)
            for load in udl_loads:
                 v_val -= load.wy * x # Effect of transverse UDL
            for pload in point_loads:
                 if pload.position <= x + 1e-9:
                     v_val -= pload.py
            V_vals[i] = v_val

            # --- Bending Moment M(x) ---
            # M(x) = -Mz_i - Py_i*x - sum(wy*x*x/2) - sum(Py*(x-pos) @ pos < x)
            m_val = -Mz_i # Start with reaction moment at i (-ve of moment in member)
            m_val -= Py_i * x # Moment from shear reaction at i
            for load in udl_loads:
                 m_val -= load.wy * x**2 / 2.0 # Moment from transverse UDL
            for pload in point_loads:
                 if pload.position <= x + 1e-9:
                     m_val -= pload.py * (x - pload.position) # Moment from point load
            M_vals[i] = m_val

        # Store data for this member
        afd_data_dict[mem_id] = np.vstack((x_coords, P_vals)).T # Shape (num_points, 2)
        sfd_data_dict[mem_id] = np.vstack((x_coords, V_vals)).T
        bmd_data_dict[mem_id] = np.vstack((x_coords, M_vals)).T

    return afd_data_dict, sfd_data_dict, bmd_data_dict


# --- Main Analysis Runner Function ---

def analyze(model: 'StructuralModel', num_diagram_points: int = 11) -> AnalysisResults:
    """
    Performs the full structural analysis pipeline.

    1. Assembles global stiffness matrix [K] and load vector {F}.
    2. Solves the system [K]{U} = {F} for active nodal displacements {U_active}.
    3. Reconstructs the full displacement vector {U_full}.
    4. Calculates member end forces.
    5. Calculates support reactions.
    6. Calculates diagram data points (optional).
    7. Packages results into an AnalysisResults object.

    Args:
        model: The StructuralModel to analyze.
        num_diagram_points: Number of points for calculating diagram data (>=2).
                            Set to 0 or None to skip diagram calculations.

    Returns:
        An AnalysisResults object containing the results or error status.
    """
    results = AnalysisResults(status="Pending", model_name=model.name)
    U_full = None
    member_forces = None
    reactions = None
    afd = None
    sfd = None
    bmd = None

    try:
        # 1. Assemble
        print("Assembling global matrices...")
        K_active = assemble_global_stiffness(model)
        F_active = assemble_global_loads(model)
        print(f"Assembly complete. K_active shape: {K_active.shape}, F_active shape: {F_active.shape}")

        # 2. Solve
        print("Solving system...")
        U_active = solve_system(K_active, F_active)
        print("System solved.")

        # 3. Reconstruct Full Displacements
        print("Reconstructing full displacement vector...")
        U_full = reconstruct_full_displacement(model, U_active)
        print("Reconstruction complete.")

        # Prepare nodal displacements for results object
        nodal_disp_dict: Dict[int, Tuple[float, float, float]] = {}
        sorted_node_ids = sorted(model.nodes.keys())
        node_id_to_seq_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}
        for node_id in sorted_node_ids:
            seq_idx = node_id_to_seq_idx[node_id]
            dx = U_full[seq_idx * 3 + 0, 0]
            dy = U_full[seq_idx * 3 + 1, 0]
            rz = U_full[seq_idx * 3 + 2, 0]
            nodal_disp_dict[node_id] = (dx, dy, rz)

        # 4. Calculate Member Forces
        print("Calculating member forces...")
        member_forces = calculate_member_forces(model, U_full)
        print("Member forces calculated.")

        # 5. Calculate Reactions
        print("Calculating reactions...")
        reactions = calculate_reactions(model, member_forces)
        print("Reactions calculated.")

        # 6. Calculate Diagram Data (Optional)
        if num_diagram_points is not None and num_diagram_points >= 2:
            print(f"Calculating diagram data ({num_diagram_points} points)...")
            afd, sfd, bmd = calculate_diagram_data(model, member_forces, num_diagram_points)
            print("Diagram data calculated.")
        else:
            print("Skipping diagram data calculation.")


        # 7. Package Results
        results.status = "Success"
        results.nodal_displacements = nodal_disp_dict
        results.support_reactions = reactions
        results.member_end_forces = member_forces
        results.member_afd_data = afd
        results.member_sfd_data = sfd
        results.member_bmd_data = bmd

    except LinAlgError as e:
        results.status = "Singular Matrix"
        results.message = str(e)
        print(f"Analysis failed: {results.status} - {results.message}")
    except (RuntimeError, KeyError, ValueError, TypeError) as e:
        results.status = "Error"
        results.message = f"An unexpected error occurred during analysis: {e}"
        print(f"Analysis failed: {results.status} - {results.message}")
        import traceback
        traceback.print_exc() # Print traceback for debugging
    except Exception as e:
        results.status = "Error"
        results.message = f"An unknown error occurred: {e}"
        print(f"Analysis failed: {results.status} - {results.message}")
        import traceback
        traceback.print_exc()

    return results


# --- Example Usage (Optional - Now runs full pipeline) ---
if __name__ == "__main__":
    # Use the same portal frame example from previous step
    from .model import StructuralModel, Node, Material, Member, Support, NodalLoad, MemberUDLoad
    # Need MockSection or a real section like RectangularProfile
    from .model import RectangularProfile # Use real section

    print("--- Running Full Analysis Pipeline for Portal Frame Example ---")
    model = StructuralModel(name="Portal Frame")
    # ... (Recreate the portal frame model using RectangularProfile as before) ...
    H = 4.0; W = 6.0; E = 200e9
    n1 = Node(1, 0, 0); n2 = Node(2, 0, H); n3 = Node(3, W, H); n4 = Node(4, W, 0)
    model.add_node(n1); model.add_node(n2); model.add_node(n3); model.add_node(n4)
    mat = Material(1, "Steel", E)
    model.add_material(mat)
    sec_col = RectangularProfile(id=102, name="Col 200x200", width=0.2, height=0.2)
    sec_beam = RectangularProfile(id=101, name="Beam 300x150", width=0.15, height=0.3)
    model.add_section(sec_beam); model.add_section(sec_col)
    mem1 = Member(1, n1, n2, mat, sec_col); mem2 = Member(2, n2, n3, mat, sec_beam); mem3 = Member(3, n4, n3, mat, sec_col)
    model.add_member(mem1); model.add_member(mem2); model.add_member(mem3)
    model.add_support(Support.fixed(n1.id)); model.add_support(Support.fixed(n4.id))
    load1 = MemberUDLoad(id=1, member_id=mem2.id, wy=-10e3)
    load2 = NodalLoad(id=2, node_id=n2.id, fx=50e3)
    model.add_load(load1); model.add_load(load2)
    # --- End model recreation ---

    # Run the analysis
    analysis_results = analyze(model, num_diagram_points=5) # Calculate diagrams with 5 points

    # Print summary of results
    print("\n--- Analysis Results ---")
    print(f"Status: {analysis_results.status}")
    if analysis_results.message:
        print(f"Message: {analysis_results.message}")

    if analysis_results.status == "Success":
        print("\nNodal Displacements (dx, dy, rz):")
        if analysis_results.nodal_displacements:
            for node_id, disp in analysis_results.nodal_displacements.items():
                print(f"  Node {node_id}: dx={disp[0]:.4e} m, dy={disp[1]:.4e} m, rz={math.degrees(disp[2]):.4f} deg")

        print("\nSupport Reactions (Rx, Ry, Mz):")
        if analysis_results.support_reactions:
            for node_id, reac in analysis_results.support_reactions.items():
                 print(f"  Node {node_id}: Rx={reac[0]/1000:.2f} kN, Ry={reac[1]/1000:.2f} kN, Mz={reac[2]/1000:.2f} kNm")

        print("\nMember End Forces (Local: Px_i, Py_i, Mz_i, Px_j, Py_j, Mz_j):")
        if analysis_results.member_end_forces:
            for mem_id, forces in analysis_results.member_end_forces.items():
                force_str = ", ".join([f"{f:.2f}" for f in forces.flatten()])
                print(f"  Member {mem_id}: [{force_str}]")

        if analysis_results.member_bmd_data:
             print("\nSample BMD Data Points (Member 2):")
             print(analysis_results.member_bmd_data.get(2)) # Print BMD data for member 2
