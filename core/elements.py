# core/elements.py

import numpy as np
import math

# Import load types for Fixed-End Forces calculation
# Use a try-except block for robustness, especially if elements.py might
# sometimes be used independently or if circular imports become an issue later.
try:
    from .model import MemberLoad, MemberPointLoad, MemberUDLoad
except ImportError:
    # Define dummy classes if model import fails, allowing file parsing
    # but functions using these types will fail at runtime if not available.
    class MemberLoad: pass
    class MemberPointLoad(MemberLoad): pass
    class MemberUDLoad(MemberLoad): pass
    print("Warning: Could not import Model classes (MemberLoad types) into elements.py.")


def local_stiffness_matrix(E: float, A: float, I: float, L: float) -> np.ndarray:
    """
    Calculates the local 6x6 stiffness matrix for a 2D Euler-Bernoulli beam element.

    Assumes standard engineering sign conventions.
    Local DOFs are ordered: [u_i, v_i, rz_i, u_j, v_j, rz_j]
    where u=axial, v=transverse (shear), rz=rotation; i=start node, j=end node.

    Args:
        E: Young's Modulus of the element material (Pascals).
        A: Cross-sectional area of the element (meters^2).
        I: Second moment of area (moment of inertia) about the bending axis (meters^4).
        L: Length of the element (meters).

    Returns:
        A 6x6 NumPy array representing the local stiffness matrix (k_local).

    Raises:
        ValueError: If L is zero or negative.
    """
    if L <= 0:
        raise ValueError("Element length (L) must be positive.")
    if E < 0 or A < 0 or I < 0:
        # Allow zero A or I for truss elements, perhaps? Or validate upstream.
        # For now, just ensure non-negative.
        print(f"Warning: E={E}, A={A}, or I={I} is negative. Results may be nonsensical.")

    # Pre-calculate common terms for efficiency and readability
    EA_L = E * A / L if L != 0 else 0 # Avoid division by zero conceptually, though ValueError raised
    EI_L = E * I / L if L != 0 else 0
    EI_L2 = EI_L / L if L != 0 else 0
    EI_L3 = EI_L2 / L if L != 0 else 0

    # Initialize 6x6 matrix
    k_local = np.zeros((6, 6))

    # Populate the matrix based on Euler-Bernoulli beam theory
    # Axial terms (Row 0 and 3)
    k_local[0, 0] = EA_L
    k_local[0, 3] = -EA_L
    k_local[3, 0] = -EA_L
    k_local[3, 3] = EA_L

    # Bending and Shear terms (Rows 1, 2, 4, 5)
    k_local[1, 1] = 12 * EI_L3
    k_local[1, 2] = 6 * EI_L2
    k_local[1, 4] = -12 * EI_L3
    k_local[1, 5] = 6 * EI_L2

    k_local[2, 1] = 6 * EI_L2
    k_local[2, 2] = 4 * EI_L
    k_local[2, 4] = -6 * EI_L2
    k_local[2, 5] = 2 * EI_L

    k_local[4, 1] = -12 * EI_L3
    k_local[4, 2] = -6 * EI_L2
    k_local[4, 4] = 12 * EI_L3
    k_local[4, 5] = -6 * EI_L2

    k_local[5, 1] = 6 * EI_L2
    k_local[5, 2] = 2 * EI_L
    k_local[5, 4] = -6 * EI_L2
    k_local[5, 5] = 4 * EI_L

    return k_local


def transformation_matrix(angle: float) -> np.ndarray:
    """
    Calculates the 6x6 transformation matrix (T) for a 2D frame element.

    This matrix transforms quantities (displacements, forces, stiffness matrices)
    from local element coordinates to global coordinates.
    T = [[lambda, 0], [0, lambda]]
    where lambda is the 3x3 rotation matrix for node DOFs [u, v, rz].

    Args:
        angle: The angle of the element's local x-axis relative to the global
               positive X-axis, measured counter-clockwise (in radians).

    Returns:
        A 6x6 NumPy array representing the transformation matrix (T).
    """
    c = math.cos(angle)
    s = math.sin(angle)

    # 3x3 rotation matrix (lambda)
    lam = np.array([
        [ c,  s,  0],
        [-s,  c,  0],
        [ 0,  0,  1]
    ])

    # Assemble the 6x6 transformation matrix T
    T = np.zeros((6, 6))
    T[0:3, 0:3] = lam
    T[3:6, 3:6] = lam

    return T


def fixed_end_forces(load: MemberLoad, L: float) -> np.ndarray:
    """
    Calculates the fixed-end force vector for a member subjected to a given load.

    Returns the forces/moments acting *on the nodes* by the fixed supports,
    consistent with the local DOF order: [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j].
    Uses standard engineering sign conventions (e.g., positive moment is CCW).

    Currently supports:
    - MemberPointLoad (perpendicular component py only)
    - MemberUDLoad (perpendicular component wy only)

    Axial components (px, wx) currently do not contribute to FEFs in this implementation.

    Args:
        load: A MemberLoad object (MemberPointLoad or MemberUDLoad).
        L: The length of the member (meters).

    Returns:
        A 6x1 NumPy array representing the fixed-end force vector in local coordinates.

    Raises:
        ValueError: If L is zero or negative, or if load position is invalid.
        TypeError: If load is not a supported MemberLoad type.
        NotImplementedError: If load type is recognized but not implemented.
    """
    if L <= 0:
        raise ValueError("Element length (L) must be positive.")
    if not isinstance(load, MemberLoad):
        raise TypeError(f"Input 'load' must be a MemberLoad object, not {type(load)}.")

    fef = np.zeros(6)

    if isinstance(load, MemberPointLoad):
        # Perpendicular force Py at distance 'a' from start node 'i'
        P = load.py # Magnitude of perpendicular force
        a = load.position
        if not (0 <= a <= L):
             raise ValueError(f"MemberPointLoad position {a} is outside member length {L}.")

        b = L - a

        # Formulas for perpendicular point load P (positive if acting in local +y)
        # Reactions on the nodes (forces/moments exerted *by* the support *onto* the beam)
        # Shear_i = P * b**2 * (L + 2*a) / L**3  (Fy_i reaction)
        # Moment_i = P * a * b**2 / L**2        (Mz_i reaction)
        # Shear_j = P * a**2 * (L + 2*b) / L**3  (Fy_j reaction)
        # Moment_j = -P * b * a**2 / L**2       (Mz_j reaction)

        # Fixed-end forces are equal and opposite to these reactions
        fef[1] = - (P * b**2 * (L + 2*a)) / (L**3) # Fy_i
        fef[2] = - (P * a * b**2) / (L**2)        # Mz_i
        fef[4] = - (P * a**2 * (L + 2*b)) / (L**3) # Fy_j
        fef[5] = + (P * b * a**2) / (L**2)        # Mz_j (Note sign change from reaction)

        # Handle axial component Px (optional, simple distribution)
        Px = load.px
        if Px != 0:
            fef[0] = - (Px * b / L) # Fx_i
            fef[3] = - (Px * a / L) # Fx_j

    elif isinstance(load, MemberUDLoad):
        # Uniformly distributed load 'w' (perpendicular)
        w = load.wy # Magnitude of perpendicular UDL (positive in local +y)

        # Formulas for perpendicular UDL w
        # Shear_i = w * L / 2
        # Moment_i = w * L**2 / 12
        # Shear_j = w * L / 2
        # Moment_j = -w * L**2 / 12

        fef[1] = - (w * L / 2.0)      # Fy_i
        fef[2] = - (w * L**2 / 12.0) # Mz_i
        fef[4] = - (w * L / 2.0)      # Fy_j
        fef[5] = + (w * L**2 / 12.0) # Mz_j

        # Handle axial component wx (optional)
        wx = load.wx
        if wx != 0:
            fef[0] = - (wx * L / 2.0) # Fx_i
            fef[3] = - (wx * L / 2.0) # Fx_j

    else:
        # Handle other potential MemberLoad types later
        raise NotImplementedError(f"Fixed-end force calculation not implemented for load type: {type(load)}")

    return fef


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Example parameters
    E = 210e9  # Pa
    A = 0.005  # m^2
    I = 3e-5   # m^4
    L = 5.0    # m
    angle_deg = 30
    angle_rad = math.radians(angle_deg)

    # Calculate local stiffness matrix
    k_local = local_stiffness_matrix(E, A, I, L)
    print(f"--- Local Stiffness Matrix (k_local) for L={L}m ---")
    print(np.round(k_local, 1)) # Round for display

    # Calculate transformation matrix
    T = transformation_matrix(angle_rad)
    print(f"\n--- Transformation Matrix (T) for angle={angle_deg} deg ---")
    print(np.round(T, 3))

    # Calculate global stiffness matrix (k_global = T.T @ k_local @ T)
    k_global = T.T @ k_local @ T
    print(f"\n--- Global Stiffness Matrix (k_global) ---")
    print(np.round(k_global, 1))

    # Calculate FEFs for a point load
    try:
        pt_load = MemberPointLoad(id=1, member_id=1, px=0, py=-10000, position=L/2) # 10kN downward at mid-span
        fef_pt = fixed_end_forces(pt_load, L)
        print(f"\n--- FEFs for Point Load {pt_load.id} (py={pt_load.py} N @ {pt_load.position} m) ---")
        print(np.round(fef_pt, 2))
        # Expected for mid-span point load P: Fy = -P/2, Mz = -PL/8 and +PL/8
        # Fy = -(-10000)/2 = 5000
        # Mz = -(-10000)*5/8 = 6250 at start, +(-10000)*5/8 = -6250 at end
        # Corrected FEF signs: Fy = +P/2, Mz = +PL/8 and -PL/8 (opposite of reactions)
        # FEF_Fy = -(-10000)/2 = +5000 N
        # FEF_Mz_i = -(-10000)*5/8 = +6250 Nm
        # FEF_Mz_j = +(-10000)*5/8 = -6250 Nm
        print("[Checks Fy_i=5000, Mz_i=6250, Fy_j=5000, Mz_j=-6250]")


    except Exception as e:
        print(f"\nError calculating point load FEF: {e}")

    # Calculate FEFs for a UDL
    try:
        udl_load = MemberUDLoad(id=2, member_id=1, wx=0, wy=-2000) # 2 kN/m downward
        fef_udl = fixed_end_forces(udl_load, L)
        print(f"\n--- FEFs for UDL {udl_load.id} (wy={udl_load.wy} N/m) ---")
        print(np.round(fef_udl, 2))
        # Expected for UDL w: Fy = -wL/2, Mz = -wL^2/12 and +wL^2/12
        # Corrected FEF signs: Fy = +wL/2, Mz = +wL^2/12 and -wL^2/12
        # FEF_Fy = -(-2000)*5/2 = +5000 N
        # FEF_Mz_i = -(-2000)*5^2/12 = +4166.67 Nm
        # FEF_Mz_j = +(-2000)*5^2/12 = -4166.67 Nm
        print("[Checks Fy_i=5000, Mz_i=4166.67, Fy_j=5000, Mz_j=-4166.67]")
    except Exception as e:
        print(f"\nError calculating UDL FEF: {e}")