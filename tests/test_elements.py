# tests/test_elements.py

import pytest
import numpy as np
import math
from numpy.testing import assert_allclose, assert_equal

# Import functions to test
from core.elements import local_stiffness_matrix, transformation_matrix, fixed_end_forces

# Try importing Load classes needed for FEF tests
try:
    from core.model import Load, MemberLoad, MemberPointLoad, MemberUDLoad
    MODEL_CLASSES_AVAILABLE = True
except ImportError:
    MODEL_CLASSES_AVAILABLE = False
    # Define dummy classes if core.model is not available during test collection
    class Load: pass
    class MemberLoad(Load): pass
    class MemberPointLoad(MemberLoad):
         def __init__(self, id, member_id, px, py, position, label=""): pass
    class MemberUDLoad(MemberLoad):
         def __init__(self, id, member_id, wx=0, wy=0, label=""): pass


# === Constants for tests ===
# Use simple values for E, A, I, L to make manual verification easier
TEST_E = 1.0
TEST_A = 1.0
TEST_I = 1.0
TEST_L = 10.0
TOLERANCE = 1e-9 # Tolerance for floating point comparisons

# === Tests for local_stiffness_matrix ===

def test_local_stiffness_matrix_known_values():
    """
    Tests the local stiffness matrix against known textbook values
    using simple E=1, A=1, I=1, L=10 parameters.
    """
    E, A, I, L = TEST_E, TEST_A, TEST_I, TEST_L
    EA_L = E * A / L     # 0.1
    EI_L = E * I / L     # 0.1
    EI_L2 = E * I / L**2 # 0.01
    EI_L3 = E * I / L**3 # 0.001

    expected_k_local = np.array([
        [ EA_L,  0,          0,      -EA_L,  0,          0       ], # u_i
        [ 0,     12*EI_L3,   6*EI_L2, 0,     -12*EI_L3,   6*EI_L2 ], # v_i
        [ 0,     6*EI_L2,    4*EI_L,  0,     -6*EI_L2,    2*EI_L  ], # rz_i
        [-EA_L,  0,          0,       EA_L,  0,          0       ], # u_j
        [ 0,    -12*EI_L3,  -6*EI_L2, 0,     12*EI_L3,   -6*EI_L2 ], # v_j
        [ 0,     6*EI_L2,    2*EI_L,  0,     -6*EI_L2,    4*EI_L  ]  # rz_j
    ])

    # Verify the hand-calculated expected values match the definition
    assert_allclose(expected_k_local[0,0], 0.1, atol=TOLERANCE)
    assert_allclose(expected_k_local[1,1], 0.012, atol=TOLERANCE)
    assert_allclose(expected_k_local[1,2], 0.06, atol=TOLERANCE)
    assert_allclose(expected_k_local[2,2], 0.4, atol=TOLERANCE)
    # etc.

    # Calculate using the function
    k_local = local_stiffness_matrix(E, A, I, L)

    # Assert the calculated matrix is close to the expected matrix
    assert_allclose(k_local, expected_k_local, atol=TOLERANCE)

def test_local_stiffness_matrix_invalid_length():
    """Tests that local_stiffness_matrix raises ValueError for L <= 0."""
    with pytest.raises(ValueError, match="Element length .* must be positive"):
        local_stiffness_matrix(TEST_E, TEST_A, TEST_I, 0.0)
    with pytest.raises(ValueError, match="Element length .* must be positive"):
        local_stiffness_matrix(TEST_E, TEST_A, TEST_I, -5.0)

# === Tests for transformation_matrix ===

def test_transformation_matrix_zero_angle():
    """Tests transformation matrix for a 0 degree angle (should be Identity)."""
    T = transformation_matrix(0.0)
    expected_T = np.identity(6)
    assert_allclose(T, expected_T, atol=TOLERANCE)

def test_transformation_matrix_90_deg():
    """Tests transformation matrix for a 90 degree angle."""
    angle_rad = math.pi / 2.0
    c = 0.0
    s = 1.0
    expected_lambda = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) # [[0,1,0],[-1,0,0],[0,0,1]]
    expected_T = np.zeros((6, 6))
    expected_T[0:3, 0:3] = expected_lambda
    expected_T[3:6, 3:6] = expected_lambda

    T = transformation_matrix(angle_rad)
    assert_allclose(T, expected_T, atol=TOLERANCE)

def test_transformation_matrix_45_deg():
    """Tests transformation matrix for a 45 degree angle."""
    angle_rad = math.pi / 4.0
    c = math.sqrt(2) / 2.0
    s = math.sqrt(2) / 2.0
    expected_lambda = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    expected_T = np.zeros((6, 6))
    expected_T[0:3, 0:3] = expected_lambda
    expected_T[3:6, 3:6] = expected_lambda

    T = transformation_matrix(angle_rad)
    assert_allclose(T, expected_T, atol=TOLERANCE)

def test_transformation_matrix_orthogonality():
    """Tests that T.T @ T = I for various angles (property of rotation matrices)."""
    identity_6x6 = np.identity(6)
    for angle_deg in [0, 30, 45, 90, 120, 180, -45, -90]:
        angle_rad = math.radians(angle_deg)
        T = transformation_matrix(angle_rad)
        # Check T^T * T == I
        result = T.T @ T
        assert_allclose(result, identity_6x6, atol=TOLERANCE,
                        err_msg=f"Orthogonality failed for angle {angle_deg} deg")
        # Check T * T^T == I (also true for orthogonal matrices)
        result_inv = T @ T.T
        assert_allclose(result_inv, identity_6x6, atol=TOLERANCE,
                        err_msg=f"Orthogonality (T*T.T) failed for angle {angle_deg} deg")


# === Tests for fixed_end_forces ===
# Use skipif marker if model classes are needed but unavailable
# This protects against errors if core.model hasn't been fully processed yet by the test runner
pytestmark_fef = pytest.mark.skipif(not MODEL_CLASSES_AVAILABLE, reason="Requires core.model classes (MemberPointLoad/UDLoad)")

@pytestmark_fef
def test_fixed_end_forces_udl_perpendicular():
    """Tests FEFs for a perpendicular UDL."""
    L = TEST_L  # 10.0
    wy = -2000.0 # Downward load (N/m)
    load = MemberUDLoad(id=1, member_id=1, wy=wy)

    # Expected FEF vector [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]
    # Fy = -wy*L/2 => -(-2000)*10/2 = +10000
    # Mz = -wy*L^2/12 => -(-2000)*10^2/12 = +16666.66...
    # Mz_j = +wy*L^2/12 => +(-2000)*10^2/12 = -16666.66...
    expected_fef = np.array([0, 10000, 16666.66666667, 0, 10000, -16666.66666667])

    fef = fixed_end_forces(load, L)
    assert_allclose(fef, expected_fef, rtol=1e-5, atol=TOLERANCE)

@pytestmark_fef
def test_fixed_end_forces_udl_axial():
    """Tests FEFs for an axial UDL."""
    L = TEST_L # 10.0
    wx = 500.0 # Positive axial load (N/m)
    load = MemberUDLoad(id=2, member_id=1, wx=wx)

    # Expected FEF vector [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]
    # Fx = -wx*L/2 => -(500)*10/2 = -2500
    expected_fef = np.array([-2500, 0, 0, -2500, 0, 0])

    fef = fixed_end_forces(load, L)
    assert_allclose(fef, expected_fef, atol=TOLERANCE)

@pytestmark_fef
def test_fixed_end_forces_point_load_perpendicular_mid():
    """Tests FEFs for a perpendicular point load at mid-span."""
    L = TEST_L # 10.0
    py = -10000.0 # Downward load (N)
    position = L / 2.0 # 5.0
    a = position
    b = L - a # 5.0
    load = MemberPointLoad(id=3, member_id=1, px=0, py=py, position=position)

    # Expected FEF vector [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]
    # Fy_i = -py*b^2*(L+2a)/L^3 => -(-10000)*5^2*(10+10)/10^3 = +5000
    # Mz_i = -py*a*b^2/L^2 => -(-10000)*5*5^2/10^2 = +12500
    # Fy_j = -py*a^2*(L+2b)/L^3 => -(-10000)*5^2*(10+10)/10^3 = +5000
    # Mz_j = +py*b*a^2/L^2 => +(-10000)*5*5^2/10^2 = -12500
    expected_fef = np.array([0, 5000, 12500, 0, 5000, -12500])

    fef = fixed_end_forces(load, L)
    assert_allclose(fef, expected_fef, atol=TOLERANCE)

@pytestmark_fef
def test_fixed_end_forces_point_load_perpendicular_off_center():
    """Tests FEFs for a perpendicular point load off mid-span."""
    L = TEST_L # 10.0
    py = 5000.0 # Upward load (N)
    position = L / 4.0 # 2.5
    a = position # 2.5
    b = L - a # 7.5
    load = MemberPointLoad(id=4, member_id=1, px=0, py=py, position=position)

    # Expected FEF vector [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]
    # Fy_i = -py*b^2*(L+2a)/L^3 => -(5000)*7.5^2*(10+5)/10^3 = -4218.75
    # Mz_i = -py*a*b^2/L^2 => -(5000)*2.5*7.5^2/10^2 = -7031.25
    # Fy_j = -py*a^2*(L+2b)/L^3 => -(5000)*2.5^2*(10+15)/10^3 = -781.25
    # Mz_j = +py*b*a^2/L^2 => +(5000)*7.5*2.5^2/10^2 = +2343.75
    expected_fef = np.array([0, -4218.75, -7031.25, 0, -781.25, 2343.75])

    fef = fixed_end_forces(load, L)
    assert_allclose(fef, expected_fef, atol=TOLERANCE)

@pytestmark_fef
def test_fixed_end_forces_point_load_axial():
    """Tests FEFs for an axial point load."""
    L = TEST_L # 10.0
    px = 1000.0 # Positive axial load (N)
    position = L * 0.3 # 3.0
    a = position # 3.0
    b = L - a # 7.0
    load = MemberPointLoad(id=5, member_id=1, px=px, py=0, position=position)

    # Expected FEF vector [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]
    # Fx_i = -px*b/L => -(1000)*7/10 = -700
    # Fx_j = -px*a/L => -(1000)*3/10 = -300
    expected_fef = np.array([-700, 0, 0, -300, 0, 0])

    fef = fixed_end_forces(load, L)
    assert_allclose(fef, expected_fef, atol=TOLERANCE)


@pytestmark_fef
def test_fixed_end_forces_invalid_length():
    """Tests that fixed_end_forces raises ValueError for L <= 0."""
    udl = MemberUDLoad(id=99, member_id=1, wy=1)
    pt = MemberPointLoad(id=99, member_id=1, px=0, py=1, position=1)
    with pytest.raises(ValueError, match="Element length .* must be positive"):
        fixed_end_forces(udl, 0.0)
    with pytest.raises(ValueError, match="Element length .* must be positive"):
        fixed_end_forces(pt, -1.0)

@pytestmark_fef
def test_fixed_end_forces_invalid_position():
    """
    Tests that fixed_end_forces raises ValueError for position > L.
    (Negative position check is now done in MemberPointLoad.__init__).
    """
    L = TEST_L
    # Position > L
    pt_long = MemberPointLoad(id=99, member_id=1, px=0, py=1, position=L + 0.1)

    # Test only the case where position exceeds length
    with pytest.raises(ValueError, match="MemberPointLoad position .* is outside member length"):
        fixed_end_forces(pt_long, L)

    # Remove the test for negative position from this function
    # # Position < 0
    # pt_neg = MemberPointLoad(id=99, member_id=1, px=0, py=1, position=-0.1)
    # with pytest.raises(ValueError, match=".* position .* cannot be negative"):
    #     fixed_end_forces(pt_neg, L) # This is now tested in test_model.py



# Dummy load class for testing unsupported types
class DummyMemberLoad(MemberLoad):
     def __init__(self, id, member_id):
         super().__init__(id, member_id)
     def __repr__(self): return "Dummy"
     def __str__(self): return "Dummy"

@pytestmark_fef
def test_fixed_end_forces_unsupported_load_type():
    """Tests that fixed_end_forces raises NotImplementedError for unsupported types."""
    dummy_load = DummyMemberLoad(id=100, member_id=1)
    with pytest.raises(NotImplementedError, match="calculation not implemented for load type"):
        fixed_end_forces(dummy_load, TEST_L)