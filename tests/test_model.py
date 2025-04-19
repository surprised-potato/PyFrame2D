# tests/test_model.py

import pytest # type: ignore
from pytest import approx  # type: ignore # For comparing floating-point numbers
# Attempt to import the Node class (it doesn't exist yet, but we write the test assuming it will)

from core.model import Node, Material, SectionProfile, RectangularProfile, SquareProfile, IBeamProfile, Member
from core.model import Support, Load, NodalLoad, MemberLoad, MemberPointLoad, MemberUDLoad
import math

# Make sure UNITS_ENABLED is accessible if needed for skipping tests, or handle errors
try:
    from units.units import UNITS_ENABLED
except ImportError:
    UNITS_ENABLED = False # Assume disabled if units module itself fails

# --- Node Class Tests ---
def test_node_init_invalid_id_type():
    """Tests that Node raises TypeError for non-integer ID."""
    with pytest.raises(TypeError, match="Node ID must be an integer"):
        Node(id="A", x=0.0, y=0.0)
    with pytest.raises(TypeError, match="Node ID must be an integer"):
        Node(id=1.5, x=0.0, y=0.0)

def test_node_init_invalid_coord_type():
    """Tests that Node raises TypeError for non-numeric coordinates."""
    with pytest.raises(TypeError, match="Node coordinates .* must be numeric"):
        Node(id=7, x="invalid", y=0.0)
    with pytest.raises(TypeError, match="Node coordinates .* must be numeric"):
        Node(id=8, x=0.0, y="invalid")
    with pytest.raises(TypeError, match="Node coordinates .* must be numeric"):
        Node(id=9, x=None, y=0.0) # None is not numeric
    with pytest.raises(TypeError, match="Node coordinates .* must be numeric"):
        Node(id=10, x=0.0, y=None)

def test_node_creation_negative_coords():
    """Tests Node creation with negative coordinates."""
    node = Node(id=4, x=-100.5, y=-200.75)
    assert node.id == 4
    assert node.x == approx(-100.5)
    assert node.y == approx(-200.75)
    assert node.get_coords() == approx((-100.5, -200.75))

def test_node_creation_zero_coords():
    """Tests Node creation with zero coordinates."""
    node = Node(id=5, x=0.0, y=0.0)
    assert node.id == 5
    assert node.x == approx(0.0)
    assert node.y == approx(0.0)
    assert node.get_coords() == approx((0.0, 0.0))

def test_node_attribute_retrieval():
    """Explicitly tests direct attribute retrieval."""
    node_id = 6
    x_coord = 1.1
    y_coord = 2.2
    node = Node(id=node_id, x=x_coord, y=y_coord)

    # Verify direct retrieval matches input values
    assert node.id == node_id
    assert node.x == approx(x_coord)
    assert node.y == approx(y_coord)
def test_node_creation_basic():
    """Tests basic creation and attribute assignment of a Node."""
    node_id = 1
    x_coord = 10.5
    y_coord = -5.0
    node = Node(id=node_id, x=x_coord, y=y_coord)

    assert node.id == node_id
    assert node.x == approx(x_coord)
    assert node.y == approx(y_coord)
    assert isinstance(node.id, int)
    assert isinstance(node.x, float)
    assert isinstance(node.y, float)

def test_node_get_coords():
    """Tests the get_coords method."""
    node = Node(id=2, x=0.0, y=100.25)
    coords = node.get_coords()

    assert isinstance(coords, tuple)
    assert len(coords) == 2
    assert coords[0] == approx(0.0)
    assert coords[1] == approx(100.25)
    assert coords == approx((0.0, 100.25))

def test_node_representation():
    """Tests the __repr__ method for a clear representation."""
    node = Node(id=3, x=12.3, y=45.6)
    expected_repr = "Node(id=3, x=12.3, y=45.6)"
    assert repr(node) == expected_repr

def test_node_equality():
    """Tests the __eq__ method based on node ID."""
    node1a = Node(id=10, x=1.0, y=1.0)
    node1b = Node(id=10, x=2.0, y=2.0) # Same ID, different coordinates
    node2 = Node(id=20, x=1.0, y=1.0)  # Different ID

    assert node1a == node1b  # Nodes with the same ID should be equal
    assert node1a != node2   # Nodes with different IDs should not be equal
    assert node1b != node2
    assert node1a != "some_string" # Should not be equal to other types

# If running pytest now, these tests *should fail* or use the DummyNode,
# because core.model.Node doesn't properly exist yet.


def test_material_creation_numeric_e():
    """Tests Material creation with numeric Young's Modulus (Pa)."""
    mat = Material(id=1, name="Generic Steel", youngs_modulus=210e9)
    assert mat.id == 1
    assert mat.name == "Generic Steel"
    assert mat.E == approx(210e9)

@pytest.mark.skipif(not UNITS_ENABLED, reason="Units module not available/enabled")
def test_material_creation_string_e():
    """Tests Material creation with string Young's Modulus and unit parsing."""
    mat1 = Material(id=2, name="Steel S235", youngs_modulus="210 GPa")
    assert mat1.id == 2
    assert mat1.name == "Steel S235"
    assert mat1.E == approx(2.1e11)

    mat2 = Material(id=3, name="Aluminum", youngs_modulus="70000 MPa")
    assert mat2.id == 3
    assert mat2.E == approx(7e10)

    mat3 = Material(id=4, name="Titanium", youngs_modulus=" 110e9 Pa ") # Test whitespace
    assert mat3.id == 4
    assert mat3.E == approx(1.1e11)

def test_material_repr_and_str():
    """Tests Material string representations."""
    mat = Material(id=5, name="Concrete C30/37", youngs_modulus=33e9)
    # Expected E string depends on units module
    e_pa_str = "3.3e+10 Pa"
    e_unit_str = "33 GPa"

    expected_repr = f"Material(id=5, name='Concrete C30/37', E='{e_unit_str}')"
    expected_str = f"Material 5: Concrete C30/37 (E = {e_unit_str})"

    assert repr(mat) == expected_repr
    assert str(mat) == expected_str

def test_material_equality_and_hash():
    """Tests Material equality and hashing based on ID."""
    mat1a = Material(id=10, name="Mat A", youngs_modulus=1e11)
    mat1b = Material(id=10, name="Mat A v2", youngs_modulus=2e11) # Same ID
    mat2 = Material(id=20, name="Mat B", youngs_modulus=1e11)

    assert mat1a == mat1b
    assert mat1a != mat2
    assert hash(mat1a) == hash(mat1b)
    assert hash(mat1a) != hash(mat2)
    assert mat1a != "string"

    mat_set = {mat1a, mat1b, mat2}
    assert len(mat_set) == 2 # Only two unique IDs

def test_material_invalid_inputs():
    """Tests invalid inputs during Material initialization."""
    # Invalid ID
    with pytest.raises(TypeError, match="Material ID must be an integer"):
        Material(id="abc", name="Test", youngs_modulus=1e9)
    # Invalid Name
    with pytest.raises(TypeError, match="Material name must be a string"):
        Material(id=1, name=123, youngs_modulus=1e9)
    with pytest.raises(ValueError, match="Material name cannot be empty"):
        Material(id=1, name="", youngs_modulus=1e9)
    # Invalid E type
    with pytest.raises(TypeError, match="Invalid type for Young's Modulus"):
        Material(id=1, name="Test", youngs_modulus=None)
    # Invalid E value
    with pytest.raises(ValueError, match="Young's Modulus must be a positive value"):
        Material(id=1, name="Test", youngs_modulus=0)
    with pytest.raises(ValueError, match="Young's Modulus must be a positive value"):
        Material(id=1, name="Test", youngs_modulus=-210e9)

@pytest.mark.skipif(not UNITS_ENABLED, reason="Units module not available/enabled")
def test_material_invalid_string_e():
     """Tests invalid string inputs for Young's Modulus."""
     with pytest.raises(ValueError, match="Invalid Young's Modulus string"):
        Material(id=1, name="Test", youngs_modulus="200 BadUnit")
     with pytest.raises(ValueError, match="Invalid Young's Modulus string"):
        Material(id=1, name="Test", youngs_modulus="GPa") # Unit only
     with pytest.raises(ValueError, match="Invalid Young's Modulus string"):
        Material(id=1, name="Test", youngs_modulus="abc GPa") # Non-numeric value

# --- SectionProfile Base Class Tests ---

def test_sectionprofile_cannot_instantiate():
    """Verifies that the abstract SectionProfile class cannot be instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class SectionProfile"):
        SectionProfile(id=1, name="Abstract Section")

# --- RectangularProfile Tests ---

def test_rectangular_creation_and_props():
    """Tests RectangularProfile creation and calculated properties."""
    section = RectangularProfile(id=101, name="Beam 200x400", width=0.2, height=0.4)
    assert section.id == 101
    assert section.name == "Beam 200x400"
    assert section.width == approx(0.2)
    assert section.height == approx(0.4)
    assert section.area == approx(0.2 * 0.4)         # A = 0.08
    assert section.moment_of_inertia == approx((0.2 * 0.4**3) / 12.0) # I = 0.0010666...

def test_rectangular_repr_and_str():
    """Tests RectangularProfile string representations."""
    section = RectangularProfile(id=102, name="Col 300x300", width=0.3, height=0.3)
    expected_repr = "RectangularProfile(id=102, name='Col 300x300', width=0.3, height=0.3)"
    expected_str = "Section 102: Col 300x300 (Rectangular 0.3m x 0.3m)"
    assert repr(section) == expected_repr
    assert str(section) == expected_str

def test_rectangular_invalid_inputs():
    """Tests invalid inputs for RectangularProfile."""
    # Invalid dimensions type
    with pytest.raises(TypeError, match="dimensions .* must be numeric"):
        RectangularProfile(id=1, name="Test", width="a", height=0.1)
    with pytest.raises(TypeError, match="dimensions .* must be numeric"):
        RectangularProfile(id=1, name="Test", width=0.1, height="b")
    # Non-positive dimensions
    with pytest.raises(ValueError, match="dimensions .* must be positive"):
        RectangularProfile(id=1, name="Test", width=0, height=0.1)
    with pytest.raises(ValueError, match="dimensions .* must be positive"):
        RectangularProfile(id=1, name="Test", width=0.1, height=-0.1)

# --- SquareProfile Tests ---

def test_square_creation_and_props():
    """Tests SquareProfile creation and calculated properties."""
    section = SquareProfile(id=201, name="SHS 150", side_length=0.15)
    assert section.id == 201
    assert section.name == "SHS 150"
    assert section.side_length == approx(0.15)
    assert section.area == approx(0.15**2)           # A = 0.0225
    assert section.moment_of_inertia == approx((0.15**4) / 12.0) # I = 4.21875e-05

def test_square_repr_and_str():
    """Tests SquareProfile string representations."""
    section = SquareProfile(id=202, name="Post 100", side_length=0.1)
    expected_repr = "SquareProfile(id=202, name='Post 100', side_length=0.1)"
    expected_str = "Section 202: Post 100 (Square 0.1m)"
    assert repr(section) == expected_repr
    assert str(section) == expected_str

def test_square_invalid_inputs():
    """Tests invalid inputs for SquareProfile."""
    # Invalid dimension type
    with pytest.raises(TypeError, match="side_length must be numeric"):
        SquareProfile(id=1, name="Test", side_length="abc")
    # Non-positive dimension
    with pytest.raises(ValueError, match="side_length must be positive"):
        SquareProfile(id=1, name="Test", side_length=0)
    with pytest.raises(ValueError, match="side_length must be positive"):
        SquareProfile(id=1, name="Test", side_length=-0.1)

# --- IBeamProfile Tests ---

def test_ibeam_creation_and_props():
    """Tests IBeamProfile creation and calculated properties."""
    # Example: HEA 200 approx dimensions (adjust if using precise tables)
    h, bf, tf, tw = 0.190, 0.200, 0.010, 0.0065 # meters
    section = IBeamProfile(id=301, name="HEA 200", height=h, flange_width=bf,
                             flange_thickness=tf, web_thickness=tw)

    assert section.id == 301
    assert section.name == "HEA 200"
    assert section.height == approx(h)
    assert section.flange_width == approx(bf)
    assert section.flange_thickness == approx(tf)
    assert section.web_thickness == approx(tw)

    # Manual calculation verification
    expected_area = 2 * (bf * tf) + (h - 2 * tf) * tw
    expected_I = (bf * h**3 - (bf - tw) * (h - 2 * tf)**3) / 12.0
    assert section.area == approx(0.005105, rel=1e-4)
    assert section.moment_of_inertia == approx(3.509454e-05, rel=1e-5) # Use correct value & maybe tighter tolerance


def test_ibeam_repr_and_str():
    """Tests IBeamProfile string representations."""
    h, bf, tf, tw = 0.190, 0.200, 0.010, 0.0065
    section = IBeamProfile(id=302, name="HEA 200", height=h, flange_width=bf,
                           flange_thickness=tf, web_thickness=tw)
    expected_repr = ("IBeamProfile(id=302, name='HEA 200', h=0.19, bf=0.2, tf=0.01, tw=0.0065)")
    expected_str = ("Section 302: HEA 200 "
                    "(I-Beam h=0.19m, bf=0.2m, tf=0.01m, tw=0.0065m)") # Using .3g formatting
    assert repr(section) == expected_repr
    # String comparison might need adjustment based on exact float formatting
    assert str(section) == expected_str


def test_ibeam_invalid_inputs():
    """Tests invalid inputs for IBeamProfile."""
    # Valid dimensions for reference
    h, bf, tf, tw = 0.2, 0.2, 0.01, 0.006

    # Invalid dimension types
    with pytest.raises(TypeError, match="dimensions must be numeric"):
        IBeamProfile(id=1, name="T", height="a", flange_width=bf, flange_thickness=tf, web_thickness=tw)
    # ... (could add more type checks for bf, tf, tw) ...

    # Non-positive dimensions
    with pytest.raises(ValueError, match="dimensions must be positive"):
        IBeamProfile(id=1, name="T", height=0, flange_width=bf, flange_thickness=tf, web_thickness=tw)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        IBeamProfile(id=1, name="T", height=h, flange_width=bf, flange_thickness=-tf, web_thickness=tw)
    # ... (could add more positivity checks for bf, tw) ...

    # Invalid geometry
    with pytest.raises(ValueError, match="height .* must be greater than 2 .* flange_thickness"):
        # h <= 2*tf  (0.02 <= 2*0.01)
        IBeamProfile(id=1, name="T", height=0.02, flange_width=bf, flange_thickness=tf, web_thickness=tw)
    with pytest.raises(ValueError, match="flange_width .* must be greater than or equal to web_thickness"):
         # bf < tw (0.005 < 0.006)
        IBeamProfile(id=1, name="T", height=h, flange_width=0.005, flange_thickness=tf, web_thickness=tw)

# --- Test SectionProfile Equality Across Types (Based on ID) ---

def test_section_equality_across_types():
    """Tests that sections compare based on ID, even if types differ."""
    rect = RectangularProfile(id=500, name="Sec 500 R", width=0.1, height=0.1)
    square = SquareProfile(id=500, name="Sec 500 S", side_length=0.1) # Same ID
    ibeam = IBeamProfile(id=501, name="Sec 501 I", height=0.2, flange_width=0.1, flange_thickness=0.01, web_thickness=0.005)

    assert rect == square # Equality based on ID only (as currently implemented)
    assert rect != ibeam
    assert square != ibeam
    assert hash(rect) == hash(square) # Hashes should match if IDs match
    assert hash(rect) != hash(ibeam)

    section_set = {rect, square, ibeam}
    assert len(section_set) == 2 # Only two unique IDs

def test_material_e_retrieval():
    """Explicitly tests retrieval of the E property after creation."""
    e_value = 195.5e9
    mat = Material(id=6, name="Special Alloy", youngs_modulus=e_value)
    assert mat.E == approx(e_value)

    # Test retrieval after creation with string (if units enabled)
    if UNITS_ENABLED:
        mat_str = Material(id=7, name="Another Alloy", youngs_modulus="195.5 GPa")
        assert mat_str.E == approx(e_value)


def test_rectangular_thin_profile():
    """Tests a rectangular profile with very different width and height."""
    # Very wide, very thin
    section1 = RectangularProfile(id=103, name="Plate", width=1.0, height=0.001)
    assert section1.area == approx(1.0 * 0.001)
    assert section1.moment_of_inertia == approx((1.0 * 0.001**3) / 12.0)

    # Very tall, very narrow
    section2 = RectangularProfile(id=104, name="Fin", width=0.002, height=0.5)
    assert section2.area == approx(0.002 * 0.5)
    assert section2.moment_of_inertia == approx((0.002 * 0.5**3) / 12.0)


def test_ibeam_near_limits():
    """Tests IBeamProfile with dimensions close to validation limits."""
    # h just slightly larger than 2*tf (e.g., h=0.021, tf=0.01)
    h, bf, tf, tw = 0.021, 0.05, 0.01, 0.005 # Ensure bf >= tw (0.05 >= 0.005)
    section1 = IBeamProfile(id=303, name="Shallow I", height=h, flange_width=bf,
                             flange_thickness=tf, web_thickness=tw)
    # Just verify creation doesn't raise error and calculate basic properties
    expected_area = 2*(bf*tf) + (h - 2*tf)*tw
    expected_I = (bf * h**3 - (bf - tw) * (h - 2*tf)**3) / 12.0
    assert section1.area == approx(expected_area)
    assert section1.moment_of_inertia == approx(expected_I)

    # bf just equal to tw (e.g., bf=0.005, tw=0.005)
    h, bf, tf, tw = 0.1, 0.005, 0.01, 0.005 # Ensure h > 2*tf (0.1 > 0.02)
    section2 = IBeamProfile(id=304, name="Narrow Flange I", height=h, flange_width=bf,
                             flange_thickness=tf, web_thickness=tw)
    expected_area = 2*(bf*tf) + (h - 2*tf)*tw
    expected_I = (bf * h**3 - (bf - tw) * (h - 2*tf)**3) / 12.0 # Note: (bf-tw) term becomes zero here
    assert section2.area == approx(expected_area)
    assert section2.moment_of_inertia == approx(expected_I)
    # Verify I simplifies correctly when bf=tw
    assert section2.moment_of_inertia == approx((bf * h**3)/12.0)

    from core.model import Member

# --- Member Class Tests ---

# Helper fixture to create nodes, material, section for member tests
@pytest.fixture
def basic_model_components():
    """Provides common nodes, material, section for member tests."""
    n1 = Node(id=1, x=0, y=0)
    n2 = Node(id=2, x=5, y=0)      # Horizontal member N1->N2, length 5
    n3 = Node(id=3, x=5, y=10)     # Vertical member N2->N3, length 10
    n4 = Node(id=4, x=-3, y=4)     # Angled member N1->N4, length 5 (3-4-5 triangle)
    n_dup = Node(id=1, x=1, y=1)   # Duplicate ID for zero-length test
    mat = Material(id=1, name="Generic Steel", youngs_modulus=200e9)
    sec = SquareProfile(id=101, name="SQ 100", side_length=0.1)
    return {"n1": n1, "n2": n2, "n3": n3, "n4": n4, "n_dup": n_dup, "mat": mat, "sec": sec}

def test_member_creation(basic_model_components):
    """Tests basic Member creation and attribute assignment."""
    n1, n2 = basic_model_components["n1"], basic_model_components["n2"]
    mat, sec = basic_model_components["mat"], basic_model_components["sec"]

    member = Member(id=1, start_node=n1, end_node=n2, material=mat, section=sec)

    assert member.id == 1
    assert member.start_node == n1
    assert member.end_node == n2
    assert member.material == mat
    assert member.section == sec
    # Check reference equality (should be the same objects)
    assert member.start_node is n1
    assert member.end_node is n2
    assert member.material is mat
    assert member.section is sec

def test_member_length_calculation(basic_model_components):
    """Tests the member length property calculation."""
    n1 = basic_model_components["n1"]
    n2 = basic_model_components["n2"]
    n3 = basic_model_components["n3"]
    n4 = basic_model_components["n4"]
    mat = basic_model_components["mat"]
    sec = basic_model_components["sec"]

    mem_horiz = Member(id=2, start_node=n1, end_node=n2, material=mat, section=sec)
    mem_vert = Member(id=3, start_node=n2, end_node=n3, material=mat, section=sec)
    mem_angled = Member(id=4, start_node=n1, end_node=n4, material=mat, section=sec)

    assert mem_horiz.length == approx(5.0)
    assert mem_vert.length == approx(10.0)
    assert mem_angled.length == approx(math.sqrt((-3 - 0)**2 + (4 - 0)**2)) # sqrt(9+16)=5
    assert mem_angled.length == approx(5.0)

def test_member_angle_calculation(basic_model_components):
    """Tests the member angle property calculation (in radians)."""
    n1 = basic_model_components["n1"]
    n2 = basic_model_components["n2"]
    n3 = basic_model_components["n3"]
    n4 = basic_model_components["n4"]
    mat = basic_model_components["mat"]
    sec = basic_model_components["sec"]

    mem_horiz = Member(id=5, start_node=n1, end_node=n2, material=mat, section=sec) # 0 degrees
    mem_vert = Member(id=6, start_node=n2, end_node=n3, material=mat, section=sec)  # 90 degrees
    mem_angled = Member(id=7, start_node=n1, end_node=n4, material=mat, section=sec)# Angle in Q2
    mem_neg_horiz = Member(id=8, start_node=n2, end_node=n1, material=mat, section=sec) # 180 degrees (-pi rad)
    mem_neg_vert = Member(id=9, start_node=n3, end_node=n2, material=mat, section=sec)  # -90 degrees (-pi/2 rad)

    assert mem_horiz.angle == approx(0.0)
    assert mem_vert.angle == approx(math.pi / 2.0)
    assert mem_angled.angle == approx(math.atan2(4, -3)) # approx 2.214 rad or 126.87 deg
    assert mem_neg_horiz.angle == approx(math.pi)        # atan2(-0, -5) -> pi
    assert mem_neg_vert.angle == approx(-math.pi / 2.0)

def test_member_property_accessors(basic_model_components):
    """Tests the convenience properties E, A, I."""
    n1, n2 = basic_model_components["n1"], basic_model_components["n2"]
    mat, sec = basic_model_components["mat"], basic_model_components["sec"]
    member = Member(id=10, start_node=n1, end_node=n2, material=mat, section=sec)

    assert member.E == approx(mat.E)
    assert member.A == approx(sec.area)
    assert member.I == approx(sec.moment_of_inertia)

def test_member_repr_and_str(basic_model_components):
    """Tests Member string representations."""
    n1, n2 = basic_model_components["n1"], basic_model_components["n2"]
    mat, sec = basic_model_components["mat"], basic_model_components["sec"]
    member = Member(id=11, start_node=n1, end_node=n2, material=mat, section=sec)

    expected_repr = ("Member(id=11, start_node=Node(id=1), end_node=Node(id=2), "
                     "material=Material(id=1), section=SquareProfile(id=101))")
    expected_str = "Member 11 (Nodes: 1 -> 2, Material: 1, Section: 101)"

    assert repr(member) == expected_repr
    assert str(member) == expected_str

def test_member_equality_and_hash(basic_model_components):
    """Tests Member equality and hashing based on ID."""
    n1, n2, n3 = basic_model_components["n1"], basic_model_components["n2"], basic_model_components["n3"]
    mat, sec = basic_model_components["mat"], basic_model_components["sec"]

    mem1a = Member(id=20, start_node=n1, end_node=n2, material=mat, section=sec)
    mem1b = Member(id=20, start_node=n2, end_node=n3, material=mat, section=sec) # Same ID
    mem2 = Member(id=21, start_node=n1, end_node=n3, material=mat, section=sec)

    assert mem1a == mem1b
    assert mem1a != mem2
    assert hash(mem1a) == hash(mem1b)
    assert hash(mem1a) != hash(mem2)
    assert mem1a != "string"

    mem_set = {mem1a, mem1b, mem2}
    assert len(mem_set) == 2 # Only two unique IDs


def test_member_invalid_inputs(basic_model_components):
    """Tests invalid inputs during Member initialization."""
    n1, n2 = basic_model_components["n1"], basic_model_components["n2"]
    n_dup = basic_model_components["n_dup"] # Same ID as n1 but different coords
    mat, sec = basic_model_components["mat"], basic_model_components["sec"]

    # Invalid ID
    with pytest.raises(TypeError, match="Member ID must be an integer"):
        Member(id="abc", start_node=n1, end_node=n2, material=mat, section=sec)
    # Invalid Node types
    with pytest.raises(TypeError, match="Member start_node must be a Node object"):
        Member(id=1, start_node="n1", end_node=n2, material=mat, section=sec)
    with pytest.raises(TypeError, match="Member end_node must be a Node object"):
        Member(id=1, start_node=n1, end_node=None, material=mat, section=sec)
    # Invalid Material type
    with pytest.raises(TypeError, match="Member material must be a Material object"):
        Member(id=1, start_node=n1, end_node=n2, material="mat", section=sec)
    # Invalid Section type
    with pytest.raises(TypeError, match="Member section must be a SectionProfile object"):
        Member(id=1, start_node=n1, end_node=n2, material=mat, section=n1) # Pass node as section

    # Zero-length member (based on coordinates)
    with pytest.raises(ValueError, match="Member start and end node coordinates are identical"): # Correct match
        node_a = Node(id=1001, x=10, y=10)
        node_b = Node(id=1002, x=10, y=10) # Different ID, same coordinates
        Member(id=1, start_node=node_a, end_node=node_b, material=mat, section=sec)

    # Let's test the case where IDs are same but coords differ - this should NOT raise ValueError
    try:
        Member(id=1, start_node=n1, end_node=n_dup, material=mat, section=sec)
    except ValueError:
        pytest.fail("ValueError raised unexpectedly for nodes with same ID but different coordinates.")

def test_support_creation_direct():
    """Tests direct creation of Support object."""
    support = Support(node_id=1, dx=True, dy=True, rz=False) # Pinned
    assert support.node_id == 1
    assert support.is_dx_restrained is True
    assert support.is_dy_restrained is True
    assert support.is_rz_restrained is False

def test_support_creation_classmethods():
    """Tests creation using classmethod constructors."""
    fixed = Support.fixed(node_id=2)
    pinned = Support.pinned(node_id=3)
    roller_x = Support.roller_x(node_id=4) # Restrains dy
    roller_y = Support.roller_y(node_id=5) # Restrains dx

    assert fixed.node_id == 2 and fixed.dx and fixed.dy and fixed.rz
    assert pinned.node_id == 3 and pinned.dx and pinned.dy and not pinned.rz
    assert roller_x.node_id == 4 and not roller_x.dx and roller_x.dy and not roller_x.rz
    assert roller_y.node_id == 5 and roller_y.dx and not roller_y.dy and not roller_y.rz

def test_support_repr_and_str():
    """Tests Support string representations."""
    fixed = Support.fixed(node_id=1)
    pinned = Support.pinned(node_id=2)
    roller_x = Support.roller_x(node_id=3)
    custom = Support(node_id=4, dx=False, dy=False, rz=True)

    assert repr(fixed) == "Support(node_id=1, dx=True, dy=True, rz=True)"
    assert str(fixed) == "Support @ Node 1: Restrains DX+DY+RZ (Fixed)"
    assert str(pinned) == "Support @ Node 2: Restrains DX+DY (Pinned)"
    assert str(roller_x) == "Support @ Node 3: Restrains DY (Roller X)"
    assert str(custom) == "Support @ Node 4: Restrains RZ" # No specific name

def test_support_equality_and_hash():
    """Tests Support equality (based on node_id)."""
    sup1a = Support.fixed(node_id=10)
    sup1b = Support.pinned(node_id=10) # Different constraints, same node
    sup2 = Support.fixed(node_id=20)

    assert sup1a == sup1b # Equal based on node_id
    assert sup1a != sup2
    assert hash(sup1a) == hash(sup1b)
    assert hash(sup1a) != hash(sup2)
    assert sup1a != "string"

    support_set = {sup1a, sup1b, sup2}
    assert len(support_set) == 2 # Only two unique node IDs

def test_support_invalid_inputs():
    """Tests invalid inputs during Support initialization."""
    # Invalid node_id type
    with pytest.raises(TypeError, match="Support node_id must be an integer"):
        Support("1", dx=True, dy=True, rz=True)
    # Invalid restraint type
    with pytest.raises(TypeError, match="Support restraints .* must be boolean"):
        Support(1, dx="True", dy=True, rz=True)
    with pytest.raises(TypeError, match="Support restraints .* must be boolean"):
        Support(1, dx=True, dy=1, rz=True)


# --- NodalLoad Class Tests ---

def test_nodalload_creation():
    """Tests NodalLoad creation with forces and moments."""
    load1 = NodalLoad(id=101, node_id=1, fx=1000.0, fy=-500.0)
    load2 = NodalLoad(id=102, node_id=2, mz=1500.0, label="Wind Moment")

    assert load1.id == 101 and load1.node_id == 1
    assert load1.fx == approx(1000.0) and load1.fy == approx(-500.0) and load1.mz == approx(0.0)
    assert load1.label == ""

    assert load2.id == 102 and load2.node_id == 2
    assert load2.fx == approx(0.0) and load2.fy == approx(0.0) and load2.mz == approx(1500.0)
    assert load2.label == "Wind Moment"

def test_nodalload_repr_and_str():
    """Tests NodalLoad string representations."""
    load = NodalLoad(id=103, node_id=5, fx=-2.5e3, mz=1.2e3, label=" Crane ")

    expected_repr = "NodalLoad(id=103, node_id=5, fx=-2500.0, fy=0.0, mz=1200.0, label='Crane')"
    expected_str = "NodalLoad 103 @ Node 5: Fx=-2.5e+03 N, Fy=0 N, Mz=1.2e+03 Nm (Crane)"

    assert repr(load) == expected_repr
    assert str(load) == expected_str


def test_nodalload_invalid_inputs():
    """Tests invalid inputs during NodalLoad initialization."""
    # Invalid ID / node_id
    with pytest.raises(TypeError, match="Load ID must be an integer"):
        NodalLoad(id="L1", node_id=1)
    with pytest.raises(TypeError, match="NodalLoad node_id must be an integer"):
        NodalLoad(id=1, node_id=1.0)
    # Invalid label
    with pytest.raises(TypeError, match="Load label must be a string"):
        NodalLoad(id=1, node_id=1, label=123)
    # Invalid force/moment types
    with pytest.raises(TypeError, match="NodalLoad components .* must be numeric"):
        NodalLoad(id=1, node_id=1, fx="100")
    with pytest.raises(TypeError, match="NodalLoad components .* must be numeric"):
        NodalLoad(id=1, node_id=1, fy=None)
    with pytest.raises(TypeError, match="NodalLoad components .* must be numeric"):
        NodalLoad(id=1, node_id=1, mz=[100])


# --- MemberLoad Base Class Tests ---

def test_memberload_cannot_instantiate():
    """Verifies that the abstract MemberLoad class cannot be instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class MemberLoad"):
        MemberLoad(id=201, member_id=1)


# --- MemberPointLoad Class Tests ---

def test_memberpointload_creation():
    """Tests MemberPointLoad creation."""
    load = MemberPointLoad(id=301, member_id=10, px=0.0, py=-1000.0, position=2.5, label="Mid span load")

    assert load.id == 301 and load.member_id == 10
    assert load.px == approx(0.0)
    assert load.py == approx(-1000.0)
    assert load.position == approx(2.5)
    assert load.label == "Mid span load"

def test_memberpointload_repr_and_str():
    """Tests MemberPointLoad string representations."""
    load = MemberPointLoad(id=302, member_id=11, px=500, py=0, position=0) # Load at start

    expected_repr = "MemberPointLoad(id=302, member_id=11, px=500.0, py=0.0, position=0.0)"
    expected_str = "MemberPointLoad 302 on Member 11: Px=500 N, Py=0 N @ 0 m"

    assert repr(load) == expected_repr
    assert str(load) == expected_str

def test_memberpointload_invalid_inputs():
    """Tests invalid inputs for MemberPointLoad."""
    # Invalid ID / member_id / label (inherited checks)
    with pytest.raises(TypeError): MemberPointLoad(id="L1", member_id=1, px=0, py=0, position=1)
    with pytest.raises(TypeError): MemberPointLoad(id=1, member_id="M1", px=0, py=0, position=1)
    with pytest.raises(TypeError): MemberPointLoad(id=1, member_id=1, px=0, py=0, position=1, label=None)
    # Invalid numeric types
    with pytest.raises(TypeError, match="MemberPointLoad components .* must be numeric"):
        MemberPointLoad(id=1, member_id=1, px="a", py=0, position=1)
    with pytest.raises(TypeError, match="MemberPointLoad components .* must be numeric"):
        MemberPointLoad(id=1, member_id=1, px=0, py="b", position=1)
    with pytest.raises(TypeError, match="MemberPointLoad components .* must be numeric"):
        MemberPointLoad(id=1, member_id=1, px=0, py=0, position="c")
    # Invalid position value
    with pytest.raises(ValueError, match="MemberPointLoad position .* cannot be negative"):
        MemberPointLoad(id=1, member_id=1, px=0, py=0, position=-1.0)


# --- MemberUDLoad Class Tests ---

def test_memberudload_creation():
    """Tests MemberUDLoad creation."""
    load = MemberUDLoad(id=401, member_id=12, wx=0.0, wy=-500.0, label="Gravity load") # Perpendicular load

    assert load.id == 401 and load.member_id == 12
    assert load.wx == approx(0.0)
    assert load.wy == approx(-500.0)
    assert load.label == "Gravity load"

def test_memberudload_defaults():
    """Tests default values for MemberUDLoad."""
    load = MemberUDLoad(id=402, member_id=13) # No wx, wy specified
    assert load.wx == approx(0.0)
    assert load.wy == approx(0.0)

def test_memberudload_repr_and_str():
    """Tests MemberUDLoad string representations."""
    load = MemberUDLoad(id=403, member_id=14, wx=50, wy=10, label="Wind + SelfWeight")

    expected_repr = "MemberUDLoad(id=403, member_id=14, wx=50.0, wy=10.0, label='Wind + SelfWeight')"
    expected_str = "MemberUDLoad 403 on Member 14: wx=50 N/m, wy=10 N/m (Wind + SelfWeight)"

    assert repr(load) == expected_repr
    assert str(load) == expected_str

def test_memberudload_invalid_inputs():
    """Tests invalid inputs for MemberUDLoad."""
    # Invalid ID / member_id / label (inherited checks)
    with pytest.raises(TypeError): MemberUDLoad(id="L1", member_id=1)
    with pytest.raises(TypeError): MemberUDLoad(id=1, member_id="M1")
    with pytest.raises(TypeError): MemberUDLoad(id=1, member_id=1, label=None)
    # Invalid numeric types
    with pytest.raises(TypeError, match="MemberUDLoad components .* must be numeric"):
        MemberUDLoad(id=1, member_id=1, wx="a")
    with pytest.raises(TypeError, match="MemberUDLoad components .* must be numeric"):
        MemberUDLoad(id=1, member_id=1, wy=None)


# --- Load Equality Across Types ---

def test_load_equality_across_types():
    """Tests that loads compare based on ID, even if types differ."""
    load1 = NodalLoad(id=500, node_id=1, fx=100)
    load2 = MemberPointLoad(id=500, member_id=1, px=0, py=-100, position=1.0) # Same ID
    load3 = NodalLoad(id=501, node_id=2, fy=200)

    assert load1 == load2 # Equal based on ID only
    assert load1 != load3
    assert load2 != load3
    assert hash(load1) == hash(load2)
    assert hash(load1) != hash(load3)

    load_set = {load1, load2, load3}
    assert len(load_set) == 2 # Only two unique IDs