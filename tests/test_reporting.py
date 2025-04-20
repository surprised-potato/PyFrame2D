# tests/test_io.py

import pytest
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import math # For checking radians conversion if needed

# --- Use Direct Imports assuming run from project root ---
# Try importing all potentially needed classes
try:
    from core.model import (StructuralModel, Node, Material, SectionProfile,
                           RectangularProfile, SquareProfile, IBeamProfile,
                           Member, Support, Load, NodalLoad, MemberLoad,
                           MemberPointLoad, MemberUDLoad)
    from core.analysis import AnalysisResults # Import results class
    from project_io.project_files import save_model_to_json, load_model_from_json
    from project_io.reporting import generate_text_report, save_report_to_file # Import reporting functions
    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
     pytest.skip(f"Skipping IO/Reporting tests: Could not import core components: {e}", allow_module_level=True)
     # Define dummy classes if import fails to allow file parsing by pytest collector
     class AnalysisResults: pass
     # ... include other dummies if needed, though skip should prevent test execution ...
# --- End Imports ---

# --- Test Fixture for a Representative Model (Keep as is) ---
@pytest.fixture
def representative_model():
    # ... (fixture definition remains the same) ...
    model = StructuralModel(name="IO Test Structure")
    # Nodes
    n1 = Node(1, 0.0, 0.0)
    n2 = Node(2, 5.0, 0.0)
    n3 = Node(3, 5.0, -3.0)
    n4 = Node(4, 10.0, 0.0)
    model.add_node(n1); model.add_node(n2); model.add_node(n3); model.add_node(n4)
    # Materials
    mat_steel = Material(10, "Steel S275", 205e9); mat_conc = Material(11, "Concrete C30", 33e9)
    model.add_material(mat_steel); model.add_material(mat_conc)
    # Sections
    sec_rect = RectangularProfile(101, "R 200x300", width=0.2, height=0.3); sec_sq = SquareProfile(102, "SHS 150x150", side_length=0.15); sec_ibeam = IBeamProfile(103, "IPE 200", height=0.2, flange_width=0.1, flange_thickness=0.008, web_thickness=0.005)
    model.add_section(sec_rect); model.add_section(sec_sq); model.add_section(sec_ibeam)
    # Members
    mem1 = Member(201, n1, n2, mat_steel, sec_ibeam); mem2 = Member(202, n2, n3, mat_conc, sec_rect); mem3 = Member(203, n2, n4, mat_steel, sec_sq)
    model.add_member(mem1); model.add_member(mem2); model.add_member(mem3)
    # Supports
    sup1 = Support.fixed(n1.id); sup3 = Support.pinned(n3.id); sup4 = Support.roller_x(n4.id)
    model.add_support(sup1); model.add_support(sup3); model.add_support(sup4)
    # Loads
    load_nodal = NodalLoad(301, n2.id, fx=10e3, mz=-5e3, label="Corner Load"); load_pt = MemberPointLoad(302, mem1.id, px=0, py=-15e3, position=2.5, label="Mid Span Pt"); load_udl = MemberUDLoad(303, mem3.id, wx=0, wy=-2e3, label="Snow Load")
    model.add_load(load_nodal); model.add_load(load_pt); model.add_load(load_udl)
    return model

# --- Helper for Deep Comparison (Keep as is) ---
def compare_models(model1: StructuralModel, model2: StructuralModel):
    # ... (comparison logic remains the same) ...
    """Compares two StructuralModel objects attribute by attribute."""
    assert model1.name == model2.name
    assert model1.nodes.keys() == model2.nodes.keys()
    assert model1.materials.keys() == model2.materials.keys()
    assert model1.sections.keys() == model2.sections.keys()
    assert model1.members.keys() == model2.members.keys()
    assert model1.supports.keys() == model2.supports.keys() # Supports keyed by node_id
    assert model1.loads.keys() == model2.loads.keys()

    # Compare Nodes
    for nid in model1.nodes:
        n1, n2 = model1.get_node(nid), model2.get_node(nid)
        assert n1.id == n2.id
        assert_allclose(n1.x, n2.x, atol=1e-9)
        assert_allclose(n1.y, n2.y, atol=1e-9)

    # Compare Materials
    for mid in model1.materials:
        m1, m2 = model1.get_material(mid), model2.get_material(mid)
        assert m1.id == m2.id
        assert m1.name == m2.name
        assert_allclose(m1.E, m2.E, rtol=1e-9)

    # Compare Sections (including type and specific attributes)
    for sid in model1.sections:
        s1, s2 = model1.get_section(sid), model2.get_section(sid)
        assert type(s1) == type(s2)
        assert s1.id == s2.id
        assert s1.name == s2.name
        if isinstance(s1, RectangularProfile):
             assert_allclose(s1.width, s2.width, atol=1e-9)
             assert_allclose(s1.height, s2.height, atol=1e-9)
        elif isinstance(s1, SquareProfile):
             assert_allclose(s1.side_length, s2.side_length, atol=1e-9)
        elif isinstance(s1, IBeamProfile):
             assert_allclose(s1.height, s2.height, atol=1e-9)
             assert_allclose(s1.flange_width, s2.flange_width, atol=1e-9)
             assert_allclose(s1.flange_thickness, s2.flange_thickness, atol=1e-9)
             assert_allclose(s1.web_thickness, s2.web_thickness, atol=1e-9)

    # Compare Members (checking referenced IDs)
    for mid in model1.members:
        m1, m2 = model1.get_member(mid), model2.get_member(mid)
        assert m1.id == m2.id
        assert m1.start_node.id == m2.start_node.id
        assert m1.end_node.id == m2.end_node.id
        assert m1.material.id == m2.material.id
        assert m1.section.id == m2.section.id

    # Compare Supports (checking node ID and constraints)
    for nid in model1.supports:
        sup1, sup2 = model1.get_support(nid), model2.get_support(nid)
        assert sup1.node_id == sup2.node_id
        assert sup1.dx == sup2.dx
        assert sup1.dy == sup2.dy
        assert sup1.rz == sup2.rz

    # Compare Loads (including type and specific attributes)
    for lid in model1.loads:
        l1, l2 = model1.get_load(lid), model2.get_load(lid)
        assert type(l1) == type(l2)
        assert l1.id == l2.id
        assert l1.label == l2.label
        if isinstance(l1, NodalLoad):
            assert l1.node_id == l2.node_id
            assert_allclose(l1.fx, l2.fx, atol=1e-9)
            assert_allclose(l1.fy, l2.fy, atol=1e-9)
            assert_allclose(l1.mz, l2.mz, atol=1e-9)
        elif isinstance(l1, MemberPointLoad):
             assert l1.member_id == l2.member_id
             assert_allclose(l1.px, l2.px, atol=1e-9)
             assert_allclose(l1.py, l2.py, atol=1e-9)
             assert_allclose(l1.position, l2.position, atol=1e-9)
        elif isinstance(l1, MemberUDLoad):
             assert l1.member_id == l2.member_id
             assert_allclose(l1.wx, l2.wx, atol=1e-9)
             assert_allclose(l1.wy, l2.wy, atol=1e-9)

# === Test Cases for File IO (Keep as is) ===
def test_save_load_round_trip(representative_model):
    # ... (test case remains the same) ...
    """Tests saving a model to JSON and loading it back."""
    original_model = representative_model

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "round_trip_test.json"

        # Save
        save_model_to_json(original_model, str(file_path))
        assert file_path.exists()

        # Load
        loaded_model = load_model_from_json(str(file_path))

        # Compare
        assert isinstance(loaded_model, StructuralModel)
        compare_models(original_model, loaded_model)

def test_load_non_existent_file():
    # ... (test case remains the same) ...
    """Tests loading a file that does not exist."""
    non_existent_path = "non_existent_model_file.json"
    # Ensure it doesn't exist from previous runs
    if os.path.exists(non_existent_path):
         os.remove(non_existent_path)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model_from_json(non_existent_path)

def test_load_invalid_json():
    # ... (test case remains the same) ...
    """Tests loading a file with invalid JSON syntax."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "invalid_json.json"
        # Write invalid JSON (e.g., trailing comma)
        with open(file_path, 'w') as f:
            f.write('{"nodes": [{"id": 1, "x": 0, "y": 0,}],}') # Extra comma

        with pytest.raises(json.JSONDecodeError):
             load_model_from_json(str(file_path))

def test_load_missing_attribute(representative_model):
    # ... (test case remains the same) ...
    """Tests loading data missing a required attribute (e.g., node 'x')."""
    model_data = save_model_to_json.__globals__['_serialize_model'](representative_model) # Access private serialize func for test
    # Modify data: remove 'x' from the first node
    if model_data.get("nodes"):
        del model_data["nodes"][0]['x']
    else:
        pytest.fail("Test setup error: No nodes found in serialized data.")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "missing_attr.json"
        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=4)

        with pytest.raises(ValueError, match="Missing required key .* 'x'"):
            load_model_from_json(str(file_path))

def test_load_invalid_data_type(representative_model):
    # ... (test case remains the same) ...
    """Tests loading data with an incorrect data type (e.g., node ID as string)."""
    model_data = save_model_to_json.__globals__['_serialize_model'](representative_model)
    # Modify data: change first node ID to string
    if model_data.get("nodes"):
        model_data["nodes"][0]['id'] = "Node1"
    else:
        pytest.fail("Test setup error: No nodes found in serialized data.")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "invalid_type.json"
        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=4)

        with pytest.raises(ValueError, match="Invalid data format or value"): # Catches TypeError from Node init
            load_model_from_json(str(file_path))

def test_load_unknown_type(representative_model):
    # ... (test case remains the same) ...
        """Tests loading data with an unrecognized __type__ for section/load."""
        model_data = save_model_to_json.__globals__['_serialize_model'](representative_model)
        # Modify data: change first section type
        if model_data.get("sections"):
            model_data["sections"][0]['__type__'] = "CircularSection" # Assume this type doesn't exist
        else:
            pytest.fail("Test setup error: No sections found in serialized data.")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "unknown_type.json"
            with open(file_path, 'w') as f:
                json.dump(model_data, f, indent=4)

            # --- Expect ValueError (from wrapper) but match original TypeError message ---
            with pytest.raises(ValueError, match="Unknown SectionProfile type 'CircularSection'"):
            # --- End Correction ---
                load_model_from_json(str(file_path))

# === Test Cases for Reporting ===

@pytest.fixture
def sample_success_results():
    """Creates a sample successful AnalysisResults object."""
    displacements = {
        1: (0.0, 0.0, 0.0),
        2: (0.001, -0.005, 0.0001),
        3: (0.0015, -0.0045, -0.0002),
    }
    reactions = {
        1: (1000.0, 15000.0, 2500.0), # Node 1 is fixed
        # Node 3 might be pinned, reactions depend on constraints
    }
    forces = {
        101: np.array([[1000.0], [-500.0], [0.0], [-1000.0], [500.0], [2500.0]]), # Mem 101
        102: np.array([[-1000.0], [750.0], [2500.0], [1000.0], [-750.0], [-1500.0]]), # Mem 102
    }
    # Optional diagram data (simple example)
    x_vals = np.linspace(0, 5, 3).reshape(-1, 1) # Shape (3,1)
    bmd_data = {
        101: np.hstack((x_vals, np.array([0, 1250, 2500]).reshape(-1, 1))), # Shape (3,2)
        102: np.hstack((x_vals, np.array([2500, 1000, -1500]).reshape(-1, 1)))
    }

    return AnalysisResults(
        status="Success",
        model_name="Sample Report Model",
        nodal_displacements=displacements,
        support_reactions=reactions,
        member_end_forces=forces,
        member_bmd_data=bmd_data
        # afd/sfd left as None
    )

@pytest.fixture
def sample_failed_results():
    """Creates a sample failed AnalysisResults object."""
    return AnalysisResults(
        status="Singular Matrix",
        message="Structure may be unstable.",
        model_name="Failed Model"
    )

@pytest.fixture
def sample_model_for_header():
     """Creates a minimal model just for header testing."""
     model = StructuralModel(name="Header Test Model")
     model.add_node(Node(1,0,0))
     model.add_node(Node(2,1,0))
     model.add_material(Material(1,"M",1))
     model.add_section(RectangularProfile(1,"S",1,1))
     model.add_member(Member(1, model.get_node(1), model.get_node(2), model.get_material(1), model.get_section(1)))
     return model

# Optional: Add a separate test specifically for the header generation
def test_report_generation_header(sample_success_results, sample_model_for_header):
    """Tests the Model Summary header generation."""
    results = sample_success_results
    model = sample_model_for_header
    report_str = generate_text_report(results, model)

    assert "Model Summary:" in report_str
    assert "Nodes: 2" in report_str
    assert "Materials: 1" in report_str
    assert "Sections: 1" in report_str
    assert "Members: 1" in report_str
    assert "Supports: 0" in report_str # Based on sample_model_for_header fixture
    assert "Loads: 0" in report_str    # Based on sample_model_for_header fixture
def test_report_generation_success(sample_success_results): # Removed sample_model_for_header
    """Tests generating report for a successful analysis (data sections only)."""
    results = sample_success_results
    # model = sample_model_for_header

    report_str = generate_text_report(results) # Generate WITHOUT model context

    # Check key sections and content
    assert "Analysis Report for Model: Sample Report Model" in report_str
    assert "Status: Success" in report_str

    # Check Nodal Displacements Section
    assert "NODAL DISPLACEMENTS" in report_str
    assert "Node ID    DX (m)          DY (m)          RZ (rad)" in report_str
    # Use regex or careful string formatting to check lines robustly
    # Check presence of key parts, avoiding strict whitespace matching
    assert "1          0.000000e+00" in report_str # Check start of line 1
    assert "2          1.000000e-03" in report_str # Check start of line 2
    assert "3          1.500000e-03" in report_str # Check start of line 3
    assert f"{0.0001:<15.6e}" in report_str # Check last value of line 2

    # Check Support Reactions Section
    assert "SUPPORT REACTIONS" in report_str
    assert "Node ID    RX (N)          RY (N)          MZ (Nm)" in report_str
    assert "1          1.000000e+03" in report_str # Check start of reaction line 1
    assert f"{2500.0:<15.6e}" in report_str # Check last value of reaction line 1

    # Check Member End Forces Section
    assert "MEMBER END FORCES" in report_str
    assert "Member ID    Pxi (N)      Pyi (N)      Mzi (Nm)" in report_str # Check header
    f101 = results.member_end_forces[101].flatten()
    f102 = results.member_end_forces[102].flatten()
    assert f"{101:<12} {f101[0]:<12.4e}" in report_str # Check start of line 101
    assert f"{f101[5]:<12.4e}" in report_str # Check end of line 101
    assert f"{102:<12} {f102[0]:<12.4e}" in report_str # Check start of line 102
    assert f"{f102[5]:<12.4e}" in report_str # Check end of line 102

    # Check Diagram Data Section
    assert "DIAGRAM DATA" in report_str
    assert "(AFD), 0 (SFD), 2 (BMD) members." in report_str # Check summary line content


def test_report_generation_failure(sample_failed_results):
    """Tests generating report for a failed analysis."""
    results = sample_failed_results
    report_str = generate_text_report(results) # No model context needed

    assert "Analysis Report for Model: Failed Model" in report_str
    assert "Status: Singular Matrix" in report_str
    assert "Message: Structure may be unstable." in report_str
    assert "Analysis did not complete successfully." in report_str
    assert "NODAL DISPLACEMENTS" in report_str # Header IS present
    assert "(Nodal displacements not available)" in report_str
    assert "SUPPORT REACTIONS" in report_str # Header IS present
    assert "(Support reactions not available)" in report_str
    assert "MEMBER END FORCES" in report_str # Header IS present
    assert "(Member end forces not available)" in report_str
    assert "DIAGRAM DATA" in report_str # Header IS present
    assert "(Diagram data not calculated or available)" in report_str


def test_save_report_to_file(sample_success_results, tmp_path):
    """Tests saving the generated report to a file."""
    results = sample_success_results
    report_str = generate_text_report(results)
    file_path = tmp_path / "test_report_output.txt"

    # Save the report
    save_report_to_file(report_str, str(file_path))

    # Verify file exists and content matches
    assert file_path.exists()
    content_read = file_path.read_text(encoding='utf-8')
    assert content_read == report_str