# io/project_files.py

import json
import os
from typing import Type, Dict, Any

# Import necessary model classes from core module
# --- Change to Absolute Import ---
try:
    # Import directly from packages assuming project root is in path
    from core.model import (StructuralModel, Node, Material, SectionProfile,
                             RectangularProfile, SquareProfile, IBeamProfile,
                             Member, Support, Load, NodalLoad, MemberLoad,
                             MemberPointLoad, MemberUDLoad)
    # Mapping from type name string to actual class (for deserialization)
    SECTION_TYPE_MAP: Dict[str, Type[SectionProfile]] = {
        "RectangularProfile": RectangularProfile,
        "SquareProfile": SquareProfile,
        "IBeamProfile": IBeamProfile,
    }
    LOAD_TYPE_MAP: Dict[str, Type[Load]] = {
        "NodalLoad": NodalLoad,
        "MemberPointLoad": MemberPointLoad,
        "MemberUDLoad": MemberUDLoad,
    }
    CORE_MODEL_AVAILABLE = True
except ImportError as e:
    # Keep the except block for robustness, but hopefully it won't trigger now
    print(f"Warning: Could not import core model classes into project_files.py using absolute import: {e}")
    CORE_MODEL_AVAILABLE = False
    # Define dummy classes/maps if import fails
    class StructuralModel: pass
    class Node: pass
    # ... etc ... keep dummy classes
    SECTION_TYPE_MAP = {}
    LOAD_TYPE_MAP = {}
# --- End Import Change ---
# Define a version for the file format
FILE_FORMAT_VERSION = "1.0"

# --- Serialization (Save) ---

def _serialize_model(model: StructuralModel) -> Dict[str, Any]:
    """Converts a StructuralModel object into a JSON-serializable dictionary."""
    if not CORE_MODEL_AVAILABLE:
        raise RuntimeError("Core model classes not available for serialization.")

    model_data: Dict[str, Any] = {
        "file_format_version": FILE_FORMAT_VERSION,
        "model_name": model.name,
        "nodes": [],
        "materials": [],
        "sections": [],
        "members": [],
        "supports": [],
        "loads": [],
    }

    # Serialize Nodes
    for node in model.nodes.values():
        model_data["nodes"].append({
            "id": node.id,
            "x": node.x,
            "y": node.y,
        })

    # Serialize Materials
    for mat in model.materials.values():
        model_data["materials"].append({
            "id": mat.id,
            "name": mat.name,
            "E": mat.E, # Store E in base unit (Pa)
        })

    # Serialize Sections (including type information)
    for sec in model.sections.values():
        sec_data = {
            "id": sec.id,
            "name": sec.name,
            "__type__": sec.__class__.__name__ # Store class name
        }
        # Add type-specific attributes
        if isinstance(sec, RectangularProfile):
            sec_data.update({"width": sec.width, "height": sec.height})
        elif isinstance(sec, SquareProfile):
            sec_data.update({"side_length": sec.side_length})
        elif isinstance(sec, IBeamProfile):
            sec_data.update({
                "height": sec.height,
                "flange_width": sec.flange_width,
                "flange_thickness": sec.flange_thickness,
                "web_thickness": sec.web_thickness,
            })
        # Add elif blocks for other SectionProfile subclasses here
        else:
            print(f"Warning: Skipping unknown SectionProfile type during save: {type(sec)}")
            continue # Skip unknown types
        model_data["sections"].append(sec_data)

    # Serialize Members (referencing components by ID)
    for mem in model.members.values():
        model_data["members"].append({
            "id": mem.id,
            "start_node_id": mem.start_node.id,
            "end_node_id": mem.end_node.id,
            "material_id": mem.material.id,
            "section_id": mem.section.id,
            # Add serialization for other member properties (e.g., releases) here later
        })

    # Serialize Supports
    for sup in model.supports.values():
        model_data["supports"].append({
            "node_id": sup.node_id,
            "dx": sup.dx,
            "dy": sup.dy,
            "rz": sup.rz,
        })

    # Serialize Loads (including type information)
    for load in model.loads.values():
        load_data = {
            "id": load.id,
            "label": load.label,
            "__type__": load.__class__.__name__ # Store class name
        }
        # Add type-specific attributes
        if isinstance(load, NodalLoad):
            load_data.update({
                "node_id": load.node_id,
                "fx": load.fx,
                "fy": load.fy,
                "mz": load.mz,
            })
        elif isinstance(load, MemberPointLoad):
             load_data.update({
                "member_id": load.member_id,
                "px": load.px,
                "py": load.py,
                "position": load.position,
            })
        elif isinstance(load, MemberUDLoad):
             load_data.update({
                "member_id": load.member_id,
                "wx": load.wx,
                "wy": load.wy,
            })
        # Add elif blocks for other Load subclasses here
        else:
            print(f"Warning: Skipping unknown Load type during save: {type(load)}")
            continue # Skip unknown types
        model_data["loads"].append(load_data)

    return model_data


def save_model_to_json(model: StructuralModel, file_path: str):
    """
    Saves the StructuralModel object to a JSON file.

    Args:
        model: The StructuralModel instance to save.
        file_path: The full path to the output JSON file.

    Raises:
        IOError: If there's an error writing to the file.
        TypeError: If the model contains unserializable components.
        RuntimeError: If core model classes are unavailable.
    """
    print(f"Attempting to save model '{model.name}' to {file_path}...")
    try:
        model_data = _serialize_model(model)
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4, ensure_ascii=False)
        print("Model saved successfully.")
    except (IOError, OSError) as e:
        raise IOError(f"Error writing model file '{file_path}': {e}") from e
    except (TypeError, ValueError) as e:
        # Catch errors during serialization itself
        raise TypeError(f"Error serializing model data: {e}") from e
    except Exception as e: # Catch unexpected errors
        raise RuntimeError(f"An unexpected error occurred during saving: {e}") from e


# --- Deserialization (Load) ---

def _deserialize_model(model_data: Dict[str, Any]) -> StructuralModel:
    """Reconstructs a StructuralModel object from a dictionary."""
    if not CORE_MODEL_AVAILABLE:
        raise RuntimeError("Core model classes not available for deserialization.")

    # Check version (optional, but good practice)
    version = model_data.get("file_format_version")
    if version != FILE_FORMAT_VERSION:
         print(f"Warning: Loading file with version {version}, expected {FILE_FORMAT_VERSION}. Compatibility not guaranteed.")

    model = StructuralModel(name=model_data.get("model_name", "Loaded Model"))

    # Temporary storage for error reporting if needed
    processed_ids: Dict[str, set] = {"nodes": set(), "materials": set(), "sections": set(), "members": set(), "supports": set(), "loads": set()}

    try:
        # 1. Deserialize Nodes
        for node_data in model_data.get("nodes", []):
            node = Node(id=node_data['id'], x=node_data['x'], y=node_data['y'])
            model.add_node(node)
            processed_ids["nodes"].add(node.id)

        # 2. Deserialize Materials
        for mat_data in model_data.get("materials", []):
            mat = Material(id=mat_data['id'], name=mat_data['name'], youngs_modulus=mat_data['E'])
            model.add_material(mat)
            processed_ids["materials"].add(mat.id)

        # 3. Deserialize Sections (using type map)
        for sec_data in model_data.get("sections", []):
            sec_type_name = sec_data.get("__type__")
            sec_class = SECTION_TYPE_MAP.get(sec_type_name)
            if not sec_class:
                raise TypeError(f"Unknown SectionProfile type '{sec_type_name}' encountered in file.")

            sec_id = sec_data['id']
            sec_name = sec_data['name']
            # Instantiate correct subclass based on type
            if sec_class is RectangularProfile:
                 sec = sec_class(id=sec_id, name=sec_name, width=sec_data['width'], height=sec_data['height'])
            elif sec_class is SquareProfile:
                 sec = sec_class(id=sec_id, name=sec_name, side_length=sec_data['side_length'])
            elif sec_class is IBeamProfile:
                 sec = sec_class(id=sec_id, name=sec_name, height=sec_data['height'],
                                 flange_width=sec_data['flange_width'], flange_thickness=sec_data['flange_thickness'],
                                 web_thickness=sec_data['web_thickness'])
            # Add elif blocks for other SectionProfile subclasses here
            else:
                 # Should have been caught by map lookup, but as safeguard:
                 raise TypeError(f"Logic error: Section class '{sec_type_name}' found in map but not handled.")

            model.add_section(sec)
            processed_ids["sections"].add(sec.id)

        # 4. Deserialize Members (linking components)
        for mem_data in model_data.get("members", []):
            # Get referenced objects from the partially built model
            start_node = model.get_node(mem_data['start_node_id'])
            end_node = model.get_node(mem_data['end_node_id'])
            material = model.get_material(mem_data['material_id'])
            section = model.get_section(mem_data['section_id'])
            # Instantiate member
            mem = Member(id=mem_data['id'], start_node=start_node, end_node=end_node,
                         material=material, section=section)
            # Add deserialization for other member properties here later
            model.add_member(mem)
            processed_ids["members"].add(mem.id)

        # 5. Deserialize Supports
        for sup_data in model_data.get("supports", []):
            # Check node exists (add_support doesn't check anymore)
            if sup_data['node_id'] not in model.nodes:
                 raise ValueError(f"Support references node ID {sup_data['node_id']} which was not loaded.")
            sup = Support(node_id=sup_data['node_id'], dx=sup_data['dx'], dy=sup_data['dy'], rz=sup_data['rz'])
            model.add_support(sup)
            processed_ids["supports"].add(sup.node_id) # Supports keyed by node_id

        # 6. Deserialize Loads (using type map)
        for load_data in model_data.get("loads", []):
            load_type_name = load_data.get("__type__")
            load_class = LOAD_TYPE_MAP.get(load_type_name)
            if not load_class:
                 raise TypeError(f"Unknown Load type '{load_type_name}' encountered in file.")

            load_id = load_data['id']
            load_label = load_data.get('label', "")
            # Instantiate correct subclass
            if load_class is NodalLoad:
                if load_data['node_id'] not in model.nodes:
                    raise ValueError(f"NodalLoad {load_id} references node ID {load_data['node_id']} which was not loaded.")
                load = load_class(id=load_id, node_id=load_data['node_id'],
                                  fx=load_data['fx'], fy=load_data['fy'], mz=load_data['mz'],
                                  label=load_label)
            elif load_class is MemberPointLoad:
                if load_data['member_id'] not in model.members:
                     raise ValueError(f"MemberPointLoad {load_id} references member ID {load_data['member_id']} which was not loaded.")
                load = load_class(id=load_id, member_id=load_data['member_id'],
                                  px=load_data['px'], py=load_data['py'], position=load_data['position'],
                                  label=load_label)
            elif load_class is MemberUDLoad:
                if load_data['member_id'] not in model.members:
                     raise ValueError(f"MemberUDLoad {load_id} references member ID {load_data['member_id']} which was not loaded.")
                load = load_class(id=load_id, member_id=load_data['member_id'],
                                  wx=load_data.get('wx', 0.0), wy=load_data.get('wy', 0.0), # Use .get for defaults
                                  label=load_label)
            # Add elif blocks for other Load subclasses here
            else:
                raise TypeError(f"Logic error: Load class '{load_type_name}' found in map but not handled.")

            model.add_load(load)
            processed_ids["loads"].add(load.id)

    except KeyError as e:
        # More specific error for missing required keys or missing referenced IDs
        raise ValueError(f"Error during deserialization: Missing required key or referenced ID - {e}") from e
    except (TypeError, ValueError) as e:
        # Catch errors from class constructors or add methods
        raise ValueError(f"Error during deserialization: Invalid data format or value - {e}") from e
    except Exception as e: # Catch unexpected errors
        raise RuntimeError(f"An unexpected error occurred during deserialization: {e}") from e

    return model


def load_model_from_json(file_path: str) -> StructuralModel:
    """
    Loads a StructuralModel object from a JSON file.

    Args:
        file_path: The full path to the input JSON file.

    Returns:
        The reconstructed StructuralModel instance.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there's an error reading the file.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the JSON data is structurally valid but contains inconsistent
                    or invalid model data (e.g., missing keys, wrong types, bad refs).
        RuntimeError: If core model classes are unavailable or for unexpected errors.
    """
    print(f"Attempting to load model from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: '{file_path}'")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        model = _deserialize_model(model_data)
        print("Model loaded successfully.")
        return model

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from file '{file_path}': {e.msg}", e.doc, e.pos) from e
    except (IOError, OSError) as e:
        raise IOError(f"Error reading model file '{file_path}': {e}") from e
    # ValueError, TypeError, RuntimeError are raised by _deserialize_model


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    if not CORE_MODEL_AVAILABLE:
        print("Cannot run example: Core model classes failed to import.")
    else:
        print("\n--- Testing Save/Load ---")
        # Create a simple model instance
        example_model = StructuralModel("Save/Load Test Model")
        n1 = Node(1, 0, 0)
        n2 = Node(2, 3, 4) # Length 5
        example_model.add_node(n1)
        example_model.add_node(n2)
        mat = Material(10, "Steel", 200e9)
        example_model.add_material(mat)
        sec = SquareProfile(20, "SHS 100", 0.1)
        example_model.add_section(sec)
        mem = Member(30, n1, n2, mat, sec)
        example_model.add_member(mem)
        sup = Support.pinned(1)
        example_model.add_support(sup)
        load = NodalLoad(40, 2, fx=1000, label="Wind")
        example_model.add_load(load)

        print("Original Model:", example_model)

        file_path = "test_model_save.json"

        # Save
        try:
            save_model_to_json(example_model, file_path)
        except Exception as e:
            print(f"Error saving: {e}")

        # Load
        loaded_model = None
        try:
            loaded_model = load_model_from_json(file_path)
            print("Loaded Model:", loaded_model)

            # Basic comparison (more thorough checks needed in tests)
            if loaded_model:
                print("\nComparison:")
                print(f"Names match: {example_model.name == loaded_model.name}")
                print(f"Node count match: {len(example_model.nodes) == len(loaded_model.nodes)}")
                print(f"Member count match: {len(example_model.members) == len(loaded_model.members)}")
                # Compare specific item?
                original_node2 = example_model.get_node(2)
                loaded_node2 = loaded_model.get_node(2)
                print(f"Node 2 coords match: {original_node2.get_coords() == loaded_node2.get_coords()}")

        except Exception as e:
            print(f"Error loading: {e}")

        # Clean up test file
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        #     print(f"Cleaned up {file_path}")