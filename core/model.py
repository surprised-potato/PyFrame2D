# core/model.py

import math # Might be needed later for other classes
from pytest import approx # type: ignore # Useful for potential internal float comparisons if needed
from abc import ABC, abstractmethod # Import for Abstract Base Classes
try:
    from units.units import parse_value_unit, format_value_unit
    UNITS_ENABLED = True
except ImportError:
    print("Warning: 'units' module not found or import error. Unit parsing/formatting disabled.")
    UNITS_ENABLED = False
    # Define dummy functions if units module is unavailable
    def parse_value_unit(input_str, expected_unit):
        raise NotImplementedError("Units module not available.")
    def format_value_unit(value, base_unit_str, precision=3):
        return f"{value} {base_unit_str} (units disabled)"
from typing import Type, Optional 
class Node:
    """
    Represents a node (point) in the structural model.

    Attributes:
        id (int): A unique integer identifier for the node.
        x (float): The global X-coordinate of the node.
        y (float): The global Y-coordinate of the node.
    """
    def __init__(self, id: int, x: float, y: float):
        """
        Initializes a Node object.

        Args:
            id: The unique integer identifier for the node.
            x: The global X-coordinate.
            y: The global Y-coordinate.

        Raises:
            TypeError: If id is not an integer, or if x or y cannot be converted to float.
        """
        if not isinstance(id, int):
            # Check if it's an integer *type* first
             raise TypeError(f"Node ID must be an integer (received: {id}, type: {type(id)}).")
        try:
            self.id = int(id)
        except (ValueError, TypeError):
            raise TypeError(f"Node ID must be an integer (received: {id}, type: {type(id)}).")

        try:
            self.x = float(x)
            self.y = float(y)
        except (ValueError, TypeError):
             # Be more specific about which coordinate failed if possible
            failed_coord = 'x' if not isinstance(x, (int, float)) else 'y'
            failed_val = x if failed_coord == 'x' else y
            raise TypeError(f"Node coordinates (x, y) must be numeric (received {failed_coord}={failed_val}, type: {type(failed_val)}).")


    def get_coords(self) -> tuple[float, float]:
        """
        Returns the node's coordinates as a tuple.

        Returns:
            A tuple containing the (x, y) coordinates.
        """
        return (self.x, self.y)

    def __repr__(self) -> str:
        """Provides an unambiguous string representation of the Node."""
        return f"Node(id={self.id}, x={self.x}, y={self.y})"

    def __str__(self) -> str:
        """Provides a user-friendly string representation of the Node."""
        return f"Node {self.id} @ ({self.x:.3g}, {self.y:.3g})" # Format coords for readability

    def __eq__(self, other) -> bool:
        """
        Checks if two Node objects are equal based on their ID.

        Args:
            other: The object to compare against.

        Returns:
            True if the other object is a Node with the same ID, False otherwise.
            Returns NotImplemented if the comparison is between incompatible types.
        """
        if not isinstance(other, Node):
            return NotImplemented # Important for correct comparison behavior
        return self.id == other.id

    def __ne__(self, other) -> bool:
        """Checks if two Node objects are not equal."""
        # Explicitly handle NotImplemented return from __eq__
        equal = self.__eq__(other)
        if equal is NotImplemented:
            # If comparison isn't implemented, they can't be equal,
            # but in Python's comparison model, != should also indicate this.
            # Returning NotImplemented is often preferred, but for simple inequality,
            # True might be more intuitive if we assume comparison only makes sense
            # between Nodes. Let's stick to the standard pattern:
             return NotImplemented # Propagate NotImplemented for incompatible types
            # Alternative: If you always want Node != NonNode to be True:
            # if not isinstance(other, Node):
            #     return True
            # return self.id != other.id
        # If __eq__ returned True or False, then __ne__ is the opposite.
        return not equal
    def __hash__(self) -> int:
        """
        Computes the hash of the Node based on its ID.

        Allows Nodes to be used in sets and as dictionary keys.
        """
        return hash(self.id)

class Material:
    """
    Represents a material with its physical properties.

    Attributes:
        id (int): A unique integer identifier for the material.
        name (str): A descriptive name for the material (e.g., "Steel S235").
        E (float): Young's Modulus (Modulus of Elasticity) in Pascals (Pa).
                   Stored internally in base SI units (Pa).
    """
    def __init__(self, id: int, name: str, youngs_modulus):
        """
        Initializes a Material object.

        Args:
            id: The unique integer identifier.
            name: The descriptive name.
            youngs_modulus: Young's Modulus. Can be provided as:
                - A numeric value (int or float), assumed to be in Pascals (Pa).
                - A string with units (e.g., "210 GPa", "2.1e11 Pa", "210000 MPa"),
                  which will be parsed using the 'units' module.

        Raises:
            TypeError: If id is not an integer, name is not a string, or
                       youngs_modulus is an invalid type or format.
            ValueError: If name is empty, youngs_modulus is non-positive, or unit
                        parsing fails for string input.
        """
        if not isinstance(id, int):
            raise TypeError(f"Material ID must be an integer (received: {id}, type: {type(id)}).")
        if not isinstance(name, str):
            raise TypeError(f"Material name must be a string (received: {name}, type: {type(name)}).")
        if not name:
            raise ValueError("Material name cannot be empty.")

        self.id = id
        self.name = name.strip() # Store stripped name

        e_value_pa = None
        if isinstance(youngs_modulus, (int, float)):
            e_value_pa = float(youngs_modulus)
        elif isinstance(youngs_modulus, str):
            if not UNITS_ENABLED:
                raise RuntimeError("Cannot parse units from string: 'units' module is disabled.")
            try:
                # Use the units module to parse, expecting Pascals (Pa)
                e_value_pa = parse_value_unit(youngs_modulus, "Pa")
            except (ValueError, TypeError) as e:
                # Catch parsing errors and re-raise with context
                raise ValueError(f"Invalid Young's Modulus string '{youngs_modulus}': {e}") from e
        else:
            raise TypeError(f"Invalid type for Young's Modulus: {type(youngs_modulus)}. Expected number or string.")

        # Validate the numeric value
        if e_value_pa is None or e_value_pa <= 0:
            raise ValueError(f"Young's Modulus must be a positive value (received: {youngs_modulus} -> resulting Pa: {e_value_pa}).")

        self.E = e_value_pa # Store E in base unit (Pa)

    def __repr__(self) -> str:
        """Provides an unambiguous string representation."""
        # Format E back to a common unit like GPa for representation if units are enabled
        e_str = f"{self.E:.4g} Pa"
        if UNITS_ENABLED:
            try:
                 e_str = format_value_unit(self.E, "Pa", precision=4)
            except:
                 pass # Fallback to base unit string if formatting fails
        return f"Material(id={self.id}, name='{self.name}', E='{e_str}')"

    def __str__(self) -> str:
        """Provides a user-friendly string representation."""
        e_str = f"{self.E:.4g} Pa"
        if UNITS_ENABLED:
            try:
                 e_str = format_value_unit(self.E, "Pa", precision=4)
            except:
                 pass
        return f"Material {self.id}: {self.name} (E = {e_str})"

    def __eq__(self, other) -> bool:
        """Checks equality based on material ID."""
        if not isinstance(other, Material):
            return NotImplemented
        return self.id == other.id

    def __ne__(self, other) -> bool:
        """Checks inequality."""
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __hash__(self) -> int:
        """Computes hash based on material ID."""
        return hash(self.id)


# --- Section Profile Base Class ---

class SectionProfile(ABC):
    """
    Abstract Base Class for cross-section profiles.

    Defines the common interface for all section types used in members.

    Attributes:
        id (int): Unique integer identifier for the section profile.
        name (str): Descriptive name (e.g., "IPE 300", "SHS 100x100x5").
    """
    def __init__(self, id: int, name: str):
        """
        Initializes the base section profile.

        Args:
            id: Unique integer identifier.
            name: Descriptive name.

        Raises:
            TypeError: If id is not an integer or name is not a string.
            ValueError: If name is empty.
        """
        if not isinstance(id, int):
            raise TypeError(f"SectionProfile ID must be an integer (received: {id}, type: {type(id)}).")
        if not isinstance(name, str):
            raise TypeError(f"SectionProfile name must be a string (received: {name}, type: {type(name)}).")
        if not name:
            raise ValueError("SectionProfile name cannot be empty.")
        self.id = id
        self.name = name.strip()

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Calculates the cross-sectional area (A).

        Returns:
            Area in square meters (m^2).
        """
        pass # pragma: no cover -- Abstract method

    @property
    @abstractmethod
    def moment_of_inertia(self) -> float:
        """
        Calculates the second moment of area (I) about the relevant axis.
        For 2D frame analysis, this is typically the axis perpendicular to the
        plane of bending (often denoted Izz or Ix).

        Returns:
            Moment of inertia in meters to the fourth power (m^4).
        """
        pass # pragma: no cover -- Abstract method

    def __repr__(self) -> str:
        """Provides an unambiguous string representation."""
        # Base representation, subclasses should extend this
        return f"{self.__class__.__name__}(id={self.id}, name='{self.name}')"

    def __str__(self) -> str:
        """Provides a user-friendly string representation."""
        return f"Section {self.id}: {self.name} ({self.__class__.__name__})"

    def __eq__(self, other) -> bool:
        """Checks equality based on section ID."""
        if not isinstance(other, SectionProfile):
            return NotImplemented
        # Check type compatibility as well? Maybe not strictly needed if ID is globally unique.
        # Let's assume IDs are unique across all section types for now.
        return self.id == other.id

    def __ne__(self, other) -> bool:
        """Checks inequality."""
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __hash__(self) -> int:
        """Computes hash based on section ID."""
        return hash(self.id)


# --- Concrete Section Profiles ---

class RectangularProfile(SectionProfile):
    """
    Represents a solid rectangular cross-section.

    Attributes:
        id (int): Unique identifier.
        name (str): Descriptive name.
        width (float): Width of the rectangle (dimension parallel to the typical axis of inertia) in meters.
        height (float): Height of the rectangle (dimension perpendicular to the typical axis of inertia) in meters.
    """
    def __init__(self, id: int, name: str, width: float, height: float):
        """
        Initializes a RectangularProfile.

        Args:
            id: Unique integer identifier.
            name: Descriptive name.
            width: Width in meters (must be positive).
            height: Height in meters (must be positive).

        Raises:
            TypeError: If id/name have wrong types, or width/height are not numeric.
            ValueError: If name is empty, or width/height are not positive.
        """
        super().__init__(id, name)
        try:
            w = float(width)
            h = float(height)
        except (ValueError, TypeError):
            raise TypeError("RectangularProfile dimensions (width, height) must be numeric.")

        if w <= 0 or h <= 0:
            raise ValueError("RectangularProfile dimensions (width, height) must be positive.")
        self.width = w
        self.height = h

    @property
    def area(self) -> float:
        """Area A = width * height."""
        return self.width * self.height

    @property
    def moment_of_inertia(self) -> float:
        """Moment of inertia I = width * height^3 / 12 (about axis parallel to width)."""
        return (self.width * self.height**3) / 12.0

    def __repr__(self) -> str:
        """Unambiguous representation including dimensions."""
        return (f"{self.__class__.__name__}(id={self.id}, name='{self.name}', "
                f"width={self.width}, height={self.height})")

    def __str__(self) -> str:
        """User-friendly representation."""
        return (f"Section {self.id}: {self.name} "
                f"(Rectangular {self.width:.3g}m x {self.height:.3g}m)")


class SquareProfile(SectionProfile):
    """
    Represents a solid square cross-section.

    Attributes:
        id (int): Unique identifier.
        name (str): Descriptive name.
        side_length (float): Length of the square's side in meters.
    """
    def __init__(self, id: int, name: str, side_length: float):
        """
        Initializes a SquareProfile.

        Args:
            id: Unique integer identifier.
            name: Descriptive name.
            side_length: Side length in meters (must be positive).

        Raises:
            TypeError: If id/name have wrong types, or side_length is not numeric.
            ValueError: If name is empty, or side_length is not positive.
        """
        super().__init__(id, name)
        try:
            s = float(side_length)
        except (ValueError, TypeError):
            raise TypeError("SquareProfile side_length must be numeric.")

        if s <= 0:
            raise ValueError("SquareProfile side_length must be positive.")
        self.side_length = s

    @property
    def area(self) -> float:
        """Area A = side_length^2."""
        return self.side_length ** 2

    @property
    def moment_of_inertia(self) -> float:
        """Moment of inertia I = side_length^4 / 12."""
        return (self.side_length**4) / 12.0

    def __repr__(self) -> str:
        """Unambiguous representation including dimension."""
        return (f"{self.__class__.__name__}(id={self.id}, name='{self.name}', "
                f"side_length={self.side_length})")

    def __str__(self) -> str:
        """User-friendly representation."""
        return (f"Section {self.id}: {self.name} "
                f"(Square {self.side_length:.3g}m)")


class IBeamProfile(SectionProfile):
    """
    Represents a standard I-beam (or H-beam) cross-section.

    Attributes:
        id (int): Unique identifier.
        name (str): Descriptive name.
        height (float): Overall height of the section (h) in meters.
        flange_width (float): Width of the top and bottom flanges (bf) in meters.
        flange_thickness (float): Thickness of the flanges (tf) in meters.
        web_thickness (float): Thickness of the web (tw) in meters.
    """
    def __init__(self, id: int, name: str, height: float, flange_width: float,
                 flange_thickness: float, web_thickness: float):
        """
        Initializes an IBeamProfile.

        Args:
            id: Unique identifier.
            name: Descriptive name.
            height: Overall height (h) in meters.
            flange_width: Flange width (bf) in meters.
            flange_thickness: Flange thickness (tf) in meters.
            web_thickness: Web thickness (tw) in meters.

        Raises:
            TypeError: If id/name have wrong types, or dimensions are not numeric.
            ValueError: If name is empty, any dimension is not positive, or
                        geometry is invalid (e.g., height <= 2*flange_thickness,
                        flange_width < web_thickness).
        """
        super().__init__(id, name)
        try:
            h = float(height)
            bf = float(flange_width)
            tf = float(flange_thickness)
            tw = float(web_thickness)
        except (ValueError, TypeError):
            raise TypeError("IBeamProfile dimensions must be numeric.")

        if not (h > 0 and bf > 0 and tf > 0 and tw > 0):
            raise ValueError("IBeamProfile dimensions must be positive.")

        # Geometric validation
        if h <= 2 * tf:
            raise ValueError(f"Invalid I-beam geometry: height ({h}) must be greater than 2 * flange_thickness ({2*tf}).")
        if bf < tw:
            raise ValueError(f"Invalid I-beam geometry: flange_width ({bf}) must be greater than or equal to web_thickness ({tw}).")

        self.height = h
        self.flange_width = bf
        self.flange_thickness = tf
        self.web_thickness = tw

    @property
    def area(self) -> float:
        """Area A = 2*(bf*tf) + (h - 2*tf)*tw."""
        web_height = self.height - 2 * self.flange_thickness
        return 2 * (self.flange_width * self.flange_thickness) + (web_height * self.web_thickness)

    @property
    def moment_of_inertia(self) -> float:
        """Moment of inertia about the strong axis (parallel to flanges).
           I = (bf*h^3)/12 - ((bf-tw)*(h-2*tf)^3)/12
        """
        h = self.height
        bf = self.flange_width
        tf = self.flange_thickness
        tw = self.web_thickness
        hw = h - 2*tf # Web height

        # Inertia of the circumscribing rectangle minus inertia of the voids beside the web
        inertia_total_rect = (bf * h**3) / 12.0
        inertia_voids = ((bf - tw) * hw**3) / 12.0 # Combined inertia of the two void areas
        return inertia_total_rect - inertia_voids

    def __repr__(self) -> str:
        """Unambiguous representation including dimensions."""
        return (f"{self.__class__.__name__}(id={self.id}, name='{self.name}', "
                f"h={self.height}, bf={self.flange_width}, tf={self.flange_thickness}, tw={self.web_thickness})")

    def __str__(self) -> str:
        """User-friendly representation."""
        return (f"Section {self.id}: {self.name} "
                f"(I-Beam h={self.height:.3g}m, bf={self.flange_width:.3g}m, "
                f"tf={self.flange_thickness:.3g}m, tw={self.web_thickness:.3g}m)")

class Member:
    """
    Represents a 2D frame member connecting two nodes.

    Attributes:
        id (int): A unique integer identifier for the member.
        start_node (Node): The Node object at the start of the member.
        end_node (Node): The Node object at the end of the member.
        material (Material): The Material object assigned to this member.
        section (SectionProfile): The SectionProfile object assigned to this member.
    """
    def __init__(self, id: int, start_node: Node, end_node: Node, material: Material, section: SectionProfile):
        """
        Initializes a Member object.

        Args:
            id: The unique integer identifier for the member.
            start_node: The starting Node object.
            end_node: The ending Node object.
            material: The Material object for this member.
            section: The SectionProfile object for this member.

        Raises:
            TypeError: If id is not an integer, or if start_node/end_node are not Node objects,
                       or if material is not a Material object, or section is not a SectionProfile object.
            ValueError: If start_node and end_node refer to the same node (zero-length member).
        """
        if not isinstance(id, int):
            raise TypeError(f"Member ID must be an integer (received: {id}, type: {type(id)}).")
        if not isinstance(start_node, Node):
            raise TypeError(f"Member start_node must be a Node object (received: {start_node}, type: {type(start_node)}).")
        if not isinstance(end_node, Node):
            raise TypeError(f"Member end_node must be a Node object (received: {end_node}, type: {type(end_node)}).")
        if not isinstance(material, Material):
             raise TypeError(f"Member material must be a Material object (received: {material}, type: {type(material)}).")
        if not isinstance(section, SectionProfile):
            raise TypeError(f"Member section must be a SectionProfile object (received: {section}, type: {type(section)}).")

         # --- Corrected Zero-Length Check ---
        # Check coordinates *first* to detect zero length regardless of node IDs
        if start_node.get_coords() == end_node.get_coords():
            # Provide coordinates in the error message for clarity
            coords = start_node.get_coords()
            raise ValueError(f"Member start and end node coordinates are identical ({coords}). Member must have non-zero length.")
        # --- End Correction ---
        self.id = id
        self.start_node = start_node
        self.end_node = end_node
        self.material = material
        self.section = section
        # Future attributes like end_releases can be added here

    @property
    def length(self) -> float:
        """
        Calculates the length of the member based on its node coordinates.

        Returns:
            The member length in meters.
        """
        x1, y1 = self.start_node.get_coords()
        x2, y2 = self.end_node.get_coords()
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx**2 + dy**2)

    @property
    def angle(self) -> float:
        """
        Calculates the angle of the member relative to the global positive X-axis.

        Returns:
            The angle in radians, typically in the range (-pi, pi].
        """
        x1, y1 = self.start_node.get_coords()
        x2, y2 = self.end_node.get_coords()
        dx = x2 - x1
        dy = y2 - y1
        # atan2 handles quadrants correctly and the case dx=0
        # If dx=0 and dy=0 (zero length), math.atan2(0, 0) returns 0.0.
        # We already check for zero length in __init__, so this case shouldn't be reached
        # under normal circumstances if __init__ validation is correct.
        return math.atan2(dy, dx)

    # --- Methods providing material/section properties ---
    # Convenience methods to avoid accessing nested objects everywhere

    @property
    def E(self) -> float:
        """Young's Modulus (E) of the member's material (in Pa)."""
        return self.material.E

    @property
    def A(self) -> float:
        """Cross-sectional area (A) of the member's section (in m^2)."""
        return self.section.area

    @property
    def I(self) -> float:
        """Moment of inertia (I) of the member's section (in m^4)."""
        return self.section.moment_of_inertia

    # --- Standard Python methods ---

    def __repr__(self) -> str:
        """Provides an unambiguous string representation."""
        return (f"Member(id={self.id}, start_node=Node(id={self.start_node.id}), "
                f"end_node=Node(id={self.end_node.id}), material=Material(id={self.material.id}), "
                f"section={self.section.__class__.__name__}(id={self.section.id}))")

    def __str__(self) -> str:
        """Provides a user-friendly string representation."""
        return (f"Member {self.id} (Nodes: {self.start_node.id} -> {self.end_node.id}, "
                f"Material: {self.material.id}, Section: {self.section.id})")

    def __eq__(self, other) -> bool:
        """Checks equality based on member ID."""
        if not isinstance(other, Member):
            return NotImplemented
        return self.id == other.id

    def __ne__(self, other) -> bool:
        """Checks inequality."""
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __hash__(self) -> int:
        """Computes hash based on member ID."""
        return hash(self.id)

class Support:
    """
    Represents a support condition applied to a node.

    Defines constraints on the node's Degrees of Freedom (DOFs).

    Attributes:
        node_id (int): The ID of the node where the support is applied.
        dx (bool): True if translation along the global X-axis is restrained.
        dy (bool): True if translation along the global Y-axis is restrained.
        rz (bool): True if rotation about the global Z-axis is restrained.
    """
    def __init__(self, node_id: int, dx: bool, dy: bool, rz: bool):
        """
        Initializes a Support object.

        Args:
            node_id: The ID of the node to support.
            dx: Restraint against global X translation.
            dy: Restraint against global Y translation.
            rz: Restraint against global Z rotation.

        Raises:
            TypeError: If node_id is not an integer or dx/dy/rz are not booleans.
        """
        if not isinstance(node_id, int):
            raise TypeError(f"Support node_id must be an integer (received: {node_id}).")
        if not all(isinstance(val, bool) for val in [dx, dy, rz]):
            raise TypeError("Support restraints (dx, dy, rz) must be boolean values.")

        self.node_id = node_id
        self.dx = dx
        self.dy = dy
        self.rz = rz

    @classmethod
    def fixed(cls, node_id: int):
        """Creates a fully fixed support (restrains dx, dy, rz)."""
        return cls(node_id, dx=True, dy=True, rz=True)

    @classmethod
    def pinned(cls, node_id: int):
        """Creates a pinned support (restrains dx, dy; allows rz)."""
        return cls(node_id, dx=True, dy=True, rz=False)

    @classmethod
    def roller_x(cls, node_id: int):
        """Creates a roller support allowing X-translation (restrains dy; allows dx, rz)."""
        return cls(node_id, dx=False, dy=True, rz=False)

    @classmethod
    def roller_y(cls, node_id: int):
        """Creates a roller support allowing Y-translation (restrains dx; allows dy, rz)."""
        return cls(node_id, dx=True, dy=False, rz=False)

    @property
    def is_dx_restrained(self) -> bool:
        """Returns True if DX translation is restrained."""
        return self.dx

    @property
    def is_dy_restrained(self) -> bool:
        """Returns True if DY translation is restrained."""
        return self.dy

    @property
    def is_rz_restrained(self) -> bool:
        """Returns True if RZ rotation is restrained."""
        return self.rz

    def __repr__(self) -> str:
        """Unambiguous representation."""
        return f"Support(node_id={self.node_id}, dx={self.dx}, dy={self.dy}, rz={self.rz})"

    def __str__(self) -> str:
        """User-friendly representation."""
        constraints = []
        if self.dx: constraints.append("DX")
        if self.dy: constraints.append("DY")
        if self.rz: constraints.append("RZ")
        constraint_str = "+".join(constraints) if constraints else "Free"
        # Determine common type name
        type_str = ""
        if self.dx and self.dy and self.rz: type_str = " (Fixed)"
        elif self.dx and self.dy and not self.rz: type_str = " (Pinned)"
        elif not self.dx and self.dy and not self.rz: type_str = " (Roller X)"
        elif self.dx and not self.dy and not self.rz: type_str = " (Roller Y)"
        return f"Support @ Node {self.node_id}: Restrains {constraint_str}{type_str}"

    def __eq__(self, other) -> bool:
        """Equality based on node_id. Assumes only one support per node."""
        if not isinstance(other, Support):
            return NotImplemented
        return self.node_id == other.node_id

    def __ne__(self, other) -> bool:
        """Checks inequality."""
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __hash__(self) -> int:
        """Hash based on node_id."""
        return hash(self.node_id)


# --- Load Base Class ---

class Load(ABC):
    """
    Abstract Base Class for all load types.

    Attributes:
        id (int): Unique integer identifier for the load.
        label (str): Optional descriptive label for the load.
    """
    def __init__(self, id: int, label: str = ""):
        """
        Initializes the base load.

        Args:
            id: Unique integer identifier.
            label: Optional descriptive label.

        Raises:
            TypeError: If id is not an integer or label is not a string.
        """
        if not isinstance(id, int):
            raise TypeError(f"Load ID must be an integer (received: {id}).")
        if not isinstance(label, str):
            raise TypeError(f"Load label must be a string (received: {label}).")
        self.id = id
        self.label = label.strip()

    def __eq__(self, other) -> bool:
        """Equality based on load ID."""
        if not isinstance(other, Load):
            return NotImplemented
        return self.id == other.id

    def __ne__(self, other) -> bool:
        """Checks inequality."""
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __hash__(self) -> int:
        """Hash based on load ID."""
        return hash(self.id)

    @abstractmethod
    def __repr__(self) -> str:
        pass # pragma: no cover

    @abstractmethod
    def __str__(self) -> str:
        pass # pragma: no cover


# --- Nodal Load Class ---

class NodalLoad(Load):
    """
    Represents a concentrated load (forces and/or moment) applied directly to a node.

    Attributes:
        node_id (int): The ID of the node where the load is applied.
        fx (float): Force component in the global X direction (in Newtons, N).
        fy (float): Force component in the global Y direction (in Newtons, N).
        mz (float): Moment component about the global Z axis (in Newton-meters, N·m).
    """
    def __init__(self, id: int, node_id: int, fx: float = 0.0, fy: float = 0.0, mz: float = 0.0, label: str = ""):
        """
        Initializes a NodalLoad.

        Args:
            id: Unique load identifier.
            node_id: ID of the target node.
            fx: Global X-force component (N). Defaults to 0.0.
            fy: Global Y-force component (N). Defaults to 0.0.
            mz: Global Z-moment component (N·m). Defaults to 0.0.
            label: Optional descriptive label.

        Raises:
            TypeError: If id/node_id are not integers, or fx/fy/mz are not numeric, or label is not string.
        """
        super().__init__(id, label)
        if not isinstance(node_id, int):
            raise TypeError(f"NodalLoad node_id must be an integer (received: {node_id}).")
        if not all(isinstance(val, (int, float)) for val in [fx, fy, mz]):
            raise TypeError("NodalLoad components (fx, fy, mz) must be numeric.")

        self.node_id = node_id
        self.fx = float(fx)
        self.fy = float(fy)
        self.mz = float(mz)

    def __repr__(self) -> str:
        """Unambiguous representation."""
        label_part = f", label='{self.label}'" if self.label else ""
        return (f"NodalLoad(id={self.id}, node_id={self.node_id}, "
                f"fx={self.fx}, fy={self.fy}, mz={self.mz}{label_part})")

    def __str__(self) -> str:
        """User-friendly representation."""
        label_part = f" ({self.label})" if self.label else ""
        return (f"NodalLoad {self.id} @ Node {self.node_id}: "
                f"Fx={self.fx:.3g} N, Fy={self.fy:.3g} N, Mz={self.mz:.3g} Nm{label_part}")


# --- Member Load Base Class ---

class MemberLoad(Load, ABC):
    """
    Abstract Base Class for loads applied along the length of a member.

    Attributes:
        member_id (int): The ID of the member the load is applied to.
    """
    def __init__(self, id: int, member_id: int, label: str = ""):
        """
        Initializes the base member load.

        Args:
            id: Unique load identifier.
            member_id: ID of the target member.
            label: Optional descriptive label.

        Raises:
            TypeError: If id/member_id are not integers or label is not string.
        """
        super().__init__(id, label)
        if not isinstance(member_id, int):
             raise TypeError(f"{self.__class__.__name__} member_id must be an integer (received: {member_id}).")
        self.member_id = member_id


# --- Member Point Load ---

class MemberPointLoad(MemberLoad):
    """
    Represents a concentrated force applied at a specific point along a member's length.
    Forces are defined in the member's local coordinate system.

    Attributes:
        px (float): Force component parallel to the member's local x-axis (N).
                    Positive acts from start node towards end node.
        py (float): Force component perpendicular to the member's local x-axis (N).
                    Positive usually follows the right-hand rule (e.g., upwards for horizontal member).
        position (float): The distance from the member's start node where the load is applied (m).
                          Must be non-negative. Validation against member length happens later.
    """
    def __init__(self, id: int, member_id: int, px: float, py: float, position: float, label: str = ""):
        """
        Initializes a MemberPointLoad.

        Args:
            id: Unique load identifier.
            member_id: ID of the target member.
            px: Local x-force component (N).
            py: Local y-force component (N).
            position: Distance from start node (m). Must be >= 0.
            label: Optional descriptive label.

        Raises:
            TypeError: If id/member_id not int; px/py/position not numeric; label not string.
            ValueError: If position is negative.
        """
        super().__init__(id, member_id, label)
        if not all(isinstance(val, (int, float)) for val in [px, py, position]):
             raise TypeError("MemberPointLoad components (px, py, position) must be numeric.")
        if position < 0:
            raise ValueError(f"MemberPointLoad position ({position}) cannot be negative.")

        self.px = float(px)
        self.py = float(py)
        self.position = float(position)

    def __repr__(self) -> str:
        """Unambiguous representation."""
        label_part = f", label='{self.label}'" if self.label else ""
        return (f"MemberPointLoad(id={self.id}, member_id={self.member_id}, "
                f"px={self.px}, py={self.py}, position={self.position}{label_part})")

    def __str__(self) -> str:
        """User-friendly representation."""
        label_part = f" ({self.label})" if self.label else ""
        return (f"MemberPointLoad {self.id} on Member {self.member_id}: "
                f"Px={self.px:.3g} N, Py={self.py:.3g} N @ {self.position:.3g} m{label_part}")


# --- Member Uniformly Distributed Load (UDL) ---

class MemberUDLoad(MemberLoad):
    """
    Represents a uniformly distributed load along the entire length of a member.
    Loads are defined in the member's local coordinate system.

    Attributes:
        wx (float): Distributed load parallel to the member's local x-axis (N/m).
        wy (float): Distributed load perpendicular to the member's local x-axis (N/m).
    """
    def __init__(self, id: int, member_id: int, wx: float = 0.0, wy: float = 0.0, label: str = ""):
        """
        Initializes a MemberUDLoad.

        Args:
            id: Unique load identifier.
            member_id: ID of the target member.
            wx: Local x distributed load component (N/m). Defaults to 0.0.
            wy: Local y distributed load component (N/m). Defaults to 0.0.
            label: Optional descriptive label.

         Raises:
            TypeError: If id/member_id not int; wx/wy not numeric; label not string.
        """
        super().__init__(id, member_id, label)
        if not all(isinstance(val, (int, float)) for val in [wx, wy]):
             raise TypeError("MemberUDLoad components (wx, wy) must be numeric.")

        self.wx = float(wx)
        self.wy = float(wy)

    def __repr__(self) -> str:
        """Unambiguous representation."""
        label_part = f", label='{self.label}'" if self.label else ""
        return (f"MemberUDLoad(id={self.id}, member_id={self.member_id}, "
                f"wx={self.wx}, wy={self.wy}{label_part})")

    def __str__(self) -> str:
        """User-friendly representation."""
        label_part = f" ({self.label})" if self.label else ""
        return (f"MemberUDLoad {self.id} on Member {self.member_id}: "
                f"wx={self.wx:.3g} N/m, wy={self.wy:.3g} N/m{label_part}")

# --- Future Load Types ---
# class MemberTrapezoidalLoad(MemberLoad): ...
# class ThermalLoad(Load): ...


class StructuralModel:
    """
    Container class for the entire structural model.

    Manages nodes, materials, sections, members, supports, and loads.
    Provides methods for adding, removing, retrieving components, validation,
    and generating Degree-of-Freedom (DOF) mapping.
    """
    def __init__(self, name: str = "Untitled Model"):
        """
        Initializes an empty StructuralModel.

        Args:
            name (str): Optional name for the model.
        """
        self.name: str = name
        self.nodes: dict[int, Node] = {}
        self.materials: dict[int, Material] = {}
        self.sections: dict[int, SectionProfile] = {}
        self.members: dict[int, Member] = {}
        # Supports are keyed by the node_id they apply to
        self.supports: dict[int, Support] = {}
        # Loads are keyed by their own unique load_id
        self.loads: dict[int, Load] = {}

        # DOF mapping related attributes (generated later)
        self._dof_map: Optional[dict[tuple[int, str], int]] = None
        self._constrained_dofs: Optional[set[tuple[int, str]]] = None
        self._num_active_dofs: Optional[int] = None

    # --- Component Addition Methods ---

    def add_node(self, node: Node):
        """Adds a Node object to the model."""
        if not isinstance(node, Node):
            raise TypeError("Item to add must be a Node object.")
        if node.id in self.nodes:
            raise ValueError(f"Node with ID {node.id} already exists.")
        self.nodes[node.id] = node
        self._invalidate_dof_map() # Adding nodes changes DOF map

    def add_material(self, material: Material):
        """Adds a Material object to the model."""
        if not isinstance(material, Material):
            raise TypeError("Item to add must be a Material object.")
        if material.id in self.materials:
            raise ValueError(f"Material with ID {material.id} already exists.")
        self.materials[material.id] = material

    def add_section(self, section: SectionProfile):
        """Adds a SectionProfile object to the model."""
        if not isinstance(section, SectionProfile):
            raise TypeError("Item to add must be a SectionProfile object.")
        if section.id in self.sections:
            raise ValueError(f"SectionProfile with ID {section.id} already exists.")
        self.sections[section.id] = section

    def add_member(self, member: Member):
        """Adds a Member object to the model."""
        if not isinstance(member, Member):
            raise TypeError("Item to add must be a Member object.")
        if member.id in self.members:
            raise ValueError(f"Member with ID {member.id} already exists.")

        # Basic check if nodes exist - more thorough check in validate()
        #if member.start_node.id not in self.nodes:
        #     raise ValueError(f"Member {member.id} references non-existent start_node {member.start_node.id}")
        #if member.end_node.id not in self.nodes:
        #     raise ValueError(f"Member {member.id} references non-existent end_node {member.end_node.id}")
        
        self.members[member.id] = member

    def add_support(self, support: Support):
        """Adds a Support object to the model."""
        if not isinstance(support, Support):
            raise TypeError("Item to add must be a Support object.")
        #if support.node_id not in self.nodes:
        #     raise ValueError(f"Support references non-existent node_id {support.node_id}.")
        if support.node_id in self.supports:
            raise ValueError(f"Support already defined for node_id {support.node_id}.")
        self.supports[support.node_id] = support
        self._invalidate_dof_map() # Adding supports changes DOF map

    def add_load(self, load: Load):
        """Adds any Load object (NodalLoad, MemberLoad, etc.) to the model."""
        if not isinstance(load, Load):
            raise TypeError("Item to add must be a Load-derived object.")
        if load.id in self.loads:
            raise ValueError(f"Load with ID {load.id} already exists.")
        # Basic check if target node/member exists - more thorough check in validate()
        #if isinstance(load, NodalLoad):
        #    if load.node_id not in self.nodes:
        #        raise ValueError(f"NodalLoad {load.id} references non-existent node_id {load.node_id}.")
        #elif isinstance(load, MemberLoad):
        #    if load.member_id not in self.members:
        #         raise ValueError(f"MemberLoad {load.id} references non-existent member_id {load.member_id}.")
        self.loads[load.id] = load

    # --- Component Retrieval Methods ---

    def get_node(self, node_id: int) -> Node:
        """Retrieves a Node by its ID. Raises KeyError if not found."""
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"Node with ID {node_id} not found in the model.")

    def get_material(self, material_id: int) -> Material:
        """Retrieves a Material by its ID. Raises KeyError if not found."""
        try:
            return self.materials[material_id]
        except KeyError:
            raise KeyError(f"Material with ID {material_id} not found in the model.")

    def get_section(self, section_id: int) -> SectionProfile:
        """Retrieves a SectionProfile by its ID. Raises KeyError if not found."""
        try:
            return self.sections[section_id]
        except KeyError:
            raise KeyError(f"SectionProfile with ID {section_id} not found in the model.")

    def get_member(self, member_id: int) -> Member:
        """Retrieves a Member by its ID. Raises KeyError if not found."""
        try:
            return self.members[member_id]
        except KeyError:
            raise KeyError(f"Member with ID {member_id} not found in the model.")

    def get_support(self, node_id: int) -> Optional[Support]:
        """Retrieves the Support for a given node_id. Returns None if no support exists."""
        return self.supports.get(node_id) # Use .get() for optional retrieval

    def get_load(self, load_id: int) -> Load:
        """Retrieves a Load by its ID. Raises KeyError if not found."""
        try:
            return self.loads[load_id]
        except KeyError:
            raise KeyError(f"Load with ID {load_id} not found in the model.")

    # --- Component Removal Methods (Basic - No Cascade Checks Yet) ---

    def remove_node(self, node_id: int):
        """Removes a Node by ID. Warning: Does not check for dependent members/supports/loads."""
        if node_id not in self.nodes:
            raise KeyError(f"Node with ID {node_id} not found.")
        del self.nodes[node_id]
        # Also remove associated support if it exists
        if node_id in self.supports:
            del self.supports[node_id]
        # Invalidate DOF map
        self._invalidate_dof_map()
        # Warning: Loads targeting this node might become invalid. Validation needed.

    def remove_member(self, member_id: int):
        """Removes a Member by ID. Warning: Does not check for dependent loads."""
        if member_id not in self.members:
            raise KeyError(f"Member with ID {member_id} not found.")
        del self.members[member_id]
        # Warning: Loads targeting this member might become invalid. Validation needed.

    def remove_support(self, node_id: int):
        """Removes the Support associated with a node_id."""
        if node_id not in self.supports:
            raise KeyError(f"Support for node ID {node_id} not found.")
        del self.supports[node_id]
        self._invalidate_dof_map()

    def remove_load(self, load_id: int):
        """Removes a Load by ID."""
        if load_id not in self.loads:
            raise KeyError(f"Load with ID {load_id} not found.")
        del self.loads[load_id]

    # --- Validation ---

    def validate(self) -> list[str]:
        """
        Performs consistency checks on the model.

        Returns:
            A list of error/warning messages. An empty list indicates a valid model.
        """
        errors = []

        # 1. Member Checks
        for mem_id, member in self.members.items():
            # Check node existence (already partially done in add_member, but good to re-verify)
            if member.start_node.id not in self.nodes:
                errors.append(f"Member {mem_id}: Start node {member.start_node.id} not found in model nodes.")
            if member.end_node.id not in self.nodes:
                errors.append(f"Member {mem_id}: End node {member.end_node.id} not found in model nodes.")
            # Check material/section existence (stored as objects, so type check is implicit)
            if not isinstance(member.material, Material):
                 errors.append(f"Member {mem_id}: Invalid material object assigned ({type(member.material)}).")
            elif member.material.id not in self.materials:
                 errors.append(f"Member {mem_id}: Material {member.material.id} not found in model materials.") # Check consistency
            if not isinstance(member.section, SectionProfile):
                 errors.append(f"Member {mem_id}: Invalid section object assigned ({type(member.section)}).")
            elif member.section.id not in self.sections:
                 errors.append(f"Member {mem_id}: Section {member.section.id} not found in model sections.") # Check consistency
            # Check zero length (already done in Member init, but maybe re-check)
            if member.length == approx(0.0):
                 errors.append(f"Member {mem_id}: Member has zero length between nodes {member.start_node.id} and {member.end_node.id}.")

        # 2. Support Checks
        for node_id, support in self.supports.items():
            if node_id not in self.nodes:
                errors.append(f"Support defined for non-existent node {node_id}.")

        # 3. Load Checks
        for load_id, load in self.loads.items():
            if isinstance(load, NodalLoad):
                if load.node_id not in self.nodes:
                    errors.append(f"NodalLoad {load_id}: Target node {load.node_id} not found.")
            elif isinstance(load, MemberLoad):
                if load.member_id not in self.members:
                    errors.append(f"MemberLoad {load_id}: Target member {load.member_id} not found.")
                else:
                    # Check member-specific load validity
                    member = self.get_member(load.member_id)
                    if isinstance(load, MemberPointLoad):
                        if load.position > member.length + 1e-9: # Allow slight tolerance for end node
                             errors.append(f"MemberPointLoad {load_id} on Member {load.member_id}: Position {load.position} exceeds member length {member.length:.4g}.")
                        # Position >= 0 already checked in MemberPointLoad __init__

        # 4. Orphan Checks (Optional - could be warnings)
        # Check for unused nodes, materials, sections? Might be valid in some workflows. Skip for now.

        return errors

    # --- DOF Mapping ---

    def _invalidate_dof_map(self):
        """Resets the cached DOF map when the model structure changes."""
        self._dof_map = None
        self._constrained_dofs = None
        self._num_active_dofs = None

    def _generate_dof_map(self):
        """Internal method to generate the DOF map if it doesn't exist."""
        if self._dof_map is not None:
            return # Already generated

        dof_map: dict[tuple[int, str], int] = {}
        constrained_dofs: set[tuple[int, str]] = set()
        active_dof_index = 0
        dof_types = ['dx', 'dy', 'rz'] # Order matters!

        # Sort nodes by ID for consistent DOF numbering
        sorted_node_ids = sorted(self.nodes.keys())

        for node_id in sorted_node_ids:
            support = self.get_support(node_id)
            is_constrained = {
                'dx': support.is_dx_restrained if support else False,
                'dy': support.is_dy_restrained if support else False,
                'rz': support.is_rz_restrained if support else False,
            }

            for dof_name in dof_types:
                node_dof_tuple = (node_id, dof_name)
                if is_constrained[dof_name]:
                    constrained_dofs.add(node_dof_tuple)
                    # Optionally assign -1 or similar to map for constrained DOFs
                    dof_map[node_dof_tuple] = -1
                else:
                    dof_map[node_dof_tuple] = active_dof_index
                    active_dof_index += 1

        self._dof_map = dof_map
        self._constrained_dofs = constrained_dofs
        self._num_active_dofs = active_dof_index

    def get_dof_map(self) -> tuple[dict[tuple[int, str], int], set[tuple[int, str]], int]:
        """
        Generates and returns the global Degree-of-Freedom (DOF) mapping.

        Returns:
            A tuple containing:
            - dof_map (dict): Maps (node_id, dof_name) to a global index (or -1 if constrained).
                              dof_name is one of 'dx', 'dy', 'rz'.
            - constrained_dofs (set): A set of (node_id, dof_name) tuples for constrained DOFs.
            - num_active_dofs (int): The total number of unconstrained (active) DOFs in the model.
        """
        self._generate_dof_map()
        # Ensure the attributes are not None after generation
        if self._dof_map is None or self._constrained_dofs is None or self._num_active_dofs is None:
             # This should not happen if _generate_dof_map works correctly
             raise RuntimeError("DOF map generation failed unexpectedly.")
        return self._dof_map, self._constrained_dofs, self._num_active_dofs

    def get_active_dof_indices(self, node_id: int) -> list[int]:
        """
        Gets the global indices for the *active* DOFs associated with a specific node.

        Args:
            node_id: The ID of the node.

        Returns:
            A list of active global DOF indices for the node [dx_idx, dy_idx, rz_idx].
            If a DOF is constrained, its index might be omitted or represented differently
            depending on how the analysis engine expects it. This version returns only active indices.
        """
        self._generate_dof_map()
        if self._dof_map is None:
             raise RuntimeError("DOF map has not been generated.")

        indices = []
        for dof_name in ['dx', 'dy', 'rz']:
            idx = self._dof_map.get((node_id, dof_name), -1) # Default to -1 if somehow missing
            if idx != -1: # Only include active DOFs
                indices.append(idx)
        return indices


    # --- Other Utility Methods (Optional) ---

    def __str__(self) -> str:
        """Basic summary string representation of the model."""
        return (f"StructuralModel(name='{self.name}', "
                f"Nodes={len(self.nodes)}, Materials={len(self.materials)}, Sections={len(self.sections)}, "
                f"Members={len(self.members)}, Supports={len(self.supports)}, Loads={len(self.loads)})")


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Create dummy nodes, material, section for demonstration
    n1 = Node(1, 0, 0)
    n2 = Node(2, 4, 0) # Horizontal
    n3 = Node(3, 4, 3) # Angled
    n4 = Node(4, 0, 3) # Vertical
    mat1 = Material(1, "Steel", 210e9)
    sec1 = RectangularProfile(101, "Rect 10x20", 0.1, 0.2)

    mem1 = Member(id=10, start_node=n1, end_node=n2, material=mat1, section=sec1) # Horizontal
    mem2 = Member(id=11, start_node=n2, end_node=n3, material=mat1, section=sec1) # Angled (3-4-5 triangle)
    mem3 = Member(id=12, start_node=n1, end_node=n4, material=mat1, section=sec1) # Vertical

    print(repr(mem1))
    print(str(mem2))
    print(f"Member 1: Length={mem1.length:.4f} m, Angle={math.degrees(mem1.angle):.2f} deg")
    print(f"Member 2: Length={mem2.length:.4f} m, Angle={math.degrees(mem2.angle):.2f} deg") # ~36.87 deg
    print(f"Member 3: Length={mem3.length:.4f} m, Angle={math.degrees(mem3.angle):.2f} deg") # 90 deg

    print(f"Member 1 Properties: E={mem1.E:.2e} Pa, A={mem1.A:.4f} m^2, I={mem1.I:.4e} m^4")

    # Example of equality
    mem1_again = Member(id=10, start_node=n1, end_node=n3, material=mat1, section=sec1) # Same ID, different nodes
    print(f"mem1 == mem2: {mem1 == mem2}")
    print(f"mem1 == mem1_again: {mem1 == mem1_again}")

    member_set = {mem1, mem2, mem3, mem1_again}
    print(f"Set of members: {member_set}")

    try:
        # Zero length member
        Member(id=99, start_node=n1, end_node=n1, material=mat1, section=sec1)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        # Invalid type for node
        Member(id=98, start_node="node1", end_node=n2, material=mat1, section=sec1)
    except TypeError as e:
        print(f"Caught expected error: {e}")