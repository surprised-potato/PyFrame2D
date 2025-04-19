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