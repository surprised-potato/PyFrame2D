import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Frame, Label, Entry, Button, OptionMenu, StringVar
import math
from typing import Optional, Dict, Tuple # Ensure these are imported from typing
from tkinter import filedialog
import os
import json
import math


# --- Import Core Components ---
# Define Dummies FIRST in case import fails below
class StructuralModel: pass
class Node: pass
class Material: pass
class SectionProfile: pass
class RectangularProfile(SectionProfile): pass
class SquareProfile(SectionProfile): pass
class IBeamProfile(SectionProfile): pass
class Member: pass
class Support: pass
class Load: pass
class NodalLoad(Load): pass
class MemberLoad(Load): pass
class MemberPointLoad(MemberLoad): pass
class MemberUDLoad(MemberLoad): pass
class AnalysisResults: pass # Define dummy AnalysisResults
class LinAlgError(Exception): pass # Define dummy LinAlgError
def analyze(model, num_diagram_points=11): return AnalysisResults(status="Error", message="Core components not loaded.") # Dummy analyze

try:
    from core.model import (StructuralModel, Node, Material, SectionProfile,
                            RectangularProfile, SquareProfile, IBeamProfile,
                            Member, Support, Load, NodalLoad, MemberLoad,
                            MemberPointLoad, MemberUDLoad)
    from core.analysis import analyze, AnalysisResults # Import real AnalysisResults
    from numpy.linalg import LinAlgError # Import real LinAlgError
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"ERROR importing core components: {e}")
    # If imports fail, the dummy classes defined above will be used
    CORE_AVAILABLE = False
try:
    # Use direct imports now that project_io is renamed and path is handled
    from project_io.project_files import save_model_to_json, load_model_from_json
    IO_AVAILABLE = True
except ImportError as e:
    print(f"WARNING importing project_io module: {e}. File operations will be disabled.")
    IO_AVAILABLE = False
    # Define dummy functions
    def save_model_to_json(model, file_path): raise NotImplementedError("IO module not available.")
    def load_model_from_json(file_path): raise NotImplementedError("IO module not available.")
try:
    from units.units import parse_value_unit, format_value_unit
    UNITS_AVAILABLE = True # Defined at module level
except ImportError as e:
    print(f"WARNING importing units module: {e}. Unit parsing will be disabled.")
    UNITS_AVAILABLE = False # Also defined at module level
    def parse_value_unit(value_str, unit): # Dummy function
        print("Unit parsing disabled.")
        return float(value_str)



CANVAS_PADDING = 20 # Pixels around the drawing area
NODE_RADIUS = 3     # Pixels
SUPPORT_SIZE = 10   # Pixels (approx size of symbol)
LOAD_ARROW_LENGTH = 30 # Pixels
LOAD_ARROW_WIDTH = 2
LOAD_COLOR = "red"
SUPPORT_COLOR = "blue"
MEMBER_COLOR = "black"
NODE_COLOR = "black"
DEFLECTED_COLOR = "blue"
DEFLECTED_SCALE = 50 # Factor to scale displacements for visibility (adjust!)




class MainApplicationWindow(tk.Tk):
    """
    Main application window class using Tkinter.
    Sets up the basic menu, status bar, input forms, and main frame.
    """
class MainApplicationWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- Core check ---
        if not CORE_AVAILABLE:
            # ... (handle error) ...
            return

        self.title("PyFrame2D - Untitled")
        self.geometry("1000x750")

        # --- Define Attributes FIRST ---
        self.model = StructuralModel()
        self.analysis_results: Optional[AnalysisResults] = None
        self.current_file_path: Optional[str] = None

        # Editing State Flags
        self.editing_node_id: Optional[int] = None
        self.editing_material_id: Optional[int] = None
        self.editing_section_id: Optional[int] = None
        self.editing_member_id: Optional[int] = None
        self.editing_support_node_id: Optional[int] = None
        self.editing_load_id: Optional[int] = None

        # ID Counters
        self.next_node_id = 1
        self.next_material_id = 1
        self.next_section_id = 1
        self.next_member_id = 1
        self.next_load_id = 1

        # --- View Options (Define BEFORE _create_menu) ---
        self.show_node_ids = tk.BooleanVar(value=True)
        self.show_member_ids = tk.BooleanVar(value=True)
        self.show_supports = tk.BooleanVar(value=True)
        self.show_loads = tk.BooleanVar(value=True)
        self.show_deflected_shape = tk.BooleanVar(value=False)
        # --- End View Options ---

        # --- Call creation methods AFTER attributes are defined ---
        self._create_menu()             # Uses show_... variables
        self._create_main_layout()      # Uses show_... variables indirectly via redraw
        self._create_status_bar()

        self.protocol("WM_DELETE_WINDOW", self._on_exit) # Uses _on_exit method

        # Bind tab change event AFTER input_notebook is created in _create_main_layout
        if hasattr(self, 'input_notebook'):
            self.input_notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)
        else:
             print("Warning: input_notebook not found during __init__ binding.")




    def _on_canvas_resize(self, event):
        """Callback when the canvas size changes."""
        # Debounce or add delay if needed, but redraw directly for now
        self._redraw_canvas()

    def _redraw_canvas(self):
        """Clears and redraws the entire model on the canvas."""
        self.canvas.delete("all") # Clear previous drawing items

        if not self.model or not self.model.nodes:
            # ... (display empty model message) ...
            return

        # 1. Calculate Model Bounds and Transformation
        try:
            min_x, max_x, min_y, max_y = self._get_model_bounds()
            transform = self._get_canvas_transform(min_x, max_x, min_y, max_y)
            if not transform: return
        except ValueError:
            # ... (display add more nodes message) ...
            return

        # 2. Draw Undeformed Components
        self._draw_members(transform)
        self._draw_nodes(transform)
        self._draw_supports(transform)
        self._draw_loads(transform)

        # 3. Draw Deflected Shape (if results are available) << NEW
        if self.analysis_results and self.analysis_results.status == "Success":
            self._draw_deflected_shape(transform)

        # self.set_status("Model view updated.") # Status set elsewhere now


    def _draw_deflected_shape(self, transform):
        """Draws the deflected shape of the members."""
        if not self.show_deflected_shape.get():
            return # Skip drawing if toggled off

        if not self.analysis_results or not self.analysis_results.nodal_displacements:
            return

        displacements = self.analysis_results.nodal_displacements
        scale = DEFLECTED_SCALE # Displacement magnification factor

        for member in self.model.members.values():
            n1_id = member.start_node.id
            n2_id = member.end_node.id

            # Get original coordinates
            x1_orig, y1_orig = member.start_node.x, member.start_node.y
            x2_orig, y2_orig = member.end_node.x, member.end_node.y

            # Get displacements (handle missing nodes in results gracefully)
            disp1 = displacements.get(n1_id, (0.0, 0.0, 0.0))
            disp2 = displacements.get(n2_id, (0.0, 0.0, 0.0))
            dx1, dy1, _ = disp1 # Ignore rotation for line drawing
            dx2, dy2, _ = disp2

            # Calculate deformed coordinates (magnified)
            x1_def = x1_orig + dx1 * scale
            y1_def = y1_orig + dy1 * scale
            x2_def = x2_orig + dx2 * scale
            y2_def = y2_orig + dy2 * scale

            # Map deformed coordinates to canvas
            x1_c, y1_c = self._map_coords(x1_def, y1_def, transform)
            x2_c, y2_c = self._map_coords(x2_def, y2_def, transform)

            # Draw the deformed member
            self.canvas.create_line(x1_c, y1_c, x2_c, y2_c,
                                    fill=DEFLECTED_COLOR, width=1, dash=(4, 2), # Dashed line
                                    tags=("deflected_shape", f"deflected_{member.id}"))
    def _get_model_bounds(self) -> tuple[float, float, float, float]:
        """Calculates the min/max x and y coordinates of the model nodes."""
        if not self.model.nodes:
            raise ValueError("No nodes in model to determine bounds.")

        node_coords = [(n.x, n.y) for n in self.model.nodes.values()]
        min_x = min(c[0] for c in node_coords)
        max_x = max(c[0] for c in node_coords)
        min_y = min(c[1] for c in node_coords)
        max_y = max(c[1] for c in node_coords)

        # Handle case where all nodes are at the same point
        if max_x == min_x: max_x += 1.0 # Add arbitrary width
        if max_y == min_y: max_y += 1.0 # Add arbitrary height

        return min_x, max_x, min_y, max_y

    def _get_canvas_transform(self, min_x, max_x, min_y, max_y) -> Optional[dict]:
        """Calculates scale and offset to map model coords to canvas coords."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1: # Canvas not ready
            return None

        # Usable drawing area
        draw_width = canvas_width - 2 * CANVAS_PADDING
        draw_height = canvas_height - 2 * CANVAS_PADDING
        if draw_width <= 0 or draw_height <= 0: # Not enough space
             return None

        model_width = max_x - min_x
        model_height = max_y - min_y

        # Prevent division by zero if model has zero size (already handled in bounds)
        if model_width == 0 or model_height == 0:
             print("Warning: Model has zero width or height.")
             return None # Or set a default scale

        # Calculate scale to fit, maintaining aspect ratio
        scale_x = draw_width / model_width
        scale_y = draw_height / model_height
        scale = min(scale_x, scale_y) * 0.95 # Use 95% to add a bit more padding

        # Calculate offsets to center the drawing
        # Canvas Y is inverted (0 at top)
        scaled_model_width = model_width * scale
        scaled_model_height = model_height * scale
        offset_x = CANVAS_PADDING + (draw_width - scaled_model_width) / 2.0 - (min_x * scale)
        offset_y = CANVAS_PADDING + draw_height - (draw_height - scaled_model_height) / 2.0 + (min_y * scale)

        return {"scale": scale, "offset_x": offset_x, "offset_y": offset_y, "canvas_height": canvas_height}


    def _map_coords(self, x_model, y_model, transform) -> tuple[float, float]:
        """Maps model coordinates to canvas pixel coordinates."""
        if not transform: return 0, 0
        # canvas_x = transform["offset_x"] + (x_model * transform["scale"])
        # canvas_y = transform["offset_y"] - (y_model * transform["scale"]) # Invert Y
        canvas_x = transform["offset_x"] + x_model * transform["scale"]
        canvas_y = transform["offset_y"] - y_model * transform["scale"]
        return canvas_x, canvas_y

    def _draw_nodes(self, transform):
        """Draws nodes and optionally their IDs on the canvas."""
        for node in self.model.nodes.values():
            cx, cy = self._map_coords(node.x, node.y, transform)
            r = NODE_RADIUS
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=NODE_COLOR, outline=NODE_COLOR, tags=("node", f"node_{node.id}"))
            # --- Add check for showing Node ID ---
            if self.show_node_ids.get():
                 self.canvas.create_text(cx + r + 3, cy, # Adjust positioning slightly
                                         text=f"{node.id}", anchor=tk.W, # Anchor West (left)
                                         fill="purple", font=("Segoe UI", 8),
                                         tags=("node_label", f"node_label_{node.id}"))
            # --- End check ---

    def _draw_members(self, transform):
        """Draws members (lines) on the canvas."""
        for member in self.model.members.values():
            n1 = member.start_node
            n2 = member.end_node
            x1, y1 = self._map_coords(n1.x, n1.y, transform)
            x2, y2 = self._map_coords(n2.x, n2.y, transform)
            self.canvas.create_line(x1, y1, x2, y2, fill=MEMBER_COLOR, width=2, tags=("member", f"member_{member.id}"))

    def _draw_supports(self, transform):
        """Draws support symbols on the canvas."""
        if not self.show_supports.get():
            return # Skip drawing if toggled off

        s = SUPPORT_SIZE / 2.0
        for node_id, support in self.model.supports.items():
             # ... (existing code to get node, cx, cy) ...
             if node_id not in self.model.nodes: continue
             node = self.model.nodes[node_id]
             cx, cy = self._map_coords(node.x, node.y, transform)
             tag = ("support", f"support_{node_id}")


             # Draw based on type (simplified symbols)
             if support.dx and support.dy and support.rz: # Fixed
                 self.canvas.create_rectangle(cx - s, cy - s, cx + s, cy + s, outline=SUPPORT_COLOR, width=2, tags=tag)
                 self.canvas.create_line(cx - s, cy, cx + s, cy, fill=SUPPORT_COLOR, width=1, tags=tag) # Cross
                 self.canvas.create_line(cx, cy - s, cx, cy + s, fill=SUPPORT_COLOR, width=1, tags=tag)
             elif support.dx and support.dy: # Pinned
                 self.canvas.create_polygon(cx, cy - s, cx - s, cy + s, cx + s, cy + s, outline=SUPPORT_COLOR, fill='', width=2, tags=tag)
             elif support.dy: # Roller X (restrains Y)
                 self.canvas.create_polygon(cx, cy - s, cx - s, cy + s, cx + s, cy + s, outline=SUPPORT_COLOR, fill='', width=2, tags=tag)
                 self.canvas.create_line(cx - s, cy + s + 2, cx + s, cy + s + 2, fill=SUPPORT_COLOR, width=1, tags=tag) # Line underneath
             elif support.dx: # Roller Y (restrains X)
                 # Rotated triangle symbol (simplified)
                 self.canvas.create_polygon(cx - s, cy, cx + s, cy - s, cx + s, cy + s, outline=SUPPORT_COLOR, fill='', width=2, tags=tag)
                 self.canvas.create_line(cx + s + 2, cy - s, cx + s + 2, cy + s, fill=SUPPORT_COLOR, width=1, tags=tag) # Line beside
             else: # Other combinations (e.g., only Rz - draw a small circle?)
                 self.canvas.create_oval(cx-s/2, cy-s/2, cx+s/2, cy+s/2, outline=SUPPORT_COLOR, width=1, tags=tag)

    def _draw_loads(self, transform):
        """Draws load symbols on the canvas."""
        if not self.show_loads.get():
            return # Skip drawing if toggled off

        # --- Draw Nodal Loads ---
        for load in self.model.loads.values():
            if isinstance(load, NodalLoad):
                # ... (nodal load drawing code - remains the same) ...
                if load.node_id not in self.model.nodes: continue
                node = self.model.nodes[load.node_id]
                cx, cy = self._map_coords(node.x, node.y, transform)
                self._draw_load_arrow(cx, cy, load.fx, load.fy, tags=("load", f"load_{load.id}"))
                self._draw_load_moment(cx, cy, load.mz, tags=("load", f"load_{load.id}"))


            elif isinstance(load, MemberLoad): # Includes Point and UDL
                if load.member_id not in self.model.members: continue
                member = self.model.members[load.member_id]
                n1 = member.start_node
                n2 = member.end_node
                x1_c, y1_c = self._map_coords(n1.x, n1.y, transform)
                x2_c, y2_c = self._map_coords(n2.x, n2.y, transform)
                angle_rad = member.angle # Global angle

                if isinstance(load, MemberPointLoad):
                    # Calculate position on canvas
                    pos_ratio = load.position / member.length if member.length > 0 else 0
                    pos_x_c = x1_c + (x2_c - x1_c) * pos_ratio
                    pos_y_c = y1_c + (y2_c - y1_c) * pos_ratio
                    # Rotate local loads px, py into global components fx, fy
                    c, s = math.cos(angle_rad), math.sin(angle_rad) # Defined here
                    fx_global = load.px * c - load.py * s
                    fy_global = load.px * s + load.py * c
                    self._draw_load_arrow(pos_x_c, pos_y_c, fx_global, fy_global, tags=("load", f"load_{load.id}"))

                elif isinstance(load, MemberUDLoad):
                    # Simple representation: draw perpendicular arrows along member
                    num_arrows = 5 # Draw a few arrows to represent UDL
                    # --- Calculate c and s here ---
                    c, s = math.cos(angle_rad), math.sin(angle_rad)
                    # --- End Calculation ---
                    for i in range(1, num_arrows + 1):
                         ratio = i / (num_arrows + 1)
                         pos_x_c = x1_c + (x2_c - x1_c) * ratio
                         pos_y_c = y1_c + (y2_c - y1_c) * ratio
                         # Rotate local wy into global components (ignore wx for drawing simplicity)
                         # Force vector for +wy is perpendicular to member in local 'y' direction
                         # Unit vector along local y is (-sin(angle), cos(angle)) in global coords
                         fx_global = -load.wy * s # Global X component
                         fy_global = load.wy * c  # Global Y component
                         # Draw smaller arrows for UDL
                         self._draw_load_arrow(pos_x_c, pos_y_c, fx_global, fy_global, length_scale=0.6, tags=("load", f"load_{load.id}"))


    def _draw_load_arrow(self, cx, cy, fx, fy, length_scale=1.0, tags=()):
        """Helper to draw force arrows."""
        if abs(fx) < 1e-9 and abs(fy) < 1e-9: return # Don't draw zero forces

        length = LOAD_ARROW_LENGTH * length_scale
        # Scale fx, fy to get direction vector, normalize, then scale by length
        norm = math.sqrt(fx**2 + fy**2)
        dir_x = (fx / norm) * length
        dir_y = (-fy / norm) * length # Negative Fy points UP in canvas coords

        arrow_end_x = cx + dir_x
        arrow_end_y = cy + dir_y

        self.canvas.create_line(cx, cy, arrow_end_x, arrow_end_y,
                                fill=LOAD_COLOR, width=LOAD_ARROW_WIDTH, arrow=tk.LAST, tags=tags)

    def _draw_load_moment(self, cx, cy, mz, tags=()):
        """Helper to draw moment symbols."""
        if abs(mz) < 1e-9: return # Don't draw zero moment

        radius = SUPPORT_SIZE # Reuse size constant
        # Draw a curved arrow symbol (simplified)
        if mz > 0: # Positive moment (CCW)
            start_angle, extent = 270, -180 # Arc from bottom, counter-clockwise
        else: # Negative moment (CW)
            start_angle, extent = 90, -180 # Arc from top, counter-clockwise (looks CW visually)

        self.canvas.create_arc(cx - radius, cy - radius, cx + radius, cy + radius,
                               start=start_angle, extent=extent, style=tk.ARC,
                               outline=LOAD_COLOR, width=LOAD_ARROW_WIDTH, tags=tags)
        # Add arrowhead (approximate)
        # This requires calculating end point of arc - skip for simplicity now


    # --- Action Method Modifications ---
    # Need to call _redraw_canvas() after modifying the model

    def _add_node(self):
        """Reads node data, validates, creates Node object, adds to model."""
        node_id = -1 # Initialize to default
        try:
            node_id = self.next_node_id # Get next ID

            x_str = self.node_x_entry.get()
            y_str = self.node_y_entry.get()
            if not x_str: raise ValueError("X coordinate cannot be empty.") # Explicit check
            if not y_str: raise ValueError("Y coordinate cannot be empty.") # Explicit check

            x = float(x_str) # Convert after checking empty
            y = float(y_str) # Convert after checking empty

            node = Node(id=node_id, x=x, y=y)
            self.model.add_node(node)
            self.next_node_id += 1

            self.set_status(f"Node {node_id} added successfully.")
            # Clear X and Y fields only
            self.node_x_entry.delete(0, tk.END)
            self.node_y_entry.delete(0, tk.END)
            self.node_x_entry.focus() # Focus on first editable field

            self._update_next_id_display() # Update the disabled ID field
            self._populate_nodes_listbox()
            self._redraw_canvas()
            self.update_idletasks()

        except ValueError as e:
            # More specific error message
            error_msg = f"Invalid numeric input for Node {node_id if node_id != -1 else '(?)'}: {e}"
            messagebox.showerror("Input Error", error_msg)
            self.set_status(f"Error adding node: Invalid number.")
        except TypeError as e:
             error_msg = f"Invalid input type for Node {node_id if node_id != -1 else '(?)'}: {e}"
             messagebox.showerror("Input Error", error_msg)
             self.set_status(f"Error adding node: Invalid type.")
        except Exception as e: # Catch other errors
             error_msg = f"Could not add Node {node_id if node_id != -1 else '(?)'}: {e}"
             messagebox.showerror("Error", error_msg)
             self.set_status(f"Error adding node: {e}")


    def _add_material(self):
        """Reads material data, validates, creates Material object, adds to model."""
        mat_id = -1 # Default for error message
        try:
            mat_id = self.next_material_id # <<< Use counter
            name = self.mat_name_entry.get()
            e_str = self.mat_e_entry.get()
            if not name: raise ValueError("Material name cannot be empty.")
            if not e_str: raise ValueError("Young's Modulus cannot be empty.")

            youngs_modulus_value = e_str # Pass string directly

            mat = Material(id=mat_id, name=name, youngs_modulus=youngs_modulus_value)
            self.model.add_material(mat)
            self.next_material_id += 1 # <<< Increment counter

            self.set_status(f"Material {mat_id} ({name}) added successfully.")
            # Clear entries (except ID)
            self.mat_name_entry.delete(0, tk.END)
            self.mat_e_entry.delete(0, tk.END)
            self.mat_name_entry.focus() # Focus on first editable field

            self._update_next_id_display() # Update the disabled ID field
            print(f"Model Materials: {self.model.materials}")
            self._populate_materials_listbox()
            # No redraw needed

        except ValueError as e:
            error_msg = f"Invalid input for Material {mat_id if mat_id != -1 else '(?)'}: {e}"
            messagebox.showerror("Input Error", error_msg)
            self.set_status(f"Error adding material: {e}")
        except TypeError as e:
             error_msg = f"Invalid input type for Material {mat_id if mat_id != -1 else '(?)'}: {e}"
             messagebox.showerror("Input Error", error_msg)
             self.set_status(f"Error adding material: {e}")
        except Exception as e:
             error_msg = f"Could not add Material {mat_id if mat_id != -1 else '(?)'}: {e}"
             messagebox.showerror("Error", error_msg)
             self.set_status(f"Error adding material: {e}")
    def _add_section(self):
        """Reads section data, validates, creates Section object, adds to model."""
        sec_id = -1 # Default for error
        try:
            sec_id = self.next_section_id # <<< Use counter
            name = self.sec_name_entry.get()
            sec_type = self.section_type_var.get()

            section = None
            if sec_type == "Rectangular":
                width = float(self.sec_rect_width_entry.get())
                height = float(self.sec_rect_height_entry.get())
                section = RectangularProfile(sec_id, name, width, height)
            elif sec_type == "Square":
                side = float(self.sec_sq_side_entry.get())
                section = SquareProfile(sec_id, name, side)
            elif sec_type == "I-Beam":
                h = float(self.sec_ibeam_h_entry.get())
                bf = float(self.sec_ibeam_bf_entry.get())
                tf = float(self.sec_ibeam_tf_entry.get())
                tw = float(self.sec_ibeam_tw_entry.get())
                section = IBeamProfile(sec_id, name, h, bf, tf, tw)
            else:
                raise ValueError(f"Selected section type '{sec_type}' not handled.")

            if section:
                self.model.add_section(section)
                self.next_section_id += 1 # <<< Increment counter

                self.set_status(f"Section {sec_id} ({name}) added successfully.")
                self._populate_sections_listbox()
                # Clear common entries
                self.sec_id_entry.delete(0, tk.END)
                self.sec_name_entry.delete(0, tk.END)
                # Clear specific entries by updating fields (easier than tracking)
                self._update_section_fields() # This clears specific fields
                self.sec_id_entry.focus()
                self._update_next_id_display() 
                print(f"Model Sections: {self.model.sections}") # Debug print
                # No redraw needed for sections

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for Section: {e}")
            self.set_status(f"Error adding section: {e}")
        except TypeError as e:
             messagebox.showerror("Input Error", f"Invalid input type for Section: {e}")
             self.set_status(f"Error adding section: {e}")
        except AttributeError as e:
             messagebox.showerror("Input Error", f"Missing input field for selected section type: {e}")
             self.set_status(f"Error adding section: Incomplete fields.")
        except Exception as e:
             # --- INDENT THESE LINES ---
             messagebox.showerror("Error", f"Could not add Section: {e}")
             self.set_status(f"Error adding section: {e}")
             # --- END INDENTATION ---

    def _update_section_fields(self, *args):
        """Clears and updates the section input fields based on selected type."""
        # Destroy existing widgets in the specific frame
        # Ensure self.section_fields_frame is correctly defined and accessible
        if hasattr(self, 'section_fields_frame'):
            for widget in self.section_fields_frame.winfo_children():
                widget.destroy()
        else:
            print("Error: section_fields_frame not found in _update_section_fields")
            return # Cannot proceed

        sec_type = self.section_type_var.get()

        # Create widgets based on type
        if sec_type == "Rectangular":
            ttk.Label(self.section_fields_frame, text="Width (m):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
            self.sec_rect_width_entry = ttk.Entry(self.section_fields_frame, width=10)
            self.sec_rect_width_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.section_fields_frame, text="Height (m):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
            self.sec_rect_height_entry = ttk.Entry(self.section_fields_frame, width=10)
            self.sec_rect_height_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        elif sec_type == "Square":
            ttk.Label(self.section_fields_frame, text="Side (m):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
            self.sec_sq_side_entry = ttk.Entry(self.section_fields_frame, width=10)
            self.sec_sq_side_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        elif sec_type == "I-Beam":
            ttk.Label(self.section_fields_frame, text="Height (h, m):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
            self.sec_ibeam_h_entry = ttk.Entry(self.section_fields_frame, width=10)
            self.sec_ibeam_h_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.section_fields_frame, text="Flange W (bf, m):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
            self.sec_ibeam_bf_entry = ttk.Entry(self.section_fields_frame, width=10)
            self.sec_ibeam_bf_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.section_fields_frame, text="Flange Thk (tf, m):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
            self.sec_ibeam_tf_entry = ttk.Entry(self.section_fields_frame, width=10)
            self.sec_ibeam_tf_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.section_fields_frame, text="Web Thk (tw, m):").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
            self.sec_ibeam_tw_entry = ttk.Entry(self.section_fields_frame, width=10)
            self.sec_ibeam_tw_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.EW)
        # Add elif for other types here
    def _update_member(self):
        """Handles updating an existing member after editing."""
        if self.editing_member_id is None:
            self._reset_member_button()
            return

        mem_id = self.editing_member_id # ID doesn't change
        try:
            # Get new references based on entered IDs
            start_node_id = int(self.mem_start_node_entry.get())
            end_node_id = int(self.mem_end_node_entry.get())
            material_id = int(self.mem_material_entry.get())
            section_id = int(self.mem_section_entry.get())

            # Get referenced objects from model (validate they exist)
            new_start_node = self.model.get_node(start_node_id)
            new_end_node = self.model.get_node(end_node_id)
            new_material = self.model.get_material(material_id)
            new_section = self.model.get_section(section_id)

            # Get the existing member object
            member = self.model.get_member(mem_id)

            # Check for zero-length member with new nodes BEFORE updating
            if new_start_node.get_coords() == new_end_node.get_coords():
                 raise ValueError(f"Updated nodes {start_node_id} and {end_node_id} have identical coordinates.")

            # Update attributes
            member.start_node = new_start_node
            member.end_node = new_end_node
            member.material = new_material
            member.section = new_section

            self.set_status(f"Member {mem_id} updated successfully.")
            print(f"Member {mem_id} updated. New state: {self.model.members}")

            # Reset editing state and button
            self._reset_member_button()
            # Clear fields
            self.mem_start_node_entry.delete(0, tk.END)
            self.mem_end_node_entry.delete(0, tk.END)
            self.mem_material_entry.delete(0, tk.END)
            self.mem_section_entry.delete(0, tk.END)
            self._update_next_id_display() # Update ID field too
            self.mem_start_node_entry.focus()

            # Update listbox and canvas
            self._populate_members_listbox()
            self._redraw_canvas()
            self.update_idletasks()

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid input for Member {mem_id} update: {e}"
            messagebox.showerror("Input Error", error_msg)
            self.set_status(f"Error updating member: {e}")
        except KeyError as e: # Catch errors from get_node/mat/sec
             error_msg = f"Invalid reference ID for Member {mem_id} update: {e}"
             messagebox.showerror("Input Error", error_msg)
             self.set_status(f"Error updating member: {e}")
             self._reset_member_button() # Still reset button
        except Exception as e:
             error_msg = f"Could not update Member {mem_id}: {e}"
             messagebox.showerror("Error", error_msg)
             self.set_status(f"Error updating member: {e}")
             self._reset_member_button()

    def _add_member(self):
        """Reads member data, validates, creates Member object, adds to model."""
        mem_id = -1 # Default for error message
        try:
            mem_id = self.next_member_id # <<< Use counter
            start_node_id = int(self.mem_start_node_entry.get())
            end_node_id = int(self.mem_end_node_entry.get())
            material_id = int(self.mem_material_entry.get())
            section_id = int(self.mem_section_entry.get())

            # Get referenced objects from model
            start_node = self.model.get_node(start_node_id)
            end_node = self.model.get_node(end_node_id)
            material = self.model.get_material(material_id)
            section = self.model.get_section(section_id)

            member = Member(mem_id, start_node, end_node, material, section)
            self.model.add_member(member)
            self.next_member_id += 1 # <<< Increment counter

            self.set_status(f"Member {mem_id} added successfully.")
            # Clear entries (except ID)
            self.mem_start_node_entry.delete(0, tk.END)
            self.mem_end_node_entry.delete(0, tk.END)
            self.mem_material_entry.delete(0, tk.END)
            self.mem_section_entry.delete(0, tk.END)
            self.mem_start_node_entry.focus() # Focus on first editable field

            self._update_next_id_display() # <<< Update displayed ID
            print(f"Model Members: {self.model.members}")
            self._populate_members_listbox()
            self._redraw_canvas()
            self.update_idletasks()

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid input for Member {mem_id if mem_id != -1 else '(?)'}: {e}"
            messagebox.showerror("Input Error", error_msg)
            self.set_status(f"Error adding member: {e}")
        except KeyError as e:
             error_msg = f"Invalid reference ID for Member {mem_id if mem_id != -1 else '(?)'}: {e}"
             messagebox.showerror("Input Error", error_msg)
             self.set_status(f"Error adding member: {e}")
        except Exception as e:
             error_msg = f"Could not add Member {mem_id if mem_id != -1 else '(?)'}: {e}"
             messagebox.showerror("Error", error_msg)
             self.set_status(f"Error adding member: {e}")

    def _add_support(self):
        """Reads support data, validates, creates NEW Support object, adds to model."""
        # Ensure node ID entry is enabled when adding
        if hasattr(self, 'sup_node_id_entry'):
            self.sup_node_id_entry.config(state=tk.NORMAL)

        node_id = -1 # Default for error message
        try:
            node_id = int(self.sup_node_id_entry.get())
            # --- Check if support already exists ---
            if node_id in self.model.supports:
                raise ValueError(f"Support already exists for node {node_id}. Use 'Edit Selected' to modify.")
            # --- End check ---
            # --- Check if node itself exists in the model ---
            if node_id not in self.model.nodes:
                raise ValueError(f"Node {node_id} does not exist in the model.")
            # --- End check ---

            dx = self.sup_dx_var.get()
            dy = self.sup_dy_var.get()
            rz = self.sup_rz_var.get()

            support = Support(node_id, dx, dy, rz)
            self.model.add_support(support) # Should succeed now

            self.set_status(f"Support added for Node {node_id}.")
            # Clear entry and reset dropdown/checkboxes via type var maybe?
            self.sup_node_id_entry.delete(0, tk.END)
            self.sup_type_var.set("Pinned") # Reset dropdown to default
            self._update_support_dofs() # Update checkboxes based on dropdown reset
            self.sup_node_id_entry.focus()

            print(f"Model Supports: {self.model.supports}")
            self._populate_supports_listbox()
            self._redraw_canvas()
            self.update_idletasks()

        except (ValueError, TypeError) as e:
            messagebox.showerror("Input Error", f"Invalid input for Support: {e}")
            self.set_status(f"Error adding support: {e}")
        except Exception as e:
             messagebox.showerror("Error", f"Could not add Support: {e}")
             self.set_status(f"Error adding support: {e}")
        finally:
             # Ensure node ID entry is re-enabled even if error occurred
             if hasattr(self, 'sup_node_id_entry'):
                 self.sup_node_id_entry.config(state=tk.NORMAL)

    def _add_load(self):
        """Reads load data, validates, creates Load object, adds to model."""
        load_id = -1 # Default for error message
        load_type = self.load_type_var.get() # Get type early for focus logic

        try:
            load_id = self.next_load_id # Use counter
            label = self.load_label_entry.get()
            # load_type already fetched above

            load = None
            # --- Validate target existence BEFORE creating load ---
            if load_type == "Nodal":
                node_id_str = self.load_nodal_node_entry.get()
                if not node_id_str: raise ValueError("Node ID cannot be empty.")
                node_id = int(node_id_str)
                if node_id not in self.model.nodes:
                     raise ValueError(f"Node {node_id} does not exist in the model.")
                # --- End Node check ---
                fx = float(self.load_nodal_fx_entry.get())
                fy = float(self.load_nodal_fy_entry.get())
                mz = float(self.load_nodal_mz_entry.get())
                load = NodalLoad(load_id, node_id, fx, fy, mz, label)
            elif load_type == "Member Point":
                mem_id_str = self.load_pt_mem_entry.get()
                if not mem_id_str: raise ValueError("Member ID cannot be empty.")
                mem_id = int(mem_id_str)
                # --- Check if member exists ---
                if mem_id not in self.model.members:
                     raise ValueError(f"Member {mem_id} does not exist in the model.")
                # --- End Member check ---
                px = float(self.load_pt_px_entry.get())
                py = float(self.load_pt_py_entry.get())
                pos = float(self.load_pt_pos_entry.get())
                load = MemberPointLoad(load_id, mem_id, px, py, pos, label)
            elif load_type == "Member UDL":
                mem_id_str = self.load_udl_mem_entry.get()
                if not mem_id_str: raise ValueError("Member ID cannot be empty.")
                mem_id = int(mem_id_str)
                 # --- Check if member exists ---
                if mem_id not in self.model.members:
                     raise ValueError(f"Member {mem_id} does not exist in the model.")
                # --- End Member check ---
                wx = float(self.load_udl_wx_entry.get())
                wy = float(self.load_udl_wy_entry.get())
                load = MemberUDLoad(load_id, mem_id, wx, wy, label)
            else:
                 # This case should ideally not be reachable if dropdown is correct
                 raise ValueError(f"Selected load type '{load_type}' not handled.")

            # If load object creation succeeded:
            if load:
                 self.model.add_load(load)
                 self.next_load_id += 1 # Increment counter

                 self.set_status(f"Load {load_id} added successfully.")
                 # Clear label field
                 self.load_label_entry.delete(0, tk.END)
                 # Clear specific fields by calling update
                 self._update_load_fields()
                 # Update the displayed next ID
                 self._update_next_id_display()
                 print(f"Model Loads: {self.model.loads}") # Debug print

                 # Repopulate listbox
                 self._populate_loads_listbox()
                 # Redraw canvas
                 self._redraw_canvas()
                 # Force GUI update before setting focus
                 self.update_idletasks()

                 # --- Set Focus Correctly ---
                 print(f"DEBUG [AddLoad]: Setting focus. load_type = '{load_type}'") # Debug
                 if load_type == "Nodal":
                     if hasattr(self, 'load_nodal_node_entry'): self.load_nodal_node_entry.focus()
                 elif load_type == "Member Point":
                     if hasattr(self, 'load_pt_mem_entry'): self.load_pt_mem_entry.focus()
                 elif load_type == "Member UDL":
                     if hasattr(self, 'load_udl_mem_entry'): self.load_udl_mem_entry.focus() # Focus on UDL member entry
                 else: # Fallback
                     self.load_label_entry.focus()
                 print(f"DEBUG [AddLoad]: Focus set attempt finished.") # Debug
                 # --- End Focus ---

        # --- Exception Handling ---
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid input for Load {load_id if load_id != -1 else '(?)'}: {e}"
            messagebox.showerror("Input Error", error_msg)
            self.set_status(f"Error adding load: {e}")
        except AttributeError as e:
            # This catches if specific entry fields don't exist when .get() is called
            error_msg = f"Missing input field for Load {load_id if load_id != -1 else '(?)'}: {e}"
            messagebox.showerror("Input Error", error_msg)
            self.set_status(f"Error adding load: Incomplete fields.")
        except Exception as e:
             error_msg = f"Could not add Load {load_id if load_id != -1 else '(?)'}: {e}"
             messagebox.showerror("Error", error_msg)
             self.set_status(f"Error adding load: {e}")


    def _update_load(self):
        """Handles updating an existing load after editing."""
        if self.editing_load_id is None:
            self._reset_load_button()
            return

        load_id = self.editing_load_id # ID doesn't change
        try:
            # Get potentially edited common fields
            label = self.load_label_entry.get()
            # Type doesn't change during edit
            load_type = self.load_type_var.get()

            # Get the existing load object
            load = self.model.get_load(load_id)

            # Update attributes based on type
            if isinstance(load, NodalLoad) and load_type == "Nodal":
                # Node ID typically shouldn't change during edit, but check if needed
                # node_id = int(self.load_nodal_node_entry.get())
                # if node_id != load.node_id: # If changing target node
                #    if node_id not in self.model.nodes: raise ValueError(...)
                #    load.node_id = node_id
                load.fx = float(self.load_nodal_fx_entry.get())
                load.fy = float(self.load_nodal_fy_entry.get())
                load.mz = float(self.load_nodal_mz_entry.get())
            elif isinstance(load, MemberPointLoad) and load_type == "Member Point":
                # mem_id = int(self.load_pt_mem_entry.get()) # Allow changing target member?
                # if mem_id != load.member_id:
                #    if mem_id not in self.model.members: raise ValueError(...)
                #    load.member_id = mem_id
                load.px = float(self.load_pt_px_entry.get())
                load.py = float(self.load_pt_py_entry.get())
                load.position = float(self.load_pt_pos_entry.get())
                # Re-validate position against member length?
                if load.member_id in self.model.members:
                    member = self.model.members[load.member_id]
                    if load.position < 0 or load.position > member.length + 1e-9:
                         raise ValueError(f"Position {load.position} is outside member {load.member_id} length {member.length:.4g}.")
                else: # Should not happen if model is consistent
                     print(f"Warning: Cannot validate position for load {load_id}, member {load.member_id} not found.")

            elif isinstance(load, MemberUDLoad) and load_type == "Member UDL":
                # mem_id = int(self.load_udl_mem_entry.get()) # Allow changing target member?
                # if mem_id != load.member_id:
                #    if mem_id not in self.model.members: raise ValueError(...)
                #    load.member_id = mem_id
                load.wx = float(self.load_udl_wx_entry.get())
                load.wy = float(self.load_udl_wy_entry.get())
            else:
                 raise TypeError(f"Cannot update load {load_id}: Type mismatch or unsupported type '{load_type}'.")

            # Update common attribute
            load.label = label

            self.set_status(f"Load {load_id} updated successfully.")
            print(f"Load {load_id} updated. New state: {self.model.loads}")

            # Reset editing state and button
            self._reset_load_button()
            # Clear fields
            self.load_label_entry.delete(0, tk.END)
            self._update_load_fields() # Clears specific fields
            self._update_next_id_display() # Update ID field too
            # Could set focus back to label or first specific field

            # Update listbox and canvas
            self._populate_loads_listbox()
            self._redraw_canvas()
            self.update_idletasks()

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid input for Load {load_id} update: {e}"
            messagebox.showerror("Input Error", error_msg)
            self.set_status(f"Error updating load: {e}")
        except KeyError:
             messagebox.showerror("Error", f"Load {load_id} not found for update (unexpected).")
             self.set_status(f"Error updating load: Not found.")
             self._reset_load_button()
        except AttributeError as e:
            messagebox.showerror("Input Error", f"Missing input field for load type update: {e}")
            self.set_status(f"Error updating load: Field error.")
            self._reset_load_button()
        except Exception as e:
             error_msg = f"Could not update Load {load_id}: {e}"
             messagebox.showerror("Error", error_msg)
             self.set_status(f"Error updating load: {e}")
             self._reset_load_button()

    # --- Utility Methods ---


    def _new_project(self):
        """Clears the current model and results."""
        try:
            if messagebox.askokcancel("New Project", "Clear current model and start new project?"):
                 self.model = StructuralModel()
                 self.analysis_results = None # <<< CLEAR RESULTS
                 self._clear_results_display() # <<< CLEAR TEXT WIDGETS
                 self.set_status("New project started.")
                 print("Model Cleared.")
                 self._redraw_canvas() # Redraw to clear canvas
        except Exception as e:
            messagebox.showerror("Error", f"Could not create new project: {e}")
            self.set_status(f"Error creating new project: {e}")

    def _create_menu(self):
        """Creates the main menu bar and its submenus."""
        # ... (Menu creation code remains the same as before) ...
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        # --- File Menu ---
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        # --- Link to actual functions ---
        file_menu.add_command(label="New Project", command=self._file_new)
        file_menu.add_command(label="Open Project...", command=self._file_open)
        file_menu.add_command(label="Save Project", command=self._file_save)
        file_menu.add_command(label="Save Project As...", command=self._file_save_as)
        # --- End Link ---
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)

        # --- Edit Menu (Now mainly for context, adding done via tabs) ---
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo (Placeholder)", command=self._placeholder_command)
        edit_menu.add_command(label="Redo (Placeholder)", command=self._placeholder_command)

        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        # Add checkbuttons linked to BooleanVars
        view_menu.add_checkbutton(label="Show Node IDs", variable=self.show_node_ids, command=self._redraw_canvas)
        view_menu.add_checkbutton(label="Show Member IDs", variable=self.show_member_ids, command=self._redraw_canvas)
        view_menu.add_checkbutton(label="Show Supports", variable=self.show_supports, command=self._redraw_canvas)
        view_menu.add_checkbutton(label="Show Loads", variable=self.show_loads, command=self._redraw_canvas)
        view_menu.add_checkbutton(label="Show Deflected Shape", variable=self.show_deflected_shape, command=self._redraw_canvas)
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In (Placeholder)", command=self._placeholder_command)
        view_menu.add_command(label="Zoom Out (Placeholder)", command=self._placeholder_command)
        view_menu.add_command(label="Pan (Placeholder)", command=self._placeholder_command)
        view_menu.add_command(label="Fit to View (Placeholder)", command=self._placeholder_command)
        # --- End View Menu ---


        # --- Analyze Menu ---
        analyze_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Analyze", menu=analyze_menu)
        analyze_menu.add_command(label="Run Analysis", command=self._run_analysis) # <<< LINKED
        analyze_menu.add_command(label="Show Results Report", command=self._placeholder_command)

        # --- Help Menu ---
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._placeholder_command)
        help_menu.add_command(label="About PyFrame2D", command=self._placeholder_command)


    def _create_main_layout(self):
        """Creates the main layout with input panel, canvas, and results panel."""
        # Main frame to hold input (left) and display/results (right)
        main_paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Input Panel Frame (Left Side) ---
        self.input_panel = ttk.Frame(main_paned_window, width=350, relief=tk.RIDGE) # Slightly wider
        main_paned_window.add(self.input_panel, weight=1) # Add to paned window
        # self.input_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0), pady=5) # Removed pack
        # self.input_panel.pack_propagate(False) # Not needed with paned window?

        # Notebook (Tabs) for different inputs
        self.input_notebook = ttk.Notebook(self.input_panel)
        self.input_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5,0))

        # Create input tabs (use self.input_notebook)
        self._create_nodes_tab()
        self._create_materials_tab()
        self._create_sections_tab()
        self._create_members_tab()
        self._create_supports_tab()
        self._create_loads_tab()

        # Add Run Analysis Button below input notebook
        run_button = ttk.Button(self.input_panel, text="Run Analysis", command=self._run_analysis)
        run_button.pack(pady=10)

        # --- Right Side Frame (Canvas + Results) ---
        right_frame = ttk.Frame(main_paned_window)
        main_paned_window.add(right_frame, weight=3) # Give more weight to display area

        # Use another PanedWindow for Canvas and Results (Vertical)
        display_results_pane = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        display_results_pane.pack(fill=tk.BOTH, expand=True)

        # --- Canvas Frame (Top Right) ---
        self.canvas_frame = ttk.Frame(display_results_pane, relief=tk.SUNKEN)
        display_results_pane.add(self.canvas_frame, weight=4) # Give more weight to canvas
        # self.canvas_frame.pack(...) # Removed pack

        self.canvas = tk.Canvas(self.canvas_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # --- Results Panel (Bottom Right) ---
        self.results_panel = ttk.Frame(display_results_pane, height=200, relief=tk.RIDGE)
        display_results_pane.add(self.results_panel, weight=1) # Less weight for results initially
        # self.results_panel.pack(...) # Removed pack
        # self.results_panel.pack_propagate(False) # Prevent resizing based on content

        # Notebook for Results Tabs
        self.results_notebook = ttk.Notebook(self.results_panel)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create results tabs with Text widgets and Scrollbars
        self.disp_text = self._create_results_tab("Displacements")
        self.react_text = self._create_results_tab("Reactions")
        self.forces_text = self._create_results_tab("Member Forces")

        # Initial drawing
        self._redraw_canvas() # Call after layout is created

    def _create_results_tab(self, tab_name: str) -> tk.Text:
        """Creates a tab with a scrollable text widget for displaying results."""
        frame = ttk.Frame(self.results_notebook, padding="5")
        self.results_notebook.add(frame, text=tab_name)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        text_widget = tk.Text(frame, wrap=tk.NONE, yscrollcommand=scrollbar.set,
                              state=tk.DISABLED, # Start read-only
                              font=("Consolas", 9)) # Use monospace font

        scrollbar.config(command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        return text_widget

    def _update_window_title(self):
        """Updates the window title based on the current file path."""
        base_title = "PyFrame2D"
        if self.current_file_path:
            # Get just the filename from the path
            filename = os.path.basename(self.current_file_path)
            self.title(f"{base_title} - {filename}")
        else:
            self.title(f"{base_title} - Untitled")
        # TODO: Add '*' indicator for unsaved changes later

    def _file_new(self):
        """Clears the current model and results, resets GUI state."""
        try:
            # TODO: Add check for unsaved changes
            if messagebox.askokcancel("New Project", "Clear current model and start new project?\nAny unsaved changes will be lost."):
                 self.model = StructuralModel() # Create new empty model
                 self.analysis_results = None

                 # --- Reset ID Counters ---
                 self.next_node_id = 1
                 self.next_material_id = 1
                 self.next_section_id = 101
                 self.next_member_id = 201
                 self.next_load_id = 301
                 # --- End Reset ---

                 self._clear_results_display() # Clear results tabs
                 self.current_file_path = None
                 self._update_window_title()

                 # --- ADD GUI UPDATES ---
                 self._update_all_listboxes() # Clears listboxes based on empty model
                 self._reset_all_buttons_and_ids() # Resets buttons and updates ID displays
                 # --- END ADD ---

                 self.set_status("New project started.")
                 print("Model Cleared.")
                 self._redraw_canvas() # Redraw to clear canvas
                 self.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Could not create new project: {e}")
            self.set_status(f"Error creating new project: {e}")

    def _file_open(self):
        """Handles the File -> Open Project action."""
        if not IO_AVAILABLE:
             messagebox.showerror("Error", "File IO module not available.")
             return
        # TODO: Add check for unsaved changes before opening

        # Ask user for file path
        filepath = filedialog.askopenfilename(
            title="Open PyFrame2D Project",
            filetypes=[("PyFrame2D JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if not filepath: # User cancelled
            self.set_status("Open cancelled.")
            return

        self.set_status(f"Opening file: {filepath}...")
        self.update_idletasks()

        try:
            loaded_model = load_model_from_json(filepath)
            self.model = loaded_model # Replace current model
            self.analysis_results = None # Clear previous results
            self._clear_results_display()
            self.current_file_path = filepath # Store path
            self._update_window_title()
            max_node_id = max(list(self.model.nodes.keys()) + [0])
            max_mat_id = max(list(self.model.materials.keys()) + [0])
            max_sec_id = max(list(self.model.sections.keys()) + [0])
            max_mem_id = max(list(self.model.members.keys()) + [0])
            max_load_id = max(list(self.model.loads.keys()) + [0])

            # Set next ID to be one greater than the highest found ID
            self.next_node_id = max_node_id + 1
            self.next_material_id = max_mat_id + 1
            # Ensure minimum starting points if needed (optional)
            self.next_section_id = max(max_sec_id + 1, 101)
            self.next_member_id = max(max_mem_id + 1, 201)
            self.next_load_id = max(max_load_id + 1, 301)
            # --- End Update Counters ---

            self._update_all_listboxes()
            self._reset_all_buttons_and_ids() # Reset buttons and update ID display fields
            self._redraw_canvas()
            self.update_idletasks()

            self._redraw_canvas() # Update visualization
            self.update_idletasks() # <<< ADD THIS to show loaded model
            self.set_status(f"Model loaded successfully from {os.path.basename(filepath)}")
            print(f"Model loaded from {filepath}")
            # Optionally update input forms here if needed

        except FileNotFoundError:
             messagebox.showerror("Error Loading File", f"File not found:\n{filepath}")
             self.set_status("Error: File not found.")
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
             messagebox.showerror("Error Loading File", f"Failed to load model data from file:\n{filepath}\n\nError: {e}")
             self.set_status("Error loading file.")
             print(f"Error details: {e}")
        except Exception as e:
             messagebox.showerror("Error Loading File", f"An unexpected error occurred while loading:\n{filepath}\n\nError: {e}")
             self.set_status("Error loading file.")
             print(f"Unexpected loading error: {e}")
             import traceback
             traceback.print_exc()
    def _reset_all_buttons_and_ids(self):
        """Helper to reset all edit buttons and update next ID displays."""
        self._reset_node_button()
        self._reset_material_button()
        self._reset_section_button()
        self._reset_member_button()
        self._reset_support_button() # Support doesn't use auto ID, but reset button
        self._reset_load_button()
        self._update_next_id_display() # Ensure all ID fields show correct next value
    def _file_save(self) -> bool:
        """Handles the File -> Save Project action. Returns True if successful."""
        if not IO_AVAILABLE:
             messagebox.showerror("Error", "File IO module not available.")
             return False

        if self.current_file_path:
            self.set_status(f"Saving model to {os.path.basename(self.current_file_path)}...")
            self.update_idletasks()
            try:
                save_model_to_json(self.model, self.current_file_path)
                self.set_status("Model saved successfully.")
                print(f"Model saved to {self.current_file_path}")
                # TODO: Clear 'unsaved changes' indicator later
                return True
            except Exception as e:
                 messagebox.showerror("Save Error", f"Could not save file:\n{self.current_file_path}\n\nError: {e}")
                 self.set_status("Error saving file.")
                 print(f"Error saving file: {e}")
                 return False
        else:
            # If no current path, behave like Save As
            return self._file_save_as()

    def _file_save_as(self) -> bool:
        """Handles the File -> Save Project As action. Returns True if successful."""
        if not IO_AVAILABLE:
             messagebox.showerror("Error", "File IO module not available.")
             return False

        # Ask user for save path
        filepath = filedialog.asksaveasfilename(
            title="Save PyFrame2D Project As",
            defaultextension=".json",
            initialfile="Untitled.json",
            filetypes=[("PyFrame2D JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if not filepath: # User cancelled
            self.set_status("Save As cancelled.")
            return False

        # Ensure extension is .json (optional, filedialog often handles this)
        # if not filepath.lower().endswith(".json"):
        #     filepath += ".json"

        self.set_status(f"Saving model to {os.path.basename(filepath)}...")
        self.update_idletasks()
        try:
            save_model_to_json(self.model, filepath)
            self.current_file_path = filepath # Update current path
            self._update_window_title()
            self.set_status("Model saved successfully.")
            print(f"Model saved to {filepath}")
            # TODO: Clear 'unsaved changes' indicator later
            return True
        except Exception as e:
            messagebox.showerror("Save As Error", f"Could not save file:\n{filepath}\n\nError: {e}")
            self.set_status("Error saving file.")
            print(f"Error saving file As: {e}")
            return False

    def _run_analysis(self):
        """Callback function to run the structural analysis."""
        if not CORE_AVAILABLE:
            messagebox.showerror("Error", "Core analysis components are not available.")
            return

        # Clear previous results
        self._clear_results_display()
        self.analysis_results = None
        self._redraw_canvas() # Redraw without deflected shape first

        self.set_status("Running analysis...")
        self.update_idletasks() # Force GUI update

        try:
            # Perform validation before analysis (optional but recommended)
            validation_errors = self.model.validate()
            if validation_errors:
                error_msg = "Model validation failed:\n- " + "\n- ".join(validation_errors)
                messagebox.showwarning("Validation Warning", error_msg)
                self.set_status("Analysis aborted due to validation errors.")
                return # Stop if validation fails

            # Call the analysis function from core.analysis
            self.analysis_results = analyze(self.model, num_diagram_points=11) # Request diagrams

            # Display results
            if self.analysis_results.status == "Success":
                self.set_status("Analysis complete. Displaying results...")
                self._display_results(self.analysis_results)
                self._redraw_canvas() # Redraw includes deflected shape now
                print("Analysis successful.")
            else:
                 self.set_status(f"Analysis failed: {self.analysis_results.status}. {self.analysis_results.message or ''}")
                 messagebox.showerror("Analysis Error", f"Analysis Failed: {self.analysis_results.status}\n{self.analysis_results.message or ''}")
                 print(f"Analysis failed: {self.analysis_results.status}")

        except LinAlgError as e:
             # Specific error for unstable structures
             err_msg = f"Analysis Error: Stiffness matrix is singular.\nCheck supports and model stability.\n({e})"
             messagebox.showerror("Analysis Error", err_msg)
             self.set_status("Analysis failed: Singular Matrix.")
             print(err_msg)
        except Exception as e:
            # Catch any other unexpected errors during analysis
            err_msg = f"An unexpected error occurred during analysis:\n{type(e).__name__}: {e}"
            messagebox.showerror("Analysis Error", err_msg)
            self.set_status("Analysis failed: Unexpected error.")
            print(err_msg)
            import traceback
            traceback.print_exc() # Print full traceback to console for debugging


    def _clear_results_display(self):
        """Clears the content of the results text widgets."""
        for text_widget in [self.disp_text, self.react_text, self.forces_text]:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete('1.0', tk.END)
            text_widget.config(state=tk.DISABLED)

    def _display_results(self, results: AnalysisResults):
        """Populates the results tabs with formatted data."""
        self._clear_results_display() # Ensure clean slate

        # --- Displacements ---
        self.disp_text.config(state=tk.NORMAL)
        self.disp_text.insert(tk.END, f"{'Node ID':<10} {'DX (m)':<15} {'DY (m)':<15} {'RZ (rad)':<15}\n")
        self.disp_text.insert(tk.END, "-" * 55 + "\n")
        if results.nodal_displacements:
            for node_id in sorted(results.nodal_displacements.keys()):
                dx, dy, rz = results.nodal_displacements[node_id]
                self.disp_text.insert(tk.END, f"{node_id:<10} {dx:<15.6e} {dy:<15.6e} {rz:<15.6e}\n")
        else:
             self.disp_text.insert(tk.END, "(Not Available)\n")
        self.disp_text.config(state=tk.DISABLED)

        # --- Reactions ---
        self.react_text.config(state=tk.NORMAL)
        self.react_text.insert(tk.END, f"{'Node ID':<10} {'RX (N)':<15} {'RY (N)':<15} {'MZ (Nm)':<15}\n")
        self.react_text.insert(tk.END, "-" * 55 + "\n")
        if results.support_reactions:
            for node_id in sorted(results.support_reactions.keys()):
                 rx, ry, mz = results.support_reactions[node_id]
                 # Optionally add back the check for non-zero if desired
                 self.react_text.insert(tk.END, f"{node_id:<10} {rx:<15.6e} {ry:<15.6e} {mz:<15.6e}\n")
        else:
            self.react_text.insert(tk.END, "(Not Available)\n")
        self.react_text.config(state=tk.DISABLED)

        # --- Member Forces ---
        self.forces_text.config(state=tk.NORMAL)
        self.forces_text.insert(tk.END, f"{'Member ID':<12} {'Pxi (N)':<12} {'Pyi (N)':<12} {'Mzi (Nm)':<12} {'Pxj (N)':<12} {'Pyj (N)':<12} {'Mzj (Nm)':<12}\n")
        self.forces_text.insert(tk.END, "-" * 84 + "\n")
        if results.member_end_forces:
             for mem_id in sorted(results.member_end_forces.keys()):
                forces = results.member_end_forces[mem_id].flatten()
                if len(forces) == 6:
                    pxi, pyi, mzi, pxj, pyj, mzj = forces
                    self.forces_text.insert(tk.END, f"{mem_id:<12} {pxi:<12.4e} {pyi:<12.4e} {mzi:<12.4e} {pxj:<12.4e} {pyj:<12.4e} {mzj:<12.4e}\n")
                else:
                    self.forces_text.insert(tk.END, f"{mem_id:<12} -- Invalid Data Shape --\n")
        else:
             self.forces_text.insert(tk.END, "(Not Available)\n")
        self.forces_text.config(state=tk.DISABLED)


    def _create_status_bar(self):
        # ... (Status bar creation remains the same) ...
        """Creates the status bar at the bottom."""
        self.status_bar = ttk.Label(
            self,
            text="Ready",
            relief=tk.SUNKEN, # Give it a sunken appearance
            anchor=tk.W     # Anchor text to the West (left)
        )
        # Pack it at the bottom, making it fill the horizontal space
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)


    # --- Input Tab Creation Methods ---

    def _create_nodes_tab(self):
        """Creates the GUI elements for the Nodes input tab."""
        # Use a main frame for the entire tab content
        tab_frame = ttk.Frame(self.input_notebook, padding="10")
        self.input_notebook.add(tab_frame, text="Nodes")

        # --- Frame specifically for input fields and Add button ---
        input_frame = ttk.Frame(tab_frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10)) # Pack at top
        input_frame.columnconfigure(1, weight=1) # Allow entry column to expand

        # Input Widgets (using grid within input_frame)
        ttk.Label(input_frame, text="Node ID:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.node_id_entry = ttk.Entry(input_frame, width=15)
        self.node_id_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(input_frame, text="X Coord (m):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.node_x_entry = ttk.Entry(input_frame, width=15)
        self.node_x_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(input_frame, text="Y Coord (m):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.node_y_entry = ttk.Entry(input_frame, width=15)
        self.node_y_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)

        add_button = ttk.Button(input_frame, text="Add Node", command=self._add_node)
        add_button.grid(row=3, column=0, columnspan=2, pady=5)
         # --- Store button reference ---
        self.node_add_update_button = ttk.Button(input_frame, text="Add Node", command=self._add_node)
        self.node_add_update_button.grid(row=3, column=0, columnspan=2, pady=5)
        # --- End Store ---
        # --- End Input Frame ---

        # --- Separator between input and list ---
        ttk.Separator(tab_frame, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, pady=5)

        # --- Frame specifically for the Listbox and Scrollbar ---
        list_frame = ttk.Frame(tab_frame)
        # Pack below separator, allow to fill remaining space
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Listbox and Scrollbar inside list_frame
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.node_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, exportselection=False)
        list_scrollbar.config(command=self.node_listbox.yview)

        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y) # Scrollbar on right
        self.node_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) # Listbox takes rest of space
        # --- End Listbox Frame ---

        # --- Frame specifically for Action Buttons (Edit/Delete) ---
        action_frame = ttk.Frame(tab_frame)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0)) # Pack below listbox

        # Buttons inside action_frame
        edit_button = ttk.Button(action_frame, text="Edit Selected", command=self._edit_selected_node)
        delete_button = ttk.Button(action_frame, text="Delete Selected", command=self._delete_selected_node)
        # Pack buttons side-by-side, aligned right
        delete_button.pack(side=tk.RIGHT, padx=5)
        edit_button.pack(side=tk.RIGHT)
        # --- End Action Buttons Frame ---

        # Initial population of the listbox
        self._populate_nodes_listbox()

    def _create_materials_tab(self):
        """Creates the GUI elements for the Materials input tab."""
        tab_frame = ttk.Frame(self.input_notebook, padding="10")
        self.input_notebook.add(tab_frame, text="Materials")

        # --- Input Frame ---
        input_frame = ttk.Frame(tab_frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Material ID:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.mat_id_entry = ttk.Entry(input_frame, width=20)
        self.mat_id_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        # --- Disable ID entry and show next ID ---
        self.mat_id_entry.config(state=tk.DISABLED)
        self.mat_id_entry.insert(0, str(self.next_material_id))
        # --- End Disable ---

        ttk.Label(input_frame, text="Name:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.mat_name_entry = ttk.Entry(input_frame, width=20)
        self.mat_name_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(input_frame, text="Young's Mod (E):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.mat_e_entry = ttk.Entry(input_frame, width=20)
        self.mat_e_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
        # Keep placeholder or clear it
        # self.mat_e_entry.insert(0, "e.g., 210 GPa or 2.1e11")

        # --- Store button reference ---
        self.mat_add_update_button = ttk.Button(input_frame, text="Add Material", command=self._add_material)
        self.mat_add_update_button.grid(row=3, column=0, columnspan=2, pady=5)
        # --- End Store ---

        # ... (Separator, Listbox Frame, Listbox, Scrollbar) ...
        list_frame = ttk.Frame(tab_frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.material_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, exportselection=False)
        list_scrollbar.config(command=self.material_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.material_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


        # --- Action Buttons Frame ---
        action_frame = ttk.Frame(tab_frame)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        edit_button = ttk.Button(action_frame, text="Edit Selected", command=self._edit_selected_material)
        delete_button = ttk.Button(action_frame, text="Delete Selected", command=self._delete_selected_material)
        delete_button.pack(side=tk.RIGHT, padx=5)
        edit_button.pack(side=tk.RIGHT)

        # Initial population
        self._populate_materials_listbox()
    def _update_material(self):
        """Handles updating an existing material after editing."""
        if self.editing_material_id is None:
            self._reset_material_button()
            return

        try:
            # ID is not changed
            mat_id = self.editing_material_id
            name = self.mat_name_entry.get()
            e_str = self.mat_e_entry.get() # Get potentially edited string
            if not name: raise ValueError("Material name cannot be empty.")
            if not e_str: raise ValueError("Young's Modulus cannot be empty.")

            # Validate/convert E string again using Material class logic
            # We need to create a temporary Material instance just for validation/conversion
            # or replicate the logic here. Let's try replicating.
            try:
                youngs_modulus_value = float(e_str) # Try direct float first
            except ValueError:
                if UNITS_AVAILABLE:
                    try:
                         # Pass string to Material constructor for parsing (more robust)
                         temp_mat_for_validation = Material(id=-1, name="validation", youngs_modulus=e_str)
                         youngs_modulus_value = temp_mat_for_validation.E
                    except (ValueError, TypeError) as parse_err:
                         raise ValueError(f"Invalid Young's Modulus format '{e_str}': {parse_err}") from parse_err
                else:
                    raise ValueError(f"Invalid numeric value for Young's Modulus: '{e_str}' (unit parsing disabled)")

            if youngs_modulus_value <= 0:
                raise ValueError("Young's Modulus must be positive.")

            # Get the existing material object
            mat = self.model.get_material(mat_id)

            # Update attributes
            mat.name = name
            mat.E = youngs_modulus_value # Update with validated/converted value

            self.set_status(f"Material {mat_id} updated successfully.")
            print(f"Material {mat_id} updated. New state: {self.model.materials}")

            # Reset editing state and button
            self._reset_material_button()
            # Clear fields
            self.mat_name_entry.delete(0, tk.END)
            self.mat_e_entry.delete(0, tk.END)
            self._update_next_id_display() # Update ID field too
            self.mat_name_entry.focus()

            # Update listbox
            self._populate_materials_listbox()
            # No canvas redraw needed

        except (ValueError, TypeError) as e:
            messagebox.showerror("Input Error", f"Invalid input for Material update: {e}")
            self.set_status(f"Error updating material: {e}")
        except KeyError:
             messagebox.showerror("Error", f"Material {mat_id} not found for update (unexpected).")
             self.set_status(f"Error updating material: Not found.")
             self._reset_material_button()
        except Exception as e:
             messagebox.showerror("Error", f"Could not update Material {mat_id}: {e}")
             self.set_status(f"Error updating material: {e}")
             self._reset_material_button()
    def _edit_selected_material(self):
        """Populates material input fields with data from the selected material."""
        mat_id = self._get_selected_id_from_listbox(self.material_listbox)
        if mat_id is None:
            self._reset_material_button() # Reset if selection lost
            return
        try:
            mat = self.model.get_material(mat_id)
            # Clear fields first
            self.mat_id_entry.config(state=tk.NORMAL) # Enable temporarily
            self.mat_id_entry.delete(0, tk.END); self.mat_name_entry.delete(0, tk.END); self.mat_e_entry.delete(0, tk.END)
            # Populate fields
            self.mat_id_entry.insert(0, str(mat.id))
            self.mat_id_entry.config(state=tk.DISABLED) # Disable again
            self.mat_name_entry.insert(0, str(mat.name))
            self.mat_e_entry.insert(0, f"{mat.E:.6g}") # Show base value

            # --- Set editing state and change button ---
            self.editing_material_id = mat_id
            self.mat_add_update_button.config(text="Update Material", command=self._update_material)
            # --- End Change ---

            self.set_status(f"Editing Material {mat_id}. Modify fields and click 'Update Material'.")

        except KeyError:
             messagebox.showerror("Error", f"Material {mat_id} not found for editing.")
             self._reset_material_button()
        except Exception as e:
             messagebox.showerror("Error", f"Could not prepare material for editing: {e}")
             self._reset_material_button()   
   
   
    def _create_sections_tab(self):
        """Creates the GUI elements for the Sections input tab."""
        tab_frame = ttk.Frame(self.input_notebook, padding="10")
        self.input_notebook.add(tab_frame, text="Sections")

        # --- Input Frame ---
        input_frame = ttk.Frame(tab_frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        # Section Type Selection
        ttk.Label(input_frame, text="Section Type:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.section_type_var = tk.StringVar(value="Rectangular")
        section_types = ["Rectangular", "Square", "I-Beam"]
        type_menu = ttk.OptionMenu(input_frame, self.section_type_var, section_types[0], *section_types,
                                   command=self._update_section_fields)
        type_menu.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)

        # Common Fields
        ttk.Label(input_frame, text="Section ID:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        # --- Create the Entry widget FIRST ---
        self.sec_id_entry = ttk.Entry(input_frame, width=15)
        self.sec_id_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        # --- NOW Configure it ---
        self.sec_id_entry.config(state=tk.DISABLED)
        self.sec_id_entry.insert(0, str(self.next_section_id)) # Use the counter
        # --- End Correction ---

        ttk.Label(input_frame, text="Name:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.sec_name_entry = ttk.Entry(input_frame, width=15)
        self.sec_name_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)

        # Frame for Type-Specific Fields
        self.section_fields_frame = ttk.Frame(input_frame) # Put inside input frame
        self.section_fields_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)
        self.section_fields_frame.columnconfigure(1, weight=1)
        self.sec_add_update_button = ttk.Button(input_frame, text="Add Section", command=self._add_section)
        self.sec_add_update_button.grid(row=4, column=0, columnspan=2, pady=5)


        # --- Separator ---
        ttk.Separator(tab_frame, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, pady=5)

        # --- Listbox Frame ---
        list_frame = ttk.Frame(tab_frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.section_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, exportselection=False)
        list_scrollbar.config(command=self.section_listbox.yview)

        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.section_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Action Buttons Frame ---
        action_frame = ttk.Frame(tab_frame)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        edit_button = ttk.Button(action_frame, text="Edit Selected", command=self._edit_selected_section) # Link edit
        delete_button = ttk.Button(action_frame, text="Delete Selected", command=self._delete_selected_section) # Link delete
        delete_button.pack(side=tk.RIGHT, padx=5)
        edit_button.pack(side=tk.RIGHT)

        # Initialize fields and listbox
        self._update_section_fields()
        self._populate_sections_listbox() # Call populate

# gui/main_window.py

    def _edit_selected_section(self):
        """Populates section input fields with data from the selected section."""
        sec_id = self._get_selected_id_from_listbox(self.section_listbox)
        if sec_id is None:
            self._reset_section_button()
            return
        try:
            sec = self.model.get_section(sec_id)
            sec_type_name = sec.__class__.__name__.replace("Profile","")

            # Set common fields FIRST
            self.sec_id_entry.config(state=tk.NORMAL)
            self.sec_id_entry.delete(0, tk.END); self.sec_name_entry.delete(0, tk.END)
            self.sec_id_entry.insert(0, str(sec.id))
            self.sec_id_entry.config(state=tk.DISABLED)
            self.sec_name_entry.insert(0, str(sec.name))

            # --- Update dropdown FIRST to recreate fields ---
            self.section_type_var.set(sec_type_name)
            # --- Force Tkinter to process the update NOW ---
            self.update_idletasks()
            # --- END UPDATE ---

            # --- Now populate the NEWLY CREATED specific fields ---
            # We assume _update_section_fields has correctly assigned the
            # new entry widgets to the corresponding self attributes.
            if isinstance(sec, RectangularProfile):
                 # Check if attribute exists before inserting (robustness)
                 if hasattr(self, 'sec_rect_width_entry'): self.sec_rect_width_entry.insert(0, str(sec.width))
                 if hasattr(self, 'sec_rect_height_entry'): self.sec_rect_height_entry.insert(0, str(sec.height))
            elif isinstance(sec, SquareProfile):
                 if hasattr(self, 'sec_sq_side_entry'): self.sec_sq_side_entry.insert(0, str(sec.side_length))
            elif isinstance(sec, IBeamProfile):
                 if hasattr(self, 'sec_ibeam_h_entry'): self.sec_ibeam_h_entry.insert(0, str(sec.height))
                 if hasattr(self, 'sec_ibeam_bf_entry'): self.sec_ibeam_bf_entry.insert(0, str(sec.flange_width))
                 if hasattr(self, 'sec_ibeam_tf_entry'): self.sec_ibeam_tf_entry.insert(0, str(sec.flange_thickness))
                 if hasattr(self, 'sec_ibeam_tw_entry'): self.sec_ibeam_tw_entry.insert(0, str(sec.web_thickness))
            # --- End Populate ---

            # Set editing state and change button
            self.editing_section_id = sec_id
            self.sec_add_update_button.config(text="Update Section", command=self._update_section)

            self.set_status(f"Editing Section {sec_id}. Modify fields and click 'Update Section'.")

        except KeyError:
             messagebox.showerror("Error", f"Section {sec_id} not found for editing.")
             self._reset_section_button()
        except AttributeError as e:
             # This might catch if an expected entry attribute wasn't created
             messagebox.showerror("Error", f"GUI Error preparing section fields for edit: {e}")
             self._reset_section_button()
        except Exception as e:
             messagebox.showerror("Error", f"Could not prepare section for editing: {e}")
             self._reset_section_button()

    def _update_section(self):
        """Handles updating an existing section after editing."""
        if self.editing_section_id is None:
            self._reset_section_button()
            return

        try:
            sec_id = self.editing_section_id # ID doesn't change
            name = self.sec_name_entry.get()
            sec_type = self.section_type_var.get() # Type doesn't change during edit
            if not name: raise ValueError("Section name cannot be empty.")

            # Get the existing section object
            section = self.model.get_section(sec_id)

            # Update attributes based on type
            # Important: We assume the type wasn't changed during edit.
            # If type *could* change, logic would be more complex (delete old, add new).
            if isinstance(section, RectangularProfile) and sec_type == "Rectangular":
                section.width = float(self.sec_rect_width_entry.get())
                section.height = float(self.sec_rect_height_entry.get())
                if section.width <= 0 or section.height <= 0: raise ValueError("Dimensions must be positive.")
            elif isinstance(section, SquareProfile) and sec_type == "Square":
                section.side_length = float(self.sec_sq_side_entry.get())
                if section.side_length <= 0: raise ValueError("Side length must be positive.")
            elif isinstance(section, IBeamProfile) and sec_type == "I-Beam":
                section.height = float(self.sec_ibeam_h_entry.get())
                section.flange_width = float(self.sec_ibeam_bf_entry.get())
                section.flange_thickness = float(self.sec_ibeam_tf_entry.get())
                section.web_thickness = float(self.sec_ibeam_tw_entry.get())
                # Add validation checks from IBeamProfile.__init__ again
                if not (section.height > 0 and section.flange_width > 0 and section.flange_thickness > 0 and section.web_thickness > 0):
                     raise ValueError("IBeamProfile dimensions must be positive.")
                if section.height <= 2 * section.flange_thickness:
                     raise ValueError("Invalid I-beam geometry: height <= 2 * flange_thickness.")
                if section.flange_width < section.web_thickness:
                     raise ValueError("Invalid I-beam geometry: flange_width < web_thickness.")
            else:
                # This case handles if the type somehow changed or doesn't match
                 raise TypeError(f"Cannot update section {sec_id}: Type mismatch or unsupported type '{sec_type}'.")

            # Update common attribute
            section.name = name

            self.set_status(f"Section {sec_id} updated successfully.")
            print(f"Section {sec_id} updated. New state: {self.model.sections}")

            # Reset editing state and button
            self._reset_section_button()
            # Clear fields
            self.sec_name_entry.delete(0, tk.END)
            self._update_section_fields() # Clears specific fields
            self._update_next_id_display() # Update ID field too
            self.sec_name_entry.focus()

            # Update listbox
            self._populate_sections_listbox()
            # No canvas redraw needed for section property changes

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid input for Section {sec_id} update: {e}"
            messagebox.showerror("Input Error", error_msg)
            self.set_status(f"Error updating section: {e}")
        except KeyError:
             messagebox.showerror("Error", f"Section {sec_id} not found for update (unexpected).")
             self.set_status(f"Error updating section: Not found.")
             self._reset_section_button()
        except AttributeError as e: # Handle case where specific entry field doesn't exist for current type
            messagebox.showerror("Input Error", f"Missing input field for section type update: {e}")
            self.set_status(f"Error updating section: Field error.")
            self._reset_section_button()
        except Exception as e:
             error_msg = f"Could not update Section {sec_id}: {e}"
             messagebox.showerror("Error", error_msg)
             self.set_status(f"Error updating section: {e}")
             self._reset_section_button()


# gui/main_window.py

    def _create_members_tab(self):
        """Creates the GUI elements for the Members input tab."""
        tab_frame = ttk.Frame(self.input_notebook, padding="10")
        self.input_notebook.add(tab_frame, text="Members")

        # --- Input Frame ---
        input_frame = ttk.Frame(tab_frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Member ID:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.mem_id_entry = ttk.Entry(input_frame, width=15)
        self.mem_id_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        # --- Disable ID entry and show next ID ---
        self.mem_id_entry.config(state=tk.DISABLED)
        self.mem_id_entry.insert(0, str(self.next_member_id))
        # --- End Disable ---

        ttk.Label(input_frame, text="Start Node ID:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.mem_start_node_entry = ttk.Entry(input_frame, width=15)
        self.mem_start_node_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(input_frame, text="End Node ID:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.mem_end_node_entry = ttk.Entry(input_frame, width=15)
        self.mem_end_node_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(input_frame, text="Material ID:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.mem_material_entry = ttk.Entry(input_frame, width=15)
        self.mem_material_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(input_frame, text="Section ID:").grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        self.mem_section_entry = ttk.Entry(input_frame, width=15)
        self.mem_section_entry.grid(row=4, column=1, padx=5, pady=2, sticky=tk.EW)

        # --- Store button reference ---
        self.mem_add_update_button = ttk.Button(input_frame, text="Add Member", command=self._add_member)
        self.mem_add_update_button.grid(row=5, column=0, columnspan=2, pady=5)
        # --- End Store ---

        # --- Separator ---
        ttk.Separator(tab_frame, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, pady=5)

        # --- Listbox Frame ---
        list_frame = ttk.Frame(tab_frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.member_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, exportselection=False)
        list_scrollbar.config(command=self.member_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.member_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Action Buttons Frame ---
        action_frame = ttk.Frame(tab_frame)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        edit_button = ttk.Button(action_frame, text="Edit Selected", command=self._edit_selected_member)
        delete_button = ttk.Button(action_frame, text="Delete Selected", command=self._delete_selected_member)
        delete_button.pack(side=tk.RIGHT, padx=5)
        edit_button.pack(side=tk.RIGHT)

        # Initial population
        self._populate_members_listbox()    
    
    def _edit_selected_member(self):
        """Populates member input fields with data from the selected member."""
        mem_id = self._get_selected_id_from_listbox(self.member_listbox)
        if mem_id is None:
            self._reset_member_button() # Reset on selection loss
            return
        try:
            mem = self.model.get_member(mem_id)
            # Clear fields
            self.mem_id_entry.config(state=tk.NORMAL) # Enable temporarily
            self.mem_id_entry.delete(0, tk.END); self.mem_start_node_entry.delete(0, tk.END)
            self.mem_end_node_entry.delete(0, tk.END); self.mem_material_entry.delete(0, tk.END)
            self.mem_section_entry.delete(0, tk.END)
            # Populate fields
            self.mem_id_entry.insert(0, str(mem.id))
            self.mem_id_entry.config(state=tk.DISABLED) # Disable again
            self.mem_start_node_entry.insert(0, str(mem.start_node.id))
            self.mem_end_node_entry.insert(0, str(mem.end_node.id))
            self.mem_material_entry.insert(0, str(mem.material.id))
            self.mem_section_entry.insert(0, str(mem.section.id))

            # --- Set editing state and change button ---
            self.editing_member_id = mem_id
            self.mem_add_update_button.config(text="Update Member", command=self._update_member)
            # --- End Change ---

            self.set_status(f"Editing Member {mem_id}. Modify fields and click 'Update Member'.")

        except KeyError:
             messagebox.showerror("Error", f"Member {mem_id} not found for editing.")
             self._reset_member_button()
        except Exception as e:
             messagebox.showerror("Error", f"Could not prepare member for editing: {e}")
             self._reset_member_button()

    def _create_supports_tab(self):
        """Creates the GUI elements for the Supports input tab."""
        tab_frame = ttk.Frame(self.input_notebook, padding="10")
        self.input_notebook.add(tab_frame, text="Supports")

        # --- Input Frame ---
        input_frame = ttk.Frame(tab_frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Node ID:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.sup_node_id_entry = ttk.Entry(input_frame, width=15) # ID is entered manually
        self.sup_node_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # OptionMenu for predefined types
        ttk.Label(input_frame, text="Type:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.sup_type_var = tk.StringVar(value="Pinned")
        sup_types = ["Pinned", "Fixed", "Roller X (Free X)", "Roller Y (Free Y)", "Custom"]
        type_menu = ttk.OptionMenu(input_frame, self.sup_type_var, sup_types[0], *sup_types,
                                  command=self._update_support_dofs)
        type_menu.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        # Checkbuttons for custom definition
        self.sup_dx_var = tk.BooleanVar(value=True)
        self.sup_dy_var = tk.BooleanVar(value=True)
        self.sup_rz_var = tk.BooleanVar(value=False)
        dx_check = ttk.Checkbutton(input_frame, text="Restrain DX", variable=self.sup_dx_var, state=tk.DISABLED)
        dx_check.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        dy_check = ttk.Checkbutton(input_frame, text="Restrain DY", variable=self.sup_dy_var, state=tk.DISABLED)
        dy_check.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        rz_check = ttk.Checkbutton(input_frame, text="Restrain RZ", variable=self.sup_rz_var, state=tk.DISABLED)
        rz_check.grid(row=4, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        self.sup_dof_checks = [dx_check, dy_check, rz_check]

        # --- Store button reference ---
        self.sup_add_update_button = ttk.Button(input_frame, text="Add Support", command=self._add_support)
        self.sup_add_update_button.grid(row=5, column=0, columnspan=2, pady=5)
        # --- End Store ---

        # --- Separator ---
        ttk.Separator(tab_frame, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, pady=5)

        # --- Listbox Frame ---
        list_frame = ttk.Frame(tab_frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.support_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, exportselection=False)
        list_scrollbar.config(command=self.support_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.support_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Action Buttons Frame ---
        action_frame = ttk.Frame(tab_frame)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        edit_button = ttk.Button(action_frame, text="Edit Selected", command=self._edit_selected_support)
        delete_button = ttk.Button(action_frame, text="Delete Selected", command=self._delete_selected_support)
        delete_button.pack(side=tk.RIGHT, padx=5)
        edit_button.pack(side=tk.RIGHT)

        # Initial population
        self._update_support_dofs()
        self._populate_supports_listbox()


    def _edit_selected_support(self):
        """Populates support input fields with data from the selected support."""
        # Parse Node ID from listbox string "Node ID: {id} ..."
        selection = self.support_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a support from the list first.")
            return # Return if nothing selected
        node_id = None # Initialize
        try:
            selected_index = selection[0]
            selected_string = self.support_listbox.get(selected_index)
            node_id = int(selected_string.split()[2]) # Get the ID after "Node ID:"

            sup = self.model.get_support(node_id)
            if sup is None:
                 # This case might occur if listbox is out of sync with model
                 messagebox.showerror("Error", f"Support for Node {node_id} not found in model data.")
                 self._populate_supports_listbox() # Refresh listbox
                 return

            # Populate Node ID field
            self.sup_node_id_entry.delete(0, tk.END)
            self.sup_node_id_entry.insert(0, str(sup.node_id))
            self.sup_node_id_entry.config(state=tk.DISABLED) # Disable ID field during edit

            # Set the boolean vars which will update checkboxes via _update_support_dofs
            self.sup_dx_var.set(sup.dx)
            self.sup_dy_var.set(sup.dy)
            self.sup_rz_var.set(sup.rz)

            # Determine best matching dropdown type or set to custom
            sup_type = "Custom" # Default if no exact match
            if sup.dx and sup.dy and sup.rz: sup_type = "Fixed"
            elif sup.dx and sup.dy and not sup.rz: sup_type = "Pinned"
            elif not sup.dx and sup.dy and not sup.rz: sup_type = "Roller X (Free X)"
            elif sup.dx and not sup.dy and not sup.rz: sup_type = "Roller Y (Free Y)"

            self.sup_type_var.set(sup_type) # This triggers _update_support_dofs
            self.editing_support_node_id = node_id # Store the node ID being edited
            self.sup_add_update_button.config(text="Update Support", command=self._update_support)


            self.set_status(f"Editing Support at Node {node_id}. Modify constraints and click 'Add/Update Support'.")
            # NOTE: No button change needed, as "Add/Update" serves both purposes.

        except (IndexError, ValueError, TypeError):
             messagebox.showerror("Parsing Error", f"Could not parse Node ID from selected item:\n'{selected_string}'")
             self._reset_support_button() # Reset button on error
        except KeyError:
             messagebox.showerror("Error", f"Support for Node {node_id} not found for editing.")
             self._reset_support_button() # Reset button on error
        except Exception as e:
             messagebox.showerror("Error", f"Could not prepare support for editing: {e}")
             self._reset_support_button() # Reset button on error
    def _update_support(self):
        """Handles updating an existing support after editing."""
        if self.editing_support_node_id is None:
            self._reset_support_button()
            return

        node_id = self.editing_support_node_id # Get the ID being edited
        try:
            # Get the new constraint values
            dx = self.sup_dx_var.get()
            dy = self.sup_dy_var.get()
            rz = self.sup_rz_var.get()

            # Get the *existing* support object from the model
            support = self.model.get_support(node_id)
            if support is None:
                 # Should not happen if editing_support_node_id is set correctly
                 raise KeyError(f"Support for node {node_id} disappeared during edit.")

            # Update the attributes of the existing support object
            support.dx = dx
            support.dy = dy
            support.rz = rz

            status_msg = f"Support updated for Node {node_id}."
            self.set_status(status_msg)
            print(f"Support {node_id} updated. New state: {self.model.supports}")

            # Reset button and clear fields
            self._reset_support_button() # This re-enables ID entry and clears it

            # Update listbox and canvas
            self._populate_supports_listbox()
            self._redraw_canvas()
            self.update_idletasks()

        except (ValueError, TypeError) as e: # Should primarily catch errors from .get() if ID somehow bad
            messagebox.showerror("Input Error", f"Invalid input for Support update: {e}")
            self.set_status(f"Error updating support: {e}")
            self._reset_support_button() # Ensure reset on error
        except KeyError as e:
             messagebox.showerror("Error", f"Support {node_id} not found for update: {e}")
             self.set_status(f"Error updating support: Not found.")
             self._reset_support_button()
        except Exception as e:
             messagebox.showerror("Error", f"Could not update Support {node_id}: {e}")
             self.set_status(f"Error updating support: {e}")
             self._reset_support_button()

    def _update_support_dofs(self, *args):
        """Updates DOF variables and checkbutton states based on selected type."""
        sup_type = self.sup_type_var.get()
        dx, dy, rz = False, False, False
        enable_checks = tk.DISABLED

        if sup_type == "Fixed":
            dx, dy, rz = True, True, True
        elif sup_type == "Pinned":
            dx, dy, rz = True, True, False
        elif sup_type == "Roller X (Free X)":
            dx, dy, rz = False, True, False
        elif sup_type == "Roller Y (Free Y)":
            dx, dy, rz = True, False, False
        elif sup_type == "Custom":
            enable_checks = tk.NORMAL # Enable checkboxes for custom
            # Keep current values if switching to custom
            dx, dy, rz = self.sup_dx_var.get(), self.sup_dy_var.get(), self.sup_rz_var.get()

        self.sup_dx_var.set(dx)
        self.sup_dy_var.set(dy)
        self.sup_rz_var.set(rz)

        for check in self.sup_dof_checks:
            check.config(state=enable_checks)


    def _create_loads_tab(self):
        """Creates the GUI elements for the Loads input tab."""
        tab_frame = ttk.Frame(self.input_notebook, padding="10")
        self.input_notebook.add(tab_frame, text="Loads")

        # --- Input Frame ---
        input_frame = ttk.Frame(tab_frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Load Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.load_type_var = tk.StringVar(value="Nodal")
        load_types = ["Nodal", "Member Point", "Member UDL"]
        load_menu = ttk.OptionMenu(input_frame, self.load_type_var, load_types[0], *load_types,
                                   command=self._update_load_fields)
        load_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(input_frame, text="Load ID:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.load_id_entry = ttk.Entry(input_frame, width=15)
        self.load_id_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        # --- Disable ID entry and show next ID ---
        self.load_id_entry.config(state=tk.DISABLED)
        self.load_id_entry.insert(0, str(self.next_load_id))
        # --- End Disable ---

        ttk.Label(input_frame, text="Label (Optional):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.load_label_entry = ttk.Entry(input_frame, width=15)
        self.load_label_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

        # Frame for type-specific fields
        self.load_fields_frame = ttk.Frame(input_frame)
        self.load_fields_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)
        self.load_fields_frame.columnconfigure(1, weight=1)

        # --- Store button reference ---
        self.load_add_update_button = ttk.Button(input_frame, text="Add Load", command=self._add_load)
        self.load_add_update_button.grid(row=4, column=0, columnspan=2, pady=5)
        # --- End Store ---

        # --- Separator ---
        ttk.Separator(tab_frame, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, pady=5)

        # --- Listbox Frame ---
        list_frame = ttk.Frame(tab_frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.load_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, exportselection=False)
        list_scrollbar.config(command=self.load_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.load_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Action Buttons Frame ---
        action_frame = ttk.Frame(tab_frame)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        edit_button = ttk.Button(action_frame, text="Edit Selected", command=self._edit_selected_load)
        delete_button = ttk.Button(action_frame, text="Delete Selected", command=self._delete_selected_load)
        delete_button.pack(side=tk.RIGHT, padx=5)
        edit_button.pack(side=tk.RIGHT)

        # Initialize fields and listbox
        self._update_load_fields()
        self._populate_loads_listbox()


    def _get_selected_id_from_listbox(self, listbox: tk.Listbox) -> Optional[int]:
        """Helper to get the ID from the selected listbox item."""
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select an item from the list first.")
            return None
        try:
            selected_index = selection[0]
            selected_string = listbox.get(selected_index)
            # Parse ID (assumes format "ID: {id} ...")
            id_part = selected_string.split()[1] # Get the part after "ID:"
            item_id = int(id_part)
            return item_id
        except (IndexError, ValueError, TypeError):
            messagebox.showerror("Parsing Error", f"Could not parse ID from selected item:\n'{selected_string}'")
            return None
    def _on_exit(self):
        """Handles the window closing event."""
        if messagebox.askokcancel("Quit", "Do you really want to quit PyFrame2D?"):
            self.destroy() # Close the Tkinter window
    def _delete_selected_node(self):
        node_id = self._get_selected_id_from_listbox(self.node_listbox)
        if node_id is None: return

        # Check dependencies (simple check here, more robust needed for real app)
        dependents = [m.id for m in self.model.members.values() if m.start_node.id == node_id or m.end_node.id == node_id]
        if dependents:
            messagebox.showerror("Deletion Error", f"Cannot delete Node {node_id}. It is used by Member(s): {dependents}")
            return
        if node_id in self.model.supports:
            messagebox.showerror("Deletion Error", f"Cannot delete Node {node_id}. A support is defined on it.")
            return
        # Add check for nodal loads later

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete Node {node_id}?"):
            try:
                self.model.remove_node(node_id) # remove_node also removes support
                self._populate_nodes_listbox()
                self._redraw_canvas()
                self.update_idletasks()
                self.set_status(f"Node {node_id} deleted.")
                print(f"Node {node_id} deleted.")
            except KeyError:
                 messagebox.showerror("Error", f"Node {node_id} not found in model (unexpected).")
            except Exception as e:
                 messagebox.showerror("Error", f"Could not delete node: {e}")
    def _placeholder_command(self):
        # ... (Method remains the same) ...
        """Placeholder for menu commands not yet implemented."""
        print("Menu command executed (Not Implemented Yet)")
        self.set_status("Functionality not yet implemented.")
        messagebox.showinfo("Not Implemented", "This feature is not yet available.")


    def _populate_nodes_listbox(self):
        # --- Check 1: Does self.node_listbox exist? ---
        # If you get an AttributeError here, the listbox wasn't created correctly in _create_nodes_tab
        try:
            self.node_listbox.delete(0, tk.END) # Clear existing items
        except AttributeError:
            print("ERROR: self.node_listbox does not exist when _populate_nodes_listbox is called.")
            return # Cannot proceed

        # --- Check 2: Is self.model.nodes actually populated? ---
        # Add a print statement for debugging
        print(f"DEBUG: Populating nodes listbox. Model nodes: {self.model.nodes}")
        if not self.model.nodes:
            print("DEBUG: No nodes in model to populate.")
            return # Nothing to add

        # --- Check 3: Does the loop run and insert? ---
        try:
            # Sort by ID for consistent order
            sorted_nodes = sorted(self.model.nodes.values(), key=lambda n: n.id)
            print(f"DEBUG: Found {len(sorted_nodes)} nodes to add to listbox.") # Debug print
            for node in sorted_nodes:
                # Simple format, including ID for parsing
                display_str = f"ID: {node.id:<5} X: {node.x:<10.3f} Y: {node.y:<10.3f}"
                self.node_listbox.insert(tk.END, display_str)
                print(f"DEBUG: Inserted to listbox: {display_str}") # Debug print
            print("DEBUG: Finished populating nodes listbox.") # Debug print
        except Exception as e:
            print(f"ERROR during listbox population loop: {e}") # Catch potential errors


    def _populate_materials_listbox(self):
        print("DEBUG: ==> ENTERING _populate_materials_listbox") # Entry point
        try:
            # --- Check 1: Listbox existence ---
            print("DEBUG: Attempting to clear material_listbox...")
            self.material_listbox.delete(0, tk.END)
            print("DEBUG: material_listbox cleared.")

            # --- Check 2: Model data ---
            print(f"DEBUG: Model materials: {self.model.materials}")
            if not self.model.materials:
                print("DEBUG: No materials in model to populate.")
                print("DEBUG: <== EXITING _populate_materials_listbox (no data)")
                return

            # --- Check 3: Loop and insert ---
            sorted_mats = sorted(self.model.materials.values(), key=lambda m: m.id)
            print(f"DEBUG: Found {len(sorted_mats)} materials to add.")
            for i, mat in enumerate(sorted_mats):
                print(f"DEBUG: Processing material {i+1}/{len(sorted_mats)}: ID={mat.id}, Name={mat.name}")
                # Format E nicely if possible
                e_str = f"{mat.E:.3e}" # Default format
                print(f"DEBUG:  - Default E format: {e_str}")
                if UNITS_AVAILABLE:
                    print(f"DEBUG:  - Units available, trying format_value_unit...")
                    try:
                        e_str = format_value_unit(mat.E, "Pa", 3)
                        print(f"DEBUG:  - Formatted E: {e_str}")
                    except Exception as fmt_e:
                        # Catch specific formatting errors if needed
                        print(f"DEBUG:  - Error formatting E for mat {mat.id}: {fmt_e}. Using default.")
                        pass # Fallback to scientific notation
                else:
                     print("DEBUG:  - Units not available, using default E format.")

                display_str = f"ID: {mat.id:<5} Name: {mat.name:<15} E: {e_str}"
                print(f"DEBUG:  - Inserting to listbox: {display_str}")
                self.material_listbox.insert(tk.END, display_str)
                print(f"DEBUG:  - Inserted material {mat.id}.")

            print("DEBUG: Finished material listbox population loop.")

        except AttributeError:
            print("ERROR: self.material_listbox does not exist when _populate_materials_listbox is called.")
        except Exception as e:
            print(f"ERROR inside _populate_materials_listbox loop/logic: {e}") # Catch other potential errors

        print("DEBUG: <== EXITING _populate_materials_listbox (normally)") # Normal exit
    def _populate_sections_listbox(self):
        self.section_listbox.delete(0, tk.END)
        sorted_secs = sorted(self.model.sections.values(), key=lambda s: s.id)
        for sec in sorted_secs:
            type_name = sec.__class__.__name__.replace("Profile","") # Shorten type name
            display_str = f"ID: {sec.id:<5} Name: {sec.name:<15} Type: {type_name}"
            self.section_listbox.insert(tk.END, display_str)

    def _populate_members_listbox(self):
        self.member_listbox.delete(0, tk.END)
        sorted_mems = sorted(self.model.members.values(), key=lambda m: m.id)
        for mem in sorted_mems:
            display_str = (f"ID: {mem.id:<5} Nodes: {mem.start_node.id:>3}->{mem.end_node.id:<3} "
                           f"Mat: {mem.material.id:<4} Sec: {mem.section.id:<4}")
            self.member_listbox.insert(tk.END, display_str)

    def _populate_supports_listbox(self):
        self.support_listbox.delete(0, tk.END)
        # Supports are keyed by node_id, sort by that key
        for node_id in sorted(self.model.supports.keys()):
            sup = self.model.supports[node_id]
            constraints = []
            if sup.dx: constraints.append("DX")
            if sup.dy: constraints.append("DY")
            if sup.rz: constraints.append("RZ")
            constraint_str = "+".join(constraints) if constraints else "Free"
            display_str = f"Node ID: {sup.node_id:<5} Restrains: {constraint_str}"
            self.support_listbox.insert(tk.END, display_str)

    def _populate_loads_listbox(self):
        self.load_listbox.delete(0, tk.END)
        sorted_loads = sorted(self.model.loads.values(), key=lambda ld: ld.id)
        for load in sorted_loads:
            type_name = load.__class__.__name__.replace("Load","")
            target = ""
            if isinstance(load, NodalLoad): target = f"Node {load.node_id}"
            elif isinstance(load, MemberLoad): target = f"Member {load.member_id}"
            label_part = f" ({load.label})" if load.label else ""
            display_str = f"ID: {load.id:<5} Type: {type_name:<12} Target: {target:<12}{label_part}"
            self.load_listbox.insert(tk.END, display_str)

    def _update_all_listboxes(self):
        """Updates all listboxes, typically after loading or clearing."""
        self._populate_nodes_listbox()
        self._populate_materials_listbox()
        self._populate_sections_listbox()
        self._populate_members_listbox()
        self._populate_supports_listbox()
        self._populate_loads_listbox()

    def _delete_selected_material(self):
        mat_id = self._get_selected_id_from_listbox(self.material_listbox)
        if mat_id is None: return

        dependents = [m.id for m in self.model.members.values() if m.material.id == mat_id]
        if dependents:
             messagebox.showerror("Deletion Error", f"Cannot delete Material {mat_id}. It is used by Member(s): {dependents}")
             return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete Material {mat_id}?"):
            try:
                del self.model.materials[mat_id] # Simple dictionary removal
                self._populate_materials_listbox()
                self.set_status(f"Material {mat_id} deleted.")
                print(f"Material {mat_id} deleted.")
            except KeyError:
                 messagebox.showerror("Error", f"Material {mat_id} not found in model (unexpected).")
            except Exception as e:
                 messagebox.showerror("Error", f"Could not delete material: {e}")

    def _delete_selected_section(self):
        sec_id = self._get_selected_id_from_listbox(self.section_listbox)
        if sec_id is None: return

        dependents = [m.id for m in self.model.members.values() if m.section.id == sec_id]
        if dependents:
             messagebox.showerror("Deletion Error", f"Cannot delete Section {sec_id}. It is used by Member(s): {dependents}")
             return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete Section {sec_id}?"):
            try:
                del self.model.sections[sec_id]
                self._populate_sections_listbox()
                self.set_status(f"Section {sec_id} deleted.")
                print(f"Section {sec_id} deleted.")
            except KeyError:
                 messagebox.showerror("Error", f"Section {sec_id} not found in model (unexpected).")
            except Exception as e:
                 messagebox.showerror("Error", f"Could not delete section: {e}")

    def _delete_selected_member(self):
        mem_id = self._get_selected_id_from_listbox(self.member_listbox)
        if mem_id is None: return

        # Check dependencies (member loads)
        dependents = [ld.id for ld in self.model.loads.values() if isinstance(ld, MemberLoad) and ld.member_id == mem_id]
        if dependents:
            messagebox.showerror("Deletion Error", f"Cannot delete Member {mem_id}. It has Load(s) applied: {dependents}")
            return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete Member {mem_id}?"):
            try:
                self.model.remove_member(mem_id)
                self._populate_members_listbox()
                self._redraw_canvas()
                self.update_idletasks()
                self.set_status(f"Member {mem_id} deleted.")
                print(f"Member {mem_id} deleted.")
            except KeyError:
                 messagebox.showerror("Error", f"Member {mem_id} not found in model (unexpected).")
            except Exception as e:
                 messagebox.showerror("Error", f"Could not delete member: {e}")

    def _delete_selected_support(self):
        # Support listbox items start with "Node ID: {id}"
        selection = self.support_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a support from the list first.")
            return None
        try:
            selected_index = selection[0]
            selected_string = self.support_listbox.get(selected_index)
            # Parse ID
            node_id = int(selected_string.split()[2]) # Get the ID after "Node ID:"

            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the support at Node {node_id}?"):
                self.model.remove_support(node_id) # remove_support handles KeyError
                self._populate_supports_listbox()
                self._redraw_canvas()
                self.update_idletasks()
                self.set_status(f"Support at Node {node_id} deleted.")
                print(f"Support at Node {node_id} deleted.")

        except (IndexError, ValueError, TypeError):
             messagebox.showerror("Parsing Error", f"Could not parse Node ID from selected item:\n'{selected_string}'")
        except KeyError: # Should be handled by remove_support, but just in case
             messagebox.showerror("Error", f"Support for Node {node_id} not found in model (unexpected).")
        except Exception as e:
             messagebox.showerror("Error", f"Could not delete support: {e}")


    def _delete_selected_load(self):
        load_id = self._get_selected_id_from_listbox(self.load_listbox)
        if load_id is None: return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete Load {load_id}?"):
            try:
                self.model.remove_load(load_id)
                self._populate_loads_listbox()
                self._redraw_canvas()
                self.update_idletasks()
                self.set_status(f"Load {load_id} deleted.")
                print(f"Load {load_id} deleted.")
            except KeyError:
                 messagebox.showerror("Error", f"Load {load_id} not found in model (unexpected).")
            except Exception as e:
                 messagebox.showerror("Error", f"Could not delete load: {e}")

    # --- Placeholder Edit Methods ---
    # These just populate the fields for now

    def _edit_selected_node(self):
        node_id = self._get_selected_id_from_listbox(self.node_listbox)
        if node_id is None:
            self._reset_node_button() # Reset button if selection lost
            return
        try:
            # ... (get node, clear/populate fields) ...
            self.node_id_entry.config(state=tk.DISABLED) # Disable ID editing

            # --- Set editing state and change button ---
            self.editing_node_id = node_id
            self.node_add_update_button.config(text="Update Node", command=self._update_node)
            # --- End Change ---

            self.set_status(f"Editing Node {node_id}. Modify fields and click 'Update Node'.")
        except KeyError:
             messagebox.showerror("Error", f"Node {node_id} not found for editing.")
             self._reset_node_button() # Reset on error
        except Exception as e:
             messagebox.showerror("Error", f"Could not prepare node for editing: {e}")
             self._reset_node_button() # Reset on error
    def _reset_node_button(self):
        """Resets the Node tab's button to 'Add Node' state."""
        self.editing_node_id = None
        if hasattr(self, 'node_add_update_button'):
            self.node_add_update_button.config(text="Add Node", command=self._add_node)
        self._update_next_id_display() # Call helper

    def _update_node(self):
        """Handles updating an existing node after editing."""
        if self.editing_node_id is None:
            # Should not happen if button command is correct, but safeguard
            self._reset_node_button()
            return

        try:
            # ID is not changed, read X and Y
            x = float(self.node_x_entry.get())
            y = float(self.node_y_entry.get())

            # Get the existing node object
            node = self.model.get_node(self.editing_node_id)

            # Update attributes
            node.x = x
            node.y = y

            self.set_status(f"Node {self.editing_node_id} updated successfully.")
            print(f"Node {self.editing_node_id} updated. New state: {self.model.nodes}")

            # Reset editing state and button
            self._reset_node_button()
            # Clear fields
            self.node_id_entry.delete(0, tk.END); self.node_x_entry.delete(0, tk.END); self.node_y_entry.delete(0, tk.END)
            self.node_id_entry.config(state=tk.NORMAL) # Re-enable ID field
            self.node_id_entry.focus()

            # Update listbox and canvas
            self._populate_nodes_listbox()
            self._redraw_canvas()
            self.update_idletasks()

        except (ValueError, TypeError) as e:
            messagebox.showerror("Input Error", f"Invalid input for Node update: {e}")
            self.set_status(f"Error updating node: {e}")
        except KeyError:
             messagebox.showerror("Error", f"Node {self.editing_node_id} not found for update (unexpected).")
             self.set_status(f"Error updating node: Not found.")
             self._reset_node_button() # Reset button on error
        except Exception as e:
             messagebox.showerror("Error", f"Could not update Node {self.editing_node_id}: {e}")
             self.set_status(f"Error updating node: {e}")
             self._reset_node_button() # Reset button on error
    def _reset_section_button(self):
        """Resets the Section tab's button to 'Add Section' state."""
        self.editing_section_id = None
        if hasattr(self, 'sec_add_update_button'):
            self.sec_add_update_button.config(text="Add Section", command=self._add_section)
        # Also reset section type dropdown to default? Optional.
        # if hasattr(self, 'section_type_var'): self.section_type_var.set("Rectangular")
        self._update_next_id_display() # Update ID field

    def _on_tab_change(self, event):
        """Callback when the input notebook tab changes."""
        # Reset button states for all tabs to avoid being stuck in 'Update' mode
        self._reset_node_button()
        self._reset_material_button() # Add this later
        self._reset_section_button()  # Add this later
        self._reset_member_button()   # Add this later
        self._reset_support_button()  # Add this later
        self._reset_load_button()     # Add this later

        # Clear editing state flags
        self.editing_node_id = None
        self.editing_material_id = None
        self.editing_load_id = None
        self.editing_section_id = None
        self.editing_member_id = None

        # ... clear other editing flags ...

        current_tab_index = self.input_notebook.index(self.input_notebook.select())
        # You could potentially clear input fields here too if desired
        # print(f"DEBUG: Switched to tab index {current_tab_index}")
        self.set_status("Ready") # Reset status bar
    def _reset_material_button(self):
        """Resets the Material tab's button to 'Add Material' state."""
        self.editing_material_id = None
        if hasattr(self, 'mat_add_update_button'): # Check for the button reference attribute
            self.mat_add_update_button.config(text="Add Material", command=self._add_material)
        # ID display update moved to _update_next_id_display

    def _reset_section_button(self):
        """Resets the Section tab's button to 'Add Section' state."""
        self.editing_section_id = None # Assumes you add this flag later
        if hasattr(self, 'sec_add_update_button'): # Check for the button reference attribute
            self.sec_add_update_button.config(text="Add Section", command=self._add_section)
        # ID display update moved to _update_next_id_display

    def _reset_member_button(self):
        """Resets the Member tab's button to 'Add Member' state."""
        self.editing_member_id = None
        if hasattr(self, 'mem_add_update_button'):
            self.mem_add_update_button.config(text="Add Member", command=self._add_member)
        self._update_next_id_display() # Update ID field

    def _reset_support_button(self):
        """Resets the Support tab state after edit/update or tab change."""
        # Supports don't have an independent editing ID state like others
        # Button text "Add/Update Support" doesn't change
        if hasattr(self, 'sup_add_update_button'):
            # Ensure command is correct (it should always be _add_support)
            self.sup_add_update_button.config(command=self._add_support)
        if hasattr(self, 'sup_node_id_entry'):
             # Re-enable the Node ID entry and clear it
            self.sup_node_id_entry.config(state=tk.NORMAL)
            self.sup_node_id_entry.delete(0, tk.END)
        # Reset dropdown/checkboxes to default (optional, or let _update_support_dofs handle)
        # if hasattr(self, 'sup_type_var'): self.sup_type_var.set("Pinned")
        # self._update_support_dofs() # Update checkboxes based on dropdown

    def _reset_load_button(self):
        """Resets the Load tab's button to 'Add Load' state."""
        self.editing_load_id = None
        if hasattr(self, 'load_add_update_button'):
            self.load_add_update_button.config(text="Add Load", command=self._add_load)
        # Reset load type dropdown? Optional.
        # if hasattr(self, 'load_type_var'): self.load_type_var.set("Nodal")
        # self._update_load_fields() # Clear specific fields
        self._update_next_id_display() # Update ID field

    # --- Ensure you also have the update ID display helper ---
    def _update_next_id_display(self):
        """Updates the disabled ID entry fields to show the next available ID."""
        if hasattr(self, 'node_id_entry'):
            self.node_id_entry.config(state=tk.NORMAL)
            self.node_id_entry.delete(0, tk.END)
            self.node_id_entry.insert(0, str(self.next_node_id))
            self.node_id_entry.config(state=tk.DISABLED)
        if hasattr(self, 'mat_id_entry'):
            self.mat_id_entry.config(state=tk.NORMAL)
            self.mat_id_entry.delete(0, tk.END)
            self.mat_id_entry.insert(0, str(self.next_material_id))
            self.mat_id_entry.config(state=tk.DISABLED)
        if hasattr(self, 'sec_id_entry'):
            self.sec_id_entry.config(state=tk.NORMAL)
            self.sec_id_entry.delete(0, tk.END)
            self.sec_id_entry.insert(0, str(self.next_section_id))
            self.sec_id_entry.config(state=tk.DISABLED)
        if hasattr(self, 'mem_id_entry'):
            self.mem_id_entry.config(state=tk.NORMAL)
            self.mem_id_entry.delete(0, tk.END)
            self.mem_id_entry.insert(0, str(self.next_member_id))
            self.mem_id_entry.config(state=tk.DISABLED)
        # Support node ID entry is manually entered, not auto-incremented
        if hasattr(self, 'load_id_entry'):
            self.load_id_entry.config(state=tk.NORMAL)
            self.load_id_entry.delete(0, tk.END)
            self.load_id_entry.insert(0, str(self.next_load_id))
            self.load_id_entry.config(state=tk.DISABLED)

    def _reset_all_buttons_and_ids(self):
        """Helper to reset all edit buttons and update next ID displays."""
        self._reset_node_button()
        self._reset_material_button()
        self._reset_section_button()
        self._reset_member_button()
        self._reset_support_button()
        self._reset_load_button()
        self._update_next_id_display() # Ensure all ID fields show correct next value


    def _edit_selected_load(self):
        """Populates load input fields with data from the selected load."""
        load_id = self._get_selected_id_from_listbox(self.load_listbox)
        if load_id is None:
            self._reset_load_button()
            return
        try:
            load = self.model.get_load(load_id)
            # Determine type string for dropdown
            load_type_name = load.__class__.__name__
            load_type_display = "Nodal" # Default
            if load_type_name == "MemberPointLoad": load_type_display = "Member Point"
            elif load_type_name == "MemberUDLoad": load_type_display = "Member UDL"

            # Set common fields
            self.load_id_entry.config(state=tk.NORMAL)
            self.load_id_entry.delete(0, tk.END); self.load_label_entry.delete(0, tk.END)
            self.load_id_entry.insert(0, str(load.id))
            self.load_id_entry.config(state=tk.DISABLED)
            self.load_label_entry.insert(0, str(load.label))

            # --- Update type dropdown FIRST to recreate fields ---
            self.load_type_var.set(load_type_display)
            # --- Force Tkinter to process the update NOW ---
            self.update_idletasks()
            # --- END UPDATE ---

            # --- Now populate the NEWLY CREATED specific fields ---
            if isinstance(load, NodalLoad):
                # Check attributes exist before using
                if hasattr(self, 'load_nodal_node_entry'): self.load_nodal_node_entry.insert(0, str(load.node_id))
                if hasattr(self, 'load_nodal_fx_entry'): self.load_nodal_fx_entry.delete(0,tk.END); self.load_nodal_fx_entry.insert(0, str(load.fx))
                if hasattr(self, 'load_nodal_fy_entry'): self.load_nodal_fy_entry.delete(0,tk.END); self.load_nodal_fy_entry.insert(0, str(load.fy))
                if hasattr(self, 'load_nodal_mz_entry'): self.load_nodal_mz_entry.delete(0,tk.END); self.load_nodal_mz_entry.insert(0, str(load.mz))
            elif isinstance(load, MemberPointLoad):
                if hasattr(self, 'load_pt_mem_entry'): self.load_pt_mem_entry.insert(0, str(load.member_id))
                if hasattr(self, 'load_pt_px_entry'): self.load_pt_px_entry.delete(0,tk.END); self.load_pt_px_entry.insert(0, str(load.px))
                if hasattr(self, 'load_pt_py_entry'): self.load_pt_py_entry.delete(0,tk.END); self.load_pt_py_entry.insert(0, str(load.py))
                if hasattr(self, 'load_pt_pos_entry'): self.load_pt_pos_entry.insert(0, str(load.position))
            elif isinstance(load, MemberUDLoad):
                if hasattr(self, 'load_udl_mem_entry'): self.load_udl_mem_entry.insert(0, str(load.member_id)) # <<< Should work now
                if hasattr(self, 'load_udl_wx_entry'): self.load_udl_wx_entry.delete(0,tk.END); self.load_udl_wx_entry.insert(0, str(load.wx))
                if hasattr(self, 'load_udl_wy_entry'): self.load_udl_wy_entry.delete(0,tk.END); self.load_udl_wy_entry.insert(0, str(load.wy))
            # Add elif for other types if needed
            # --- End Populate ---

            # Set editing state and change button
            self.editing_load_id = load_id
            self.load_add_update_button.config(text="Update Load", command=self._update_load)

            self.set_status(f"Editing Load {load_id}. Modify fields and click 'Update Load'.")

        except KeyError:
             messagebox.showerror("Error", f"Load {load_id} not found for editing.")
             self._reset_load_button()
        except AttributeError as e: # Catch errors if expected entry attribute is missing
             messagebox.showerror("Error", f"GUI Error preparing load fields for edit: {e}")
             self._reset_load_button()
        except Exception as e:
             messagebox.showerror("Error", f"Could not prepare load for editing: {e}")
             self._reset_load_button()

    def _update_load_fields(self, *args):
        """Updates load input fields based on selected type."""
        for widget in self.load_fields_frame.winfo_children():
            widget.destroy()

        load_type = self.load_type_var.get()

        if load_type == "Nodal":
            ttk.Label(self.load_fields_frame, text="Node ID:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_nodal_node_entry = ttk.Entry(self.load_fields_frame, width=10)
            self.load_nodal_node_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.load_fields_frame, text="Fx (N):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_nodal_fx_entry = ttk.Entry(self.load_fields_frame, width=10); self.load_nodal_fx_entry.insert(0,"0.0")
            self.load_nodal_fx_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.load_fields_frame, text="Fy (N):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_nodal_fy_entry = ttk.Entry(self.load_fields_frame, width=10); self.load_nodal_fy_entry.insert(0,"0.0")
            self.load_nodal_fy_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.load_fields_frame, text="Mz (Nm):").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_nodal_mz_entry = ttk.Entry(self.load_fields_frame, width=10); self.load_nodal_mz_entry.insert(0,"0.0")
            self.load_nodal_mz_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.EW)
        elif load_type == "Member Point":
            ttk.Label(self.load_fields_frame, text="Member ID:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_pt_mem_entry = ttk.Entry(self.load_fields_frame, width=10)
            self.load_pt_mem_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.load_fields_frame, text="Px Local (N):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_pt_px_entry = ttk.Entry(self.load_fields_frame, width=10); self.load_pt_px_entry.insert(0,"0.0")
            self.load_pt_px_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.load_fields_frame, text="Py Local (N):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_pt_py_entry = ttk.Entry(self.load_fields_frame, width=10); self.load_pt_py_entry.insert(0,"0.0")
            self.load_pt_py_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.load_fields_frame, text="Position (m):").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_pt_pos_entry = ttk.Entry(self.load_fields_frame, width=10)
            self.load_pt_pos_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.EW)
        elif load_type == "Member UDL":
            ttk.Label(self.load_fields_frame, text="Member ID:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_udl_mem_entry = ttk.Entry(self.load_fields_frame, width=10)
            self.load_udl_mem_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.load_fields_frame, text="wx Local (N/m):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_udl_wx_entry = ttk.Entry(self.load_fields_frame, width=10); self.load_udl_wx_entry.insert(0,"0.0")
            self.load_udl_wx_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
            ttk.Label(self.load_fields_frame, text="wy Local (N/m):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
            self.load_udl_wy_entry = ttk.Entry(self.load_fields_frame, width=10); self.load_udl_wy_entry.insert(0,"0.0")
            self.load_udl_wy_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)

    # --- Action Methods (Callbacks for Buttons) ---




    # --- Utility Methods ---

    def _new_project(self):
        """Clears the current model."""
        try:
            if messagebox.askokcancel("New Project", "..."):
                self.model = StructuralModel()
                self.analysis_results = None
                self._clear_results_display()
                self.current_file_path = None
                self._update_window_title()
                self.set_status("New project started.")
                print("Model Cleared.")
                self._redraw_canvas()
                self.update_idletasks() # <<< ADD THIS
        except Exception as e:
             messagebox.showerror("Error", f"Could not clear model: {e}")



    def set_status(self, text: str):
        # ... (Method remains the same) ...
        """Updates the text displayed in the status bar."""
        self.status_bar.config(text=text)





# --- Allow running this file directly for testing ---
if __name__ == '__main__':
    if not CORE_AVAILABLE:
        print("ERROR: Cannot run GUI without core model components.")
    else:
        app = MainApplicationWindow()
        app.mainloop()