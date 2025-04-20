# gui/main_window.py

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Frame, Label, Entry, Button, OptionMenu, StringVar
import math # May be needed for calculations if not using units module directly often

# Import core model and analysis components
# Use try-except for robustness, although GUI likely won't launch if core is missing
try:
    from core.model import (StructuralModel, Node, Material, SectionProfile,
                            RectangularProfile, SquareProfile, IBeamProfile,
                            Member, Support, Load, NodalLoad, MemberLoad,
                            MemberPointLoad, MemberUDLoad)
    # Assume analysis results class might be needed later for displaying results
    # from core.analysis import AnalysisResults
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"ERROR importing core model components: {e}")
    # Handle missing core components gracefully (e.g., disable features)
    CORE_AVAILABLE = False
    # Define dummies so Pylance doesn't complain below if CORE_AVAILABLE is checked
    class StructuralModel: pass
    class Node: pass
    # ... etc.

# Import unit parsing functions
try:
    from units.units import parse_value_unit, format_value_unit
    UNITS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING importing units module: {e}. Unit parsing will be disabled.")
    UNITS_AVAILABLE = False
    def parse_value_unit(value_str, unit): # Dummy function
        print("Unit parsing disabled.")
        return float(value_str) # Simple float conversion as fallback

CANVAS_PADDING = 20 # Pixels around the drawing area
NODE_RADIUS = 3     # Pixels
SUPPORT_SIZE = 10   # Pixels (approx size of symbol)
LOAD_ARROW_LENGTH = 30 # Pixels
LOAD_ARROW_WIDTH = 2
LOAD_COLOR = "red"
SUPPORT_COLOR = "blue"
MEMBER_COLOR = "black"
NODE_COLOR = "black"
from typing import Optional


class MainApplicationWindow(tk.Tk):
    """
    Main application window class using Tkinter.
    Sets up the basic menu, status bar, input forms, and main frame.
    """
    def __init__(self):
        super().__init__()

        if not CORE_AVAILABLE:
            self.withdraw() # Hide the main window
            messagebox.showerror("Startup Error", "Core model components failed to load. Application cannot start.")
            self.destroy()
            return

        self.title("PyFrame2D - Modular 2D Frame Analysis")
        self.geometry("900x700") # Increased size slightly

        # --- Core Model Instance ---
        self.model = StructuralModel() # Create an instance to hold the data

        self._create_menu()
        self._create_main_layout() # Use a more structured layout
        self._create_status_bar()

        self.protocol("WM_DELETE_WINDOW", self._on_exit)

    def _on_canvas_resize(self, event):
        """Callback when the canvas size changes."""
        # Debounce or add delay if needed, but redraw directly for now
        self._redraw_canvas()

    def _redraw_canvas(self):
        """Clears and redraws the entire model on the canvas."""
        self.canvas.delete("all") # Clear previous drawing items

        if not self.model or not self.model.nodes:
            # Display message if model is empty
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1: # Avoid drawing before window is ready
                self.canvas.create_text(
                    canvas_width / 2, canvas_height / 2,
                    text="Model is empty. Add nodes and members.",
                    fill="grey", font=("Segoe UI", 10), tags="message"
                )
            return

        # 1. Calculate Model Bounds and Transformation
        try:
            min_x, max_x, min_y, max_y = self._get_model_bounds()
            transform = self._get_canvas_transform(min_x, max_x, min_y, max_y)
            if not transform: # Handle case where bounds are invalid
                 return
        except ValueError: # Handle case with no nodes or single node
            self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2,
                                    text="Add more nodes to visualize.", fill="grey", tags="message")
            return


        # 2. Draw Components
        self._draw_members(transform)
        self._draw_nodes(transform)
        self._draw_supports(transform)
        self._draw_loads(transform)

        self.set_status("Model view updated.")

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
        """Draws nodes on the canvas."""
        for node in self.model.nodes.values():
            cx, cy = self._map_coords(node.x, node.y, transform)
            r = NODE_RADIUS
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=NODE_COLOR, outline=NODE_COLOR, tags=("node", f"node_{node.id}"))
            # Optional: Add node ID text
            # self.canvas.create_text(cx + r + 2, cy - r - 2, text=f"{node.id}", anchor=tk.NW, fill="grey", font=("Segoe UI", 8), tags=("node_label", f"node_label_{node.id}"))


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
        s = SUPPORT_SIZE / 2.0 # Half-size for drawing relative to node center
        for node_id, support in self.model.supports.items():
             if node_id not in self.model.nodes: continue # Skip if node doesn't exist
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
        try:
            node_id = int(self.node_id_entry.get())
            x = float(self.node_x_entry.get())
            y = float(self.node_y_entry.get())

            node = Node(id=node_id, x=x, y=y)
            self.model.add_node(node)

            self.set_status(f"Node {node_id} added successfully.")
            # Clear entries
            self.node_id_entry.delete(0, tk.END)
            self.node_x_entry.delete(0, tk.END)
            self.node_y_entry.delete(0, tk.END)
            self.node_id_entry.focus() # Move cursor back to first field
            print(f"Model Nodes: {self.model.nodes}") # Debug print
            self._redraw_canvas() # <<< ADD THIS
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for Node: {e}")
            self.set_status(f"Error adding node: {e}")
        except TypeError as e:
             messagebox.showerror("Input Error", f"Invalid input type for Node: {e}")
             self.set_status(f"Error adding node: {e}")
        except Exception as e: # Catch other errors like duplicate ID from model.add_node
             # --- INDENT THESE LINES ---
             messagebox.showerror("Error", f"Could not add Node: {e}")
             self.set_status(f"Error adding node: {e}")
             # --- END INDENTATION ---

    def _add_material(self):
        """Reads material data, validates, creates Material object, adds to model."""
        try:
            mat_id = int(self.mat_id_entry.get())
            name = self.mat_name_entry.get()
            e_str = self.mat_e_entry.get() # Get E as string

            # Try parsing E using units module if available
            youngs_modulus_value = e_str # Pass string directly to Material constructor

            mat = Material(id=mat_id, name=name, youngs_modulus=youngs_modulus_value)
            self.model.add_material(mat)

            self.set_status(f"Material {mat_id} ({name}) added successfully.")
            # Clear entries
            self.mat_id_entry.delete(0, tk.END)
            self.mat_name_entry.delete(0, tk.END)
            self.mat_e_entry.delete(0, tk.END)
            self.mat_id_entry.focus()
            print(f"Model Materials: {self.model.materials}") # Debug print
            # No redraw needed
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for Material: {e}")
            self.set_status(f"Error adding material: {e}")
        except TypeError as e:
             messagebox.showerror("Input Error", f"Invalid input type for Material: {e}")
             self.set_status(f"Error adding material: {e}")
        except Exception as e:
             # --- INDENT THESE LINES ---
             messagebox.showerror("Error", f"Could not add Material: {e}")
             self.set_status(f"Error adding material: {e}")
             # --- END INDENTATION ---

    def _add_section(self):
        """Reads section data, validates, creates Section object, adds to model."""
        try:
            sec_id = int(self.sec_id_entry.get())
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
                self.set_status(f"Section {sec_id} ({name}) added successfully.")
                # Clear common entries
                self.sec_id_entry.delete(0, tk.END)
                self.sec_name_entry.delete(0, tk.END)
                # Clear specific entries by updating fields (easier than tracking)
                self._update_section_fields() # This clears specific fields
                self.sec_id_entry.focus()
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

    def _add_member(self):
        """Reads member data, validates, creates Member object, adds to model."""
        try:
            mem_id = int(self.mem_id_entry.get())
            start_node_id = int(self.mem_start_node_entry.get())
            end_node_id = int(self.mem_end_node_entry.get())
            material_id = int(self.mem_material_entry.get())
            section_id = int(self.mem_section_entry.get())

            # Get referenced objects from model
            start_node = self.model.get_node(start_node_id) # Raises KeyError if not found
            end_node = self.model.get_node(end_node_id)
            material = self.model.get_material(material_id)
            section = self.model.get_section(section_id)

            member = Member(mem_id, start_node, end_node, material, section)
            self.model.add_member(member)

            self.set_status(f"Member {mem_id} added successfully.")
            # Clear entries
            self.mem_id_entry.delete(0, tk.END); self.mem_start_node_entry.delete(0, tk.END);
            self.mem_end_node_entry.delete(0, tk.END); self.mem_material_entry.delete(0, tk.END);
            self.mem_section_entry.delete(0, tk.END)
            self.mem_id_entry.focus()
            print(f"Model Members: {self.model.members}") # Debug print
            self._redraw_canvas() # Redraw after adding member
        except (ValueError, TypeError) as e: # Catches int/float conversion errors too
            messagebox.showerror("Input Error", f"Invalid input for Member: {e}")
            self.set_status(f"Error adding member: {e}")
        except KeyError as e: # Catch errors from get_node/mat/sec
             messagebox.showerror("Input Error", f"Invalid reference ID for Member: {e}")
             self.set_status(f"Error adding member: {e}")
        except Exception as e:
             # --- INDENT THESE LINES ---
             messagebox.showerror("Error", f"Could not add Member: {e}")
             self.set_status(f"Error adding member: {e}")
             # --- END INDENTATION ---

    def _add_support(self):
        """Reads support data, validates, creates Support object, adds to model."""
        try:
            node_id = int(self.sup_node_id_entry.get())
            dx = self.sup_dx_var.get()
            dy = self.sup_dy_var.get()
            rz = self.sup_rz_var.get()

            support = Support(node_id, dx, dy, rz)
            # Use a temporary variable to store the message before potentially raising error
            status_msg = f"Support added/updated for Node {node_id}."
            self.model.add_support(support) # add_support handles duplicates

            self.set_status(status_msg)
            # Clear entry
            self.sup_node_id_entry.delete(0, tk.END)
            self.sup_node_id_entry.focus()
            print(f"Model Supports: {self.model.supports}") # Debug print
            self._redraw_canvas() # Redraw after adding support
        except (ValueError, TypeError) as e:
            messagebox.showerror("Input Error", f"Invalid input for Support: {e}")
            self.set_status(f"Error adding support: {e}")
        except Exception as e:
             # --- INDENT THESE LINES ---
             messagebox.showerror("Error", f"Could not add Support: {e}")
             self.set_status(f"Error adding support: {e}")
             # --- END INDENTATION ---

    def _add_load(self):
        """Reads load data, validates, creates Load object, adds to model."""
        try:
            load_id = int(self.load_id_entry.get())
            label = self.load_label_entry.get()
            load_type = self.load_type_var.get()

            load = None
            if load_type == "Nodal":
                node_id = int(self.load_nodal_node_entry.get())
                fx = float(self.load_nodal_fx_entry.get())
                fy = float(self.load_nodal_fy_entry.get())
                mz = float(self.load_nodal_mz_entry.get())
                load = NodalLoad(load_id, node_id, fx, fy, mz, label)
            elif load_type == "Member Point":
                mem_id = int(self.load_pt_mem_entry.get())
                px = float(self.load_pt_px_entry.get())
                py = float(self.load_pt_py_entry.get())
                pos = float(self.load_pt_pos_entry.get())
                load = MemberPointLoad(load_id, mem_id, px, py, pos, label)
            elif load_type == "Member UDL":
                mem_id = int(self.load_udl_mem_entry.get())
                wx = float(self.load_udl_wx_entry.get())
                wy = float(self.load_udl_wy_entry.get())
                load = MemberUDLoad(load_id, mem_id, wx, wy, label)
            else:
                 raise ValueError(f"Selected load type '{load_type}' not handled.")

            if load:
                 self.model.add_load(load)
                 self.set_status(f"Load {load_id} added successfully.")
                 # Clear common fields
                 self.load_id_entry.delete(0, tk.END)
                 self.load_label_entry.delete(0, tk.END)
                 # Clear specific fields
                 self._update_load_fields() # This clears specific fields
                 self.load_id_entry.focus()
                 print(f"Model Loads: {self.model.loads}") # Debug print
                 self._redraw_canvas() # Redraw after adding load

        except (ValueError, TypeError) as e:
            messagebox.showerror("Input Error", f"Invalid input for Load: {e}")
            self.set_status(f"Error adding load: {e}")
        except AttributeError as e:
             messagebox.showerror("Input Error", f"Missing input field for selected load type: {e}")
             self.set_status(f"Error adding load: Incomplete fields.")
        except Exception as e:
             # --- INDENT THESE LINES ---
             messagebox.showerror("Error", f"Could not add Load: {e}")
             self.set_status(f"Error adding load: {e}")
             # --- END INDENTATION ---

    # --- Utility Methods ---

    def _new_project(self):
        """Clears the current model."""
        # No exception handling strictly needed here unless model creation fails,
        # but adding it for consistency doesn't hurt.
        try:
            if messagebox.askokcancel("New Project", "Clear current model and start new project?"):
                 self.model = StructuralModel() # Create new empty model
                 # Optionally clear listboxes or other display elements later
                 self.set_status("New project started.")
                 print("Model Cleared.")
                 self._redraw_canvas() # Redraw to clear canvas
        except Exception as e:
            # --- INDENT THESE LINES ---
            messagebox.showerror("Error", f"Could not create new project: {e}")
            self.set_status(f"Error creating new project: {e}")
            # --- END INDENTATION ---

    def _create_menu(self):
        """Creates the main menu bar and its submenus."""
        # ... (Menu creation code remains the same as before) ...
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        # --- File Menu ---
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self._new_project) # Link to new method
        file_menu.add_command(label="Open Project...", command=self._placeholder_command)
        file_menu.add_command(label="Save Project", command=self._placeholder_command)
        file_menu.add_command(label="Save Project As...", command=self._placeholder_command)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)

        # --- Edit Menu (Now mainly for context, adding done via tabs) ---
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo (Placeholder)", command=self._placeholder_command)
        edit_menu.add_command(label="Redo (Placeholder)", command=self._placeholder_command)

        # --- Analyze Menu ---
        analyze_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Analyze", menu=analyze_menu)
        analyze_menu.add_command(label="Run Analysis", command=self._placeholder_command)
        analyze_menu.add_command(label="Show Results Report", command=self._placeholder_command)

        # --- Help Menu ---
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._placeholder_command)
        help_menu.add_command(label="About PyFrame2D", command=self._placeholder_command)


    def _create_main_layout(self):
        """Creates the main layout with input panel and canvas."""
        # --- Input Panel Frame (Left Side) ---
        self.input_panel = ttk.Frame(self, width=300, relief=tk.RIDGE, padding=5)
        self.input_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0), pady=5)
        self.input_panel.pack_propagate(False)

        # Notebook (Tabs) for different inputs
        self.notebook = ttk.Notebook(self.input_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self._create_nodes_tab()
        self._create_materials_tab()
        self._create_sections_tab()
        self._create_members_tab()
        self._create_supports_tab()
        self._create_loads_tab()

        # --- Canvas Frame (Right Side) ---
        self.canvas_frame = ttk.Frame(self, relief=tk.SUNKEN)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create the Canvas widget
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind the resize event
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Initial drawing (might be empty if model is empty)
        self._redraw_canvas()

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
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Nodes")

        # Grid layout for this frame
        frame.columnconfigure(1, weight=1) # Allow entry column to expand

        # Widgets
        ttk.Label(frame, text="Node ID:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.node_id_entry = ttk.Entry(frame, width=15)
        self.node_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="X Coord (m):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.node_x_entry = ttk.Entry(frame, width=15)
        self.node_x_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Y Coord (m):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.node_y_entry = ttk.Entry(frame, width=15)
        self.node_y_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

        add_button = ttk.Button(frame, text="Add Node", command=self._add_node)
        add_button.grid(row=3, column=0, columnspan=2, pady=10)

    def _create_materials_tab(self):
        """Creates the GUI elements for the Materials input tab."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Materials")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Material ID:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.mat_id_entry = ttk.Entry(frame, width=15)
        self.mat_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Name:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.mat_name_entry = ttk.Entry(frame, width=15)
        self.mat_name_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Young's Mod (E):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.mat_e_entry = ttk.Entry(frame, width=15)
        self.mat_e_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.mat_e_entry.insert(0, "e.g., 210 GPa or 2.1e11") # Placeholder text

        add_button = ttk.Button(frame, text="Add Material", command=self._add_material)
        add_button.grid(row=3, column=0, columnspan=2, pady=10)

    def _create_sections_tab(self):
        """Creates the GUI elements for the Sections input tab."""
        # This one is more complex due to multiple section types
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Sections")
        frame.columnconfigure(1, weight=1)

        # --- Section Type Selection ---
        ttk.Label(frame, text="Section Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.section_type_var = tk.StringVar(value="Rectangular") # Default value
        section_types = ["Rectangular", "Square", "I-Beam"] # Add more later
        type_menu = ttk.OptionMenu(frame, self.section_type_var, section_types[0], *section_types,
                                   command=self._update_section_fields)
        type_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # --- Common Fields ---
        ttk.Label(frame, text="Section ID:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.sec_id_entry = ttk.Entry(frame, width=15)
        self.sec_id_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Name:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.sec_name_entry = ttk.Entry(frame, width=15)
        self.sec_name_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

        # --- Frame for Type-Specific Fields ---
        self.section_fields_frame = ttk.Frame(frame)
        self.section_fields_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)
        self.section_fields_frame.columnconfigure(1, weight=1) # Allow entries to expand

        # --- Add Button ---
        add_button = ttk.Button(frame, text="Add Section", command=self._add_section)
        add_button.grid(row=4, column=0, columnspan=2, pady=10)

        # --- Initialize fields for the default type ---
        self._update_section_fields() # Call once to populate initially

    def _update_section_fields(self, *args):
        """Clears and updates the section input fields based on selected type."""
        # Destroy existing widgets in the specific frame
        for widget in self.section_fields_frame.winfo_children():
            widget.destroy()

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

    def _create_members_tab(self):
        """Creates the GUI elements for the Members input tab."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Members")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Member ID:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.mem_id_entry = ttk.Entry(frame, width=15)
        self.mem_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Start Node ID:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.mem_start_node_entry = ttk.Entry(frame, width=15)
        self.mem_start_node_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="End Node ID:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.mem_end_node_entry = ttk.Entry(frame, width=15)
        self.mem_end_node_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Material ID:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.mem_material_entry = ttk.Entry(frame, width=15)
        self.mem_material_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Section ID:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.mem_section_entry = ttk.Entry(frame, width=15)
        self.mem_section_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)

        add_button = ttk.Button(frame, text="Add Member", command=self._add_member)
        add_button.grid(row=5, column=0, columnspan=2, pady=10)


    def _create_supports_tab(self):
        """Creates the GUI elements for the Supports input tab."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Supports")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Node ID:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.sup_node_id_entry = ttk.Entry(frame, width=15)
        self.sup_node_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # OptionMenu for predefined types
        ttk.Label(frame, text="Type:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.sup_type_var = tk.StringVar(value="Pinned")
        sup_types = ["Pinned", "Fixed", "Roller X (Free X)", "Roller Y (Free Y)", "Custom"]
        type_menu = ttk.OptionMenu(frame, self.sup_type_var, sup_types[0], *sup_types,
                                  command=self._update_support_dofs)
        type_menu.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        # Checkbuttons for custom definition
        self.sup_dx_var = tk.BooleanVar(value=True)
        self.sup_dy_var = tk.BooleanVar(value=True)
        self.sup_rz_var = tk.BooleanVar(value=False)
        dx_check = ttk.Checkbutton(frame, text="Restrain DX", variable=self.sup_dx_var, state=tk.DISABLED)
        dx_check.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        dy_check = ttk.Checkbutton(frame, text="Restrain DY", variable=self.sup_dy_var, state=tk.DISABLED)
        dy_check.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        rz_check = ttk.Checkbutton(frame, text="Restrain RZ", variable=self.sup_rz_var, state=tk.DISABLED)
        rz_check.grid(row=4, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)

        # Store refs to checkbuttons to enable/disable them
        self.sup_dof_checks = [dx_check, dy_check, rz_check]

        add_button = ttk.Button(frame, text="Add/Update Support", command=self._add_support)
        add_button.grid(row=5, column=0, columnspan=2, pady=10)

        self._update_support_dofs() # Set initial state

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
        # Similar complexity to Sections - use type selection
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Loads")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Load Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.load_type_var = tk.StringVar(value="Nodal")
        load_types = ["Nodal", "Member Point", "Member UDL"]
        load_menu = ttk.OptionMenu(frame, self.load_type_var, load_types[0], *load_types,
                                   command=self._update_load_fields)
        load_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Load ID:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.load_id_entry = ttk.Entry(frame, width=15)
        self.load_id_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Label (Optional):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.load_label_entry = ttk.Entry(frame, width=15)
        self.load_label_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

        # Frame for type-specific fields
        self.load_fields_frame = ttk.Frame(frame)
        self.load_fields_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)
        self.load_fields_frame.columnconfigure(1, weight=1)

        add_button = ttk.Button(frame, text="Add Load", command=self._add_load)
        add_button.grid(row=4, column=0, columnspan=2, pady=10)

        self._update_load_fields() # Populate initial fields

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

    def _add_node(self):
        """Reads node data, validates, creates Node object, adds to model."""
        try:
            node_id = int(self.node_id_entry.get())
            x = float(self.node_x_entry.get())
            y = float(self.node_y_entry.get())

            node = Node(id=node_id, x=x, y=y)
            self.model.add_node(node)

            self.set_status(f"Node {node_id} added successfully.")
            # Clear entries
            self.node_id_entry.delete(0, tk.END)
            self.node_x_entry.delete(0, tk.END)
            self.node_y_entry.delete(0, tk.END)
            self.node_id_entry.focus() # Move cursor back to first field
            print(f"Model Nodes: {self.model.nodes}") # Debug print
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for Node: {e}")
            self.set_status(f"Error adding node: {e}")
        except TypeError as e:
             messagebox.showerror("Input Error", f"Invalid input type for Node: {e}")
             self.set_status(f"Error adding node: {e}")
        except Exception as e: # Catch other errors like duplicate ID from model.add_node
             messagebox.showerror("Error", f"Could not add Node: {e}")
             self.set_status(f"Error adding node: {e}")


    def _add_material(self):
        """Reads material data, validates, creates Material object, adds to model."""
        try:
            mat_id = int(self.mat_id_entry.get())
            name = self.mat_name_entry.get()
            e_str = self.mat_e_entry.get() # Get E as string

            # Try parsing E using units module if available
            youngs_modulus_value = e_str # Pass string directly to Material constructor

            mat = Material(id=mat_id, name=name, youngs_modulus=youngs_modulus_value)
            self.model.add_material(mat)

            self.set_status(f"Material {mat_id} ({name}) added successfully.")
            # Clear entries
            self.mat_id_entry.delete(0, tk.END)
            self.mat_name_entry.delete(0, tk.END)
            self.mat_e_entry.delete(0, tk.END)
            self.mat_id_entry.focus()
            print(f"Model Materials: {self.model.materials}") # Debug print
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for Material: {e}")
            self.set_status(f"Error adding material: {e}")
        except TypeError as e:
             messagebox.showerror("Input Error", f"Invalid input type for Material: {e}")
             self.set_status(f"Error adding material: {e}")
        except Exception as e:
             messagebox.showerror("Error", f"Could not add Material: {e}")
             self.set_status(f"Error adding material: {e}")

    def _add_section(self):
        """Reads section data, validates, creates Section object, adds to model."""
        try:
            sec_id = int(self.sec_id_entry.get())
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
                self.set_status(f"Section {sec_id} ({name}) added successfully.")
                # Clear common entries
                self.sec_id_entry.delete(0, tk.END)
                self.sec_name_entry.delete(0, tk.END)
                # Clear specific entries by updating fields (easier than tracking)
                self._update_section_fields()
                self.sec_id_entry.focus()
                print(f"Model Sections: {self.model.sections}") # Debug print

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
             messagebox.showerror("Error", f"Could not add Section: {e}")
             self.set_status(f"Error adding section: {e}")

    def _add_member(self):
        """Reads member data, validates, creates Member object, adds to model."""
        try:
            mem_id = int(self.mem_id_entry.get())
            start_node_id = int(self.mem_start_node_entry.get())
            end_node_id = int(self.mem_end_node_entry.get())
            material_id = int(self.mem_material_entry.get())
            section_id = int(self.mem_section_entry.get())

            # Get referenced objects from model
            start_node = self.model.get_node(start_node_id) # Raises KeyError if not found
            end_node = self.model.get_node(end_node_id)
            material = self.model.get_material(material_id)
            section = self.model.get_section(section_id)

            member = Member(mem_id, start_node, end_node, material, section)
            self.model.add_member(member)

            self.set_status(f"Member {mem_id} added successfully.")
            # Clear entries
            self.mem_id_entry.delete(0, tk.END); self.mem_start_node_entry.delete(0, tk.END);
            self.mem_end_node_entry.delete(0, tk.END); self.mem_material_entry.delete(0, tk.END);
            self.mem_section_entry.delete(0, tk.END)
            self.mem_id_entry.focus()
            print(f"Model Members: {self.model.members}") # Debug print

        except (ValueError, TypeError) as e: # Catches int/float conversion errors too
            messagebox.showerror("Input Error", f"Invalid input for Member: {e}")
            self.set_status(f"Error adding member: {e}")
        except KeyError as e: # Catch errors from get_node/mat/sec
             messagebox.showerror("Input Error", f"Invalid reference ID for Member: {e}")
             self.set_status(f"Error adding member: {e}")
        except Exception as e:
             messagebox.showerror("Error", f"Could not add Member: {e}")
             self.set_status(f"Error adding member: {e}")

    def _add_support(self):
        """Reads support data, validates, creates Support object, adds to model."""
        try:
            node_id = int(self.sup_node_id_entry.get())
            dx = self.sup_dx_var.get()
            dy = self.sup_dy_var.get()
            rz = self.sup_rz_var.get()

            support = Support(node_id, dx, dy, rz)
            self.model.add_support(support) # add_support handles duplicates

            self.set_status(f"Support added/updated for Node {node_id}.")
            # Clear entry
            self.sup_node_id_entry.delete(0, tk.END)
            self.sup_node_id_entry.focus()
            print(f"Model Supports: {self.model.supports}") # Debug print

        except (ValueError, TypeError) as e:
            messagebox.showerror("Input Error", f"Invalid input for Support: {e}")
            self.set_status(f"Error adding support: {e}")
        except Exception as e:
             messagebox.showerror("Error", f"Could not add Support: {e}")
             self.set_status(f"Error adding support: {e}")

    def _add_load(self):
        """Reads load data, validates, creates Load object, adds to model."""
        try:
            load_id = int(self.load_id_entry.get())
            label = self.load_label_entry.get()
            load_type = self.load_type_var.get()

            load = None
            if load_type == "Nodal":
                node_id = int(self.load_nodal_node_entry.get())
                fx = float(self.load_nodal_fx_entry.get())
                fy = float(self.load_nodal_fy_entry.get())
                mz = float(self.load_nodal_mz_entry.get())
                load = NodalLoad(load_id, node_id, fx, fy, mz, label)
            elif load_type == "Member Point":
                mem_id = int(self.load_pt_mem_entry.get())
                px = float(self.load_pt_px_entry.get())
                py = float(self.load_pt_py_entry.get())
                pos = float(self.load_pt_pos_entry.get())
                load = MemberPointLoad(load_id, mem_id, px, py, pos, label)
            elif load_type == "Member UDL":
                mem_id = int(self.load_udl_mem_entry.get())
                wx = float(self.load_udl_wx_entry.get())
                wy = float(self.load_udl_wy_entry.get())
                load = MemberUDLoad(load_id, mem_id, wx, wy, label)
            else:
                 raise ValueError(f"Selected load type '{load_type}' not handled.")

            if load:
                 self.model.add_load(load)
                 self.set_status(f"Load {load_id} added successfully.")
                 # Clear common fields
                 self.load_id_entry.delete(0, tk.END)
                 self.load_label_entry.delete(0, tk.END)
                 # Clear specific fields
                 self._update_load_fields()
                 self.load_id_entry.focus()
                 print(f"Model Loads: {self.model.loads}") # Debug print

        except (ValueError, TypeError) as e:
            messagebox.showerror("Input Error", f"Invalid input for Load: {e}")
            self.set_status(f"Error adding load: {e}")
        except AttributeError as e:
             messagebox.showerror("Input Error", f"Missing input field for selected load type: {e}")
             self.set_status(f"Error adding load: Incomplete fields.")
        except Exception as e:
             messagebox.showerror("Error", f"Could not add Load: {e}")
             self.set_status(f"Error adding load: {e}")


    # --- Utility Methods ---

    def _new_project(self):
        """Clears the current model."""
        if messagebox.askokcancel("New Project", "Clear current model and start new project?"):
             self.model = StructuralModel() # Create new empty model
             # Optionally clear listboxes or other display elements later
             self.set_status("New project started.")
             print("Model Cleared.")


    def set_status(self, text: str):
        # ... (Method remains the same) ...
        """Updates the text displayed in the status bar."""
        self.status_bar.config(text=text)

    def _placeholder_command(self):
        # ... (Method remains the same) ...
        """Placeholder for menu commands not yet implemented."""
        print("Menu command executed (Not Implemented Yet)")
        self.set_status("Functionality not yet implemented.")
        messagebox.showinfo("Not Implemented", "This feature is not yet available.")

    def _on_exit(self):
        # ... (Method remains the same) ...
        """Handles the window closing event."""
        if messagebox.askokcancel("Quit", "Do you really want to quit PyFrame2D?"):
            self.destroy() # Close the Tkinter window


# --- Allow running this file directly for testing ---
if __name__ == '__main__':
    if not CORE_AVAILABLE:
        print("ERROR: Cannot run GUI without core model components.")
    else:
        app = MainApplicationWindow()
        app.mainloop()