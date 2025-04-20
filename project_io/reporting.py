# project_io/reporting.py

import math
import os
import numpy as np
from typing import TYPE_CHECKING, Optional

# Use TYPE_CHECKING block for imports needed only for type hints
if TYPE_CHECKING:
    from ..core.model import StructuralModel
    from ..core.analysis import AnalysisResults

# --- Text Report Generation ---

def generate_text_report(results: 'AnalysisResults', model: Optional['StructuralModel'] = None) -> str:
    """
    Generates a formatted text report summarizing the analysis results.

    Args:
        results: The AnalysisResults object containing the analysis output.
        model: (Optional) The original StructuralModel object, used for context
               like component counts in the header.

    Returns:
        A string containing the formatted text report.
    """
    report_lines = []
    separator = "=" * 70
    sub_separator = "-" * 70

    # --- Header ---
    report_lines.append(separator)
    report_lines.append(f"Analysis Report for Model: {results.model_name}")
    report_lines.append(separator)
    report_lines.append(f"Status: {results.status}")
    if results.message:
        report_lines.append(f"Message: {results.message}")

    if model:
        report_lines.append(sub_separator)
        report_lines.append("Model Summary:")
        report_lines.append(f"  Nodes: {len(model.nodes)}")
        report_lines.append(f"  Materials: {len(model.materials)}")
        report_lines.append(f"  Sections: {len(model.sections)}")
        report_lines.append(f"  Members: {len(model.members)}")
        report_lines.append(f"  Supports: {len(model.supports)}")
        report_lines.append(f"  Loads: {len(model.loads)}")

    if results.status != "Success":
        report_lines.append(separator)
        report_lines.append("Analysis did not complete successfully. Results below may be incomplete or unavailable.")
        report_lines.append(separator)
        report_lines.append("NODAL DISPLACEMENTS")
        report_lines.append("(Nodal displacements not available)")
        report_lines.append(separator)
        report_lines.append("SUPPORT REACTIONS")
        report_lines.append("(Support reactions not available)")
        report_lines.append(separator)
        report_lines.append("MEMBER END FORCES (Local Coordinates)")
        report_lines.append("(Member end forces not available)")
        report_lines.append(separator)
        report_lines.append("DIAGRAM DATA")
        report_lines.append("(Diagram data not calculated or available)")
        report_lines.append(separator)
        report_lines.append("End of Report")
        report_lines.append(separator)
        return "\n".join(report_lines) # Return after adding messages

    # --- Nodal Displacements ---
    report_lines.append(separator)
    report_lines.append("NODAL DISPLACEMENTS")
    report_lines.append(sub_separator) # Print separator unconditionally after header
    if results.nodal_displacements:
        # Print Header Row
        report_lines.append(f"{'Node ID':<10} {'DX (m)':<15} {'DY (m)':<15} {'RZ (rad)':<15}")
        report_lines.append(sub_separator) # Print separator after header row
        # Print Data Rows
        for node_id in sorted(results.nodal_displacements.keys()):
            disp = results.nodal_displacements[node_id]
            dx, dy, rz = disp
            report_lines.append(f"{node_id:<10} {dx:<15.6e} {dy:<15.6e} {rz:<15.6e}")
    else:
        report_lines.append("(Nodal displacements not available)") # If no data in results

    # --- Support Reactions ---
    report_lines.append(separator)
    report_lines.append("SUPPORT REACTIONS")
    report_lines.append(sub_separator) # Print separator unconditionally
    if results.support_reactions:
        has_printed_header = False
        for node_id in sorted(results.support_reactions.keys()):
            reac = results.support_reactions[node_id]
            supported = model and model.get_support(node_id) is not None
            # Determine if this reaction should be printed
            # Print if EITHER the node is explicitly supported in the model OR
            # if there's a non-negligible calculated reaction (covers cases where model isn't passed)
            should_print_reaction = supported or not all(abs(val) < 1e-9 for val in reac)

            if should_print_reaction: # Use the calculated boolean
                 if not has_printed_header: # Print header only once needed
                    report_lines.append(f"{'Node ID':<10} {'RX (N)':<15} {'RY (N)':<15} {'MZ (Nm)':<15}")
                    report_lines.append(sub_separator)
                    has_printed_header = True
                 rx, ry, mz = reac # Unpack reactions here
                 report_lines.append(f"{node_id:<10} {rx:<15.6e} {ry:<15.6e} {mz:<15.6e}")

        if not has_printed_header: # If no reactions were printed
             report_lines.append("(No non-zero reactions to display or model context unavailable)")
    else:
        report_lines.append("(Support reactions not available)")

    # --- Member End Forces (Local Coordinates) ---
    report_lines.append(separator)
    report_lines.append("MEMBER END FORCES (Local Coordinates)")
    report_lines.append(sub_separator) # Print separator unconditionally
    if results.member_end_forces:
        # Print Header Row
        report_lines.append(f"{'Member ID':<12} {'Pxi (N)':<12} {'Pyi (N)':<12} {'Mzi (Nm)':<12} {'Pxj (N)':<12} {'Pyj (N)':<12} {'Mzj (Nm)':<12}")
        report_lines.append(sub_separator)
        # Print Data Rows
        for mem_id in sorted(results.member_end_forces.keys()):
             forces = results.member_end_forces[mem_id].flatten() # Use flattened array
             if len(forces) == 6:
                 pxi, pyi, mzi, pxj, pyj, mzj = forces
                 formatted_force_line = f"{mem_id:<12} {pxi:<12.4e} {pyi:<12.4e} {mzi:<12.4e} {pxj:<12.4e} {pyj:<12.4e} {mzj:<12.4e}"
             else:
                  formatted_force_line = f"{mem_id:<12} {'--Invalid Force Vector Shape--'}"
             report_lines.append(formatted_force_line) # Append the created string
    else:
         report_lines.append("(Member end forces not available)")

    # --- Diagram Data Summary ---
    report_lines.append(separator)
    report_lines.append("DIAGRAM DATA")
    report_lines.append(sub_separator) # Print separator unconditionally
    has_diagrams = results.member_afd_data or results.member_sfd_data or results.member_bmd_data
    if has_diagrams:
         num_afd = len(results.member_afd_data) if results.member_afd_data else 0
         num_sfd = len(results.member_sfd_data) if results.member_sfd_data else 0
         num_bmd = len(results.member_bmd_data) if results.member_bmd_data else 0
         report_lines.append(f"Diagram data calculated for {num_afd} (AFD), {num_sfd} (SFD), {num_bmd} (BMD) members.")
    else:
         report_lines.append("(Diagram data not calculated or available)")

    report_lines.append(separator)
    report_lines.append("End of Report")
    report_lines.append(separator)

    return "\n".join(report_lines)



def save_report_to_file(report_string: str, file_path: str):
    """
    Saves a generated report string to a text file.

    Args:
        report_string: The string content of the report.
        file_path: The full path to the output text file (e.g., "results/report.txt").

    Raises:
        IOError: If there's an error writing the file.
    """
    print(f"Attempting to save report to {file_path}...")
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir: # Avoid error if saving to current directory
            os.makedirs(output_dir, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_string)
        print("Report saved successfully.")
    except (IOError, OSError) as e:
        raise IOError(f"Error writing report file '{file_path}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred saving report: {e}") from e


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Need to import analysis and model classes to run example
    from ..core.model import StructuralModel, Node, Material, RectangularProfile, Member, Support, NodalLoad, MemberUDLoad
    from ..core.analysis import analyze, AnalysisResults

    print("--- Generating Report for Portal Frame Example ---")
    model = StructuralModel(name="Portal Frame")
    # ... (Recreate the portal frame model as in previous examples) ...
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

    # Run analysis
    results = analyze(model, num_diagram_points=5)

    # Generate report string
    report_str = generate_text_report(results, model)

    # Print to console
    print("\n--- Generated Report ---")
    print(report_str)

    # Save to file
    output_file = "portal_frame_report.txt"
    try:
        save_report_to_file(report_str, output_file)
    except Exception as e:
        print(f"\nError saving report to file: {e}")