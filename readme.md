# PyFrame2D - Modular 2D Frame Analysis Program

## Overview

PyFrame2D is a desktop application built with Python for performing structural analysis of 2D frame structures (beams and columns) using the Direct Stiffness Method. It features a modular design separating the core analysis engine, data model, user interface, and utility functions.

Users can define structural geometry (nodes, members), materials (Young's Modulus with unit support), standard section profiles (Rectangle, Square, I-Beam), boundary conditions (fixed, pinned, roller supports), and various load types (nodal point/moment, member point/UDL). The application calculates nodal displacements, support reactions, member end forces, and generates data for internal force diagrams (Axial, Shear, Bending Moment). Results are presented numerically and visually, including a plot of the deflected shape.

## Features

*   **Graphical User Interface (GUI):** Built with Tkinter for cross-platform compatibility.
    *   Tabbed interface for defining Nodes, Materials, Sections, Members, Supports, and Loads.
    *   Listboxes for viewing and managing (Edit/Delete) added components.
    *   Canvas area for visualizing the model geometry, supports, loads, and deflected shape.
    *   Menu bar for File operations (New, Open, Save, Save As, Export Report), Analysis control, and View options.
    *   Status bar for user feedback.
    *   Toggleable display options for Node IDs, Member IDs, Supports, Loads, Deflected Shape.
*   **Core Model:**
    *   Object-oriented representation of structural components.
    *   Handles various section types (Rectangular, Square, I-Beam).
    *   Supports standard boundary conditions (Fixed, Pinned, Roller X/Y, Custom).
    *   Supports Nodal (Fx, Fy, Mz), Member Point (Px, Py), and Member UDL (wx, wy) loads.
    *   Includes model validation checks.
*   **Analysis Engine:**
    *   Uses the Direct Stiffness Method for 2D frame elements.
    *   Calculates local and global stiffness matrices.
    *   Handles coordinate transformations.
    *   Calculates Fixed-End Forces for member loads.
    *   Assembles and solves the global system of equations [K]{U} = {F}.
    *   Performs post-processing for reactions and member end forces.
    *   Generates data points for AFD, SFD, and BMD.
*   **Utilities:**
    *   Unit parsing module allows input with prefixes (e.g., "210 GPa", "5 kN", "500 mm").
*   **Persistence & Reporting:**
    *   Save/Load complete structural models to/from JSON files.
    *   Generate formatted text reports summarizing analysis results.
    *   Save text reports to files.
*   **Testing:** Comprehensive unit tests using `pytest` cover core model, analysis, element calculations, units, and IO functions.

## Installation & Setup

1.  **Prerequisites:**
    *   Python >= 3.8 (developed with 3.13). Ensure Python is installed and correctly configured in your system's PATH. (Avoid Microsoft Store version if possible).
    *   `git` (for cloning).
    *   `pip` (Python package installer).

2.  **Clone Repository:**
    ```bash
    git clone <your-repository-url> PyFrame2D
    cd PyFrame2D
    ```

3.  **Create & Activate Virtual Environment:**
    ```bash
    # Create venv (use python/py -3.x matching your desired base version)
    python -m venv .venv

    # Activate venv
    # Windows PowerShell:
    .\.venv\Scripts\Activate.ps1
    # Windows Command Prompt:
    .\.venv\Scripts\activate.bat
    # Linux/macOS Bash:
    source .venv/bin/activate
    ```
    *(Look for the `(.venv)` prefix in your prompt).*

4.  **Install Dependencies & Project:**
    *(Ensure your virtual environment is active)*
    ```bash
    # Install dependencies and the project packages (core, project_io, etc.) in editable mode
    pip install -e .
    # Ensure pytest is installed for testing
    pip install pytest
    ```
    *(This uses the `pyproject.toml` file for setup).*

## Usage

1.  **Activate Virtual Environment:**
    ```powershell
    # Example for PowerShell
    .\.venv\Scripts\Activate.ps1
    ```
2.  **Run Application:** From the `PyFrame2D` root directory:
    ```bash
    python main.py
    ```
3.  **GUI Workflow:**
    *   Define model components using the tabs on the left panel. Enter data and click "Add [...]". IDs are usually auto-generated.
    *   View added components in the listboxes below the input fields.
    *   Use "Edit Selected" to populate fields with an existing component's data, modify, and click "Update [...]".
    *   Use "Delete Selected" to remove components (with confirmation).
    *   Observe the model visualization on the canvas (right panel). Use the "View" menu to toggle labels/elements.
    *   Use the "File" menu to manage projects (New, Save As..., Open...). Models are saved as `.json` files.
    *   Click the "Run Analysis" button or use Analyze -> Run Analysis.
    *   View numerical results in the tabs at the bottom right (Displacements, Reactions, Member Forces).
    *   View the deflected shape overlay (toggle via "View" menu).
    *   Use File -> Export Text Report... to save a summary of the results.

## Running Tests

Ensure `pytest` is installed in your virtual environment.

From the `PyFrame2D` root directory (with the venv active):

```bash
# Run all tests verbose, using the active Python environment
python -m pytest -v