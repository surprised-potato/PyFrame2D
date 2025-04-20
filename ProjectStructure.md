PyFrame2D/
├── core/             # Core analysis engine, model definitions, element calculations
│   ├── __init__.py
│   ├── model.py
│   ├── analysis.py
│   └── elements.py
├── gui/              # Graphical User Interface (Tkinter)
│   ├── __init__.py
│   └── main_window.py
├── project_io/       # Input/Output operations
│   ├── __init__.py
│   ├── project_files.py
│   └── reporting.py
├── tests/            # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_analysis.py
│   ├── test_basics.py
│   ├── test_elements.py
│   ├── test_io.py
│   ├── test_model.py
│   ├── test_reporting.py
│   └── test_units.py
├── units/            # Unit definitions and parsing
│   ├── __init__.py
│   └── units.py
├── .venv/            # Virtual environment files
├── .gitignore
├── main.py           # Main application entry point
├── requirements.txt  # Dependency list (alternative/complement to pyproject.toml)
├── pyproject.toml    # Project build/metadata definition
└── README.md         # This file