# tests/conftest.py
import sys
import os
from pathlib import Path

print("\n--- conftest.py START ---") # Mark start

# Get the absolute path to the project root directory (PyFrame2D)
project_root = Path(__file__).resolve().parent.parent

# Check if the path is already in sys.path to avoid duplicates
if str(project_root) not in sys.path:
    # Insert at the beginning to give it priority
    sys.path.insert(0, str(project_root))
    print(f"INFO [conftest.py]: Added '{project_root}' to sys.path")
else:
    print(f"INFO [conftest.py]: Path '{project_root}' already in sys.path")

# Print sys.path contents for verification
print("DEBUG [conftest.py]: Current sys.path contents:")
for i, p in enumerate(sys.path):
    print(f"  [{i}] {p}")

# Try importing directly here to see if it works at this stage
try:
    import io as built_in_io # Check built-in
    print(f"INFO [conftest.py]: Successfully imported built-in 'io': {built_in_io}")
    # Now try importing *our* io package
    import io as project_io_pkg # Try importing the directory as a package
    print(f"INFO [conftest.py]: Successfully imported project 'io' package: {project_io_pkg}")
    # And the specific module
    from io import project_files
    print(f"INFO [conftest.py]: Successfully imported 'io.project_files': {project_files}")
except ImportError as e:
    print(f"ERROR [conftest.py]: Import failed during conftest execution: {e}")
except Exception as e:
    print(f"ERROR [conftest.py]: Unexpected error during conftest import test: {e}")


print("--- conftest.py END ---") # Mark end