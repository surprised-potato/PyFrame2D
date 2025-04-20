# main.py

import sys
import tkinter as tk # Need tkinter root if main_window doesn't inherit from tk.Tk
# Import the main application window class
try:
    from gui.main_window import MainApplicationWindow
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import GUI components: {e}")
    print("Please ensure GUI dependencies (like Tkinter) are available and")
    print("the project structure is correct.")
    GUI_AVAILABLE = False
    # Define a dummy class if needed, although sys.exit is better
    class MainApplicationWindow: # Dummy
        def mainloop(self): pass

if __name__ == "__main__":
    print("Starting PyFrame2D Application...")

    if not GUI_AVAILABLE:
        print("Exiting due to missing GUI components.")
        sys.exit(1) # Exit with an error code

    # Create and run the main application window
    app = MainApplicationWindow()
    app.mainloop() # Start the Tkinter event loop

    print("PyFrame2D Application Closed.")