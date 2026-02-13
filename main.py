import tkinter as tk
from gui import DetectorApp

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorApp(root)
    
    # Handle the 'X' button on the window properly
    root.protocol("WM_DELETE_WINDOW", app.stop_demo)
    
    root.mainloop()