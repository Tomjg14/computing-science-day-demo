import tkinter as tk
import customtkinter as ctk
from gui import DetectorApp

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    app = DetectorApp(root)
    
    # Handle the 'X' button on the window properly
    root.protocol("WM_DELETE_WINDOW", app.stop_demo)
    
    root.mainloop()