"""
GUI module for dataset selection.
"""
import tkinter as tk
from tkinter import ttk


def select_dataset():
    """
    Show a popup dialog to select which dataset to use.
    Returns the selected dataset name or None if cancelled.
    """
    root = tk.Tk()
    root.title("Select Dataset")
    root.geometry("400x150")
    root.resizable(False, False)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    selected_dataset = [None]  # Use list to allow modification in nested function
    
    # Label
    label = ttk.Label(root, text="Please select a dataset (dimensions in parentheses):", font=("Arial", 10))
    label.pack(pady=10)
    
    # Dropdown
    dataset_var = tk.StringVar(value="Iris (4D)")
    datasets = ["Iris (4D)", "MNIST (784D)", "Wisconsin Breast Cancer (30D)", "Wisconsin Breast Cancer (9D)"]
    dropdown = ttk.Combobox(root, textvariable=dataset_var, values=datasets, state="readonly", width=40)
    dropdown.pack(pady=10)
    
    # Buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=10)
    
    def on_ok():
        selected_dataset[0] = dataset_var.get()
        root.destroy()
    
    def on_cancel():
        selected_dataset[0] = None
        root.destroy()
    
    ok_button = ttk.Button(button_frame, text="OK", command=on_ok, width=10)
    ok_button.pack(side=tk.LEFT, padx=5)
    
    cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel, width=10)
    cancel_button.pack(side=tk.LEFT, padx=5)
    
    # Make Enter key trigger OK
    root.bind('<Return>', lambda e: on_ok())
    root.bind('<Escape>', lambda e: on_cancel())
    
    # Focus on dropdown
    dropdown.focus()
    
    root.mainloop()
    
    return selected_dataset[0]

