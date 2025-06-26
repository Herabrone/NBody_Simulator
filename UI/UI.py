import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add the parent directory to sys.path so that Logic can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import run_simulation from the Logic module
from Logic.Simulation import run_simulation
from Logic.Simulation import run_simulation_gpu

# Default values
DEFAULT_N = "50"
DEFAULT_R0 = "3.0856776e10"
DEFAULT_TIMESPAN = "1e6"
DEFAULT_THETA = "0.5"
DEFAULT_BOUND = "10"

def run_gui_simulation():
    try:
        n = int(entry_n.get())
        R0 = float(entry_R0.get())
        timespan = float(entry_timespan.get())
        theta = float(entry_theta.get())
        bound_condition = int(entry_bound_condition.get())

        # Run the simulation
        # messagebox.showinfo("Success", "Simulation has started!")
        # run_simulation(n, R0, timespan, theta, bound_condition)


        # will run the GPU simulation
        messagebox.showinfo("Success", "Simulation has started!")
        run_simulation_gpu(n, R0, timespan, theta, 1e3, bound_condition)

    except ValueError as e:
        messagebox.showerror("Error", "Please enter valid values.")

def set_default_settings():
    entry_n.delete(0, tk.END)
    entry_n.insert(0, DEFAULT_N)
    entry_R0.delete(0, tk.END)
    entry_R0.insert(0, DEFAULT_R0)
    entry_timespan.delete(0, tk.END)
    entry_timespan.insert(0, DEFAULT_TIMESPAN)
    entry_theta.delete(0, tk.END)
    entry_theta.insert(0, DEFAULT_THETA)
    entry_bound_condition.delete(0, tk.END)
    entry_bound_condition.insert(0, DEFAULT_BOUND)

# Create the main window
root = tk.Tk()
root.title("N-Body Simulation GUI")
root.geometry("1280x720")  # Set window size to 1280x720

# Create and place labels and input fields for simulation parameters with default values
tk.Label(root, text="Number of Bodies (n)").grid(row=0, column=0, padx=10, pady=10, sticky='e')
entry_n = tk.Entry(root)
entry_n.grid(row=0, column=1, padx=10, pady=10)
entry_n.insert(0, DEFAULT_N)

tk.Label(root, text="Initial Radius (R0)").grid(row=1, column=0, padx=10, pady=10, sticky='e')
entry_R0 = tk.Entry(root)
entry_R0.grid(row=1, column=1, padx=10, pady=10)
entry_R0.insert(0, DEFAULT_R0)

tk.Label(root, text="Timespan (seconds)").grid(row=2, column=0, padx=10, pady=10, sticky='e')
entry_timespan = tk.Entry(root)
entry_timespan.grid(row=2, column=1, padx=10, pady=10)
entry_timespan.insert(0, DEFAULT_TIMESPAN)

tk.Label(root, text="Opening Angle (theta)").grid(row=3, column=0, padx=10, pady=10, sticky='e')
entry_theta = tk.Entry(root)
entry_theta.grid(row=3, column=1, padx=10, pady=10)
entry_theta.insert(0, DEFAULT_THETA)

tk.Label(root, text="Bound Condition").grid(row=4, column=0, padx=10, pady=10, sticky='e')
entry_bound_condition = tk.Entry(root)
entry_bound_condition.grid(row=4, column=1, padx=10, pady=10)
entry_bound_condition.insert(0, DEFAULT_BOUND)

# Button to run the simulation
btn_run = tk.Button(root, text="Run Simulation", command=run_gui_simulation)
btn_run.grid(row=5, column=0, columnspan=2, pady=20)

# Button to reset to default settings
btn_defaults = tk.Button(root, text="Use Default Settings", command=set_default_settings)
btn_defaults.grid(row=6, column=0, columnspan=2, pady=10)

# Start the Tkinter event loop
root.mainloop()
