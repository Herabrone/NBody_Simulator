import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add the parent directory to sys.path so that Logic can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the run_simulation function from the Logic module
from Logic.Simulation import run_simulation

# Function to execute the simulation based on user inputs
def run_gui_simulation():
    try:
        n = int(entry_n.get())
        R0 = float(entry_R0.get())
        timespan = float(entry_timespan.get())
        theta = float(entry_theta.get())
        bound_condition = int(entry_bound_condition.get())

        # Run the simulation
        run_simulation(n, R0, timespan, theta, bound_condition)
        messagebox.showinfo("Success", "Simulation has started!")

    except ValueError as e:
        messagebox.showerror("Error", "Please enter valid values.")

# Create the main window
root = tk.Tk()
root.title("N-Body Simulation GUI")

# Create and place labels and input fields for simulation parameters
tk.Label(root, text="Number of Bodies (n)").grid(row=0, column=0, padx=10, pady=10)
entry_n = tk.Entry(root)
entry_n.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Initial Radius (R0)").grid(row=1, column=0, padx=10, pady=10)
entry_R0 = tk.Entry(root)
entry_R0.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Timespan (seconds)").grid(row=2, column=0, padx=10, pady=10)
entry_timespan = tk.Entry(root)
entry_timespan.grid(row=2, column=1, padx=10, pady=10)

tk.Label(root, text="Opening Angle (theta)").grid(row=3, column=0, padx=10, pady=10)
entry_theta = tk.Entry(root)
entry_theta.grid(row=3, column=1, padx=10, pady=10)

tk.Label(root, text="Bound Condition").grid(row=4, column=0, padx=10, pady=10)
entry_bound_condition = tk.Entry(root)
entry_bound_condition.grid(row=4, column=1, padx=10, pady=10)

# Button to run the simulation
btn_run = tk.Button(root, text="Run Simulation", command=run_gui_simulation)
btn_run.grid(row=5, column=0, columnspan=2, pady=20)

# Start the Tkinter event loop
root.mainloop()
