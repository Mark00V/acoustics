import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create the main window
root = tk.Tk()
root.title("Matplotlib in Tkinter Example")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Create a tab for the Matplotlib plot
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Plot")

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(5, 4))

# Plot some data
x = [1, 2, 3, 4, 5]
y = [10, 5, 8, 6, 2]
ax.plot(x, y)

# Create a FigureCanvasTkAgg widget to display the Matplotlib plot
canvas = FigureCanvasTkAgg(fig, master=tab1)
canvas.get_tk_widget().pack()

# Add a button to update the plot (optional)
update_button = tk.Button(tab1, text="Update Plot")
update_button.pack()

# You can define a function to update the plot when the button is clicked
def update_plot():
    # Update the plot with new data
    new_y = [6, 8, 4, 3, 7]
    ax.clear()
    ax.plot(x, new_y)
    canvas.draw()

update_button.config(command=update_plot)

# Start the Tkinter main loop
root.mainloop()
