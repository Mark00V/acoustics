# Python imports
from typing import Tuple, List
import numpy as np
import tkinter as tk
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Custom imports
from calcfem import CalcFEM
from mesh import CreateMesh

#################################################
# tkinter static parameters
# window
WINDOW_HEIGHT = 800
WINDOWS_WIDTH = 1400
# canvas
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 600
DOTSIZE = 6  # click points in canvas
LINEWIDTH = 2
GRIDSPACE = 25  # space between grid
# Borderwidth (space between window frame and widgets and widgets between widgets, relative to windowsize)
BORDER_WIDTH = 0.01
#################################################


class FEMGUI:
    """
    Create Gui
    """

    def __init__(self):
        """
        constructor
        """
        self.polygon_coordinates = [[0, 600]]
        self.firstclick_canvas = True
        self.polygon_closed = False
        self.mesh = None
        self.calcfem = None
        self.click_counter = 0 # counts clicks in canvas
        self.boundaries_set = None

        # FEM Data
        self.all_points_numbered = None
        self.all_outline_vertices_numbered = None
        self.boundaries_numbered = None
        self.triangles = None
        self.boundary_values = dict()

    def finish_polygon_coordinates(self):

        self.polygon_closed = True
        self.polygon_coordinates.append([0, 600])

        # create closing line
        self.canvas.create_line(self.polygon_coordinates[-2][0], self.polygon_coordinates[-2][1],
                                self.polygon_coordinates[-1][0], self.polygon_coordinates[-1][1],
                                fill="black", width=LINEWIDTH)

        # create last boundary marker
        self.click_counter += 1
        boundary_text = f"B-{self.click_counter}"
        x1 = self.polygon_coordinates[-2][0]
        y1 = self.polygon_coordinates[-2][1]
        x2 = self.polygon_coordinates[-1][0]
        y2 = self.polygon_coordinates[-1][1]
        center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
        center_x, center_y = center_point
        self.canvas.create_oval(center_x - 15, center_y - 15, center_x + 15, center_y + 15, outline="black",
                                fill="white")
        self.canvas.create_text(center_x, center_y, text=boundary_text, fill="blue", font=("Helvetica", 11))
        #print(self.polygon_coordinates)

    def add_grid(self):
        for x in range(0, CANVAS_WIDTH + GRIDSPACE, GRIDSPACE):
            self.canvas.create_line(x, 0, x, CANVAS_HEIGHT, fill="black", width=1)
        for y in range(0, CANVAS_HEIGHT + GRIDSPACE, GRIDSPACE):
            self.canvas.create_line(0, y, CANVAS_WIDTH, y, fill="black", width=1)

    def find_grid(self, coord_x: int, coord_y: int):
        grid_x = range(0, CANVAS_WIDTH + GRIDSPACE, GRIDSPACE)
        grid_y = range(0, CANVAS_HEIGHT + GRIDSPACE, GRIDSPACE)

        div_x = math.floor(coord_x / GRIDSPACE)
        mod_x = coord_x % GRIDSPACE
        div_y = math.floor(coord_y / GRIDSPACE)
        mod_y = coord_y % GRIDSPACE
        if mod_x <= 12:
            new_x = grid_x[div_x]
        else:
            new_x = grid_x[div_x + 1]
        if mod_y <= 12:
            new_y = grid_y[div_y]
        else:
            new_y = grid_y[div_y + 1]
        return new_x, new_y

    def coord_transform(self):
        scale_factor = 0.01  # todo
        polygon_coordinates_transformed = [[scale_factor * elem[0], -1 * scale_factor * (elem[1] - 600)]
                                           for elem in self.polygon_coordinates]
        polygon_coordinates_transformed = [[elem[0], 0 if elem[1] == -0 else elem[1]]
                                           for elem in polygon_coordinates_transformed]
        self.polygon_coordinates_transformed = polygon_coordinates_transformed

        return polygon_coordinates_transformed

    def create_mesh(self):
        method = self.methods_dropdown_var.get()
        density_un = self.density_slider.get()
        density = 1 / density_un
        polygon_coordinates_transformed = self.coord_transform()
        polygon_vertices = np.array(polygon_coordinates_transformed)

        start_time = time.time()
        self.mesh = CreateMesh(polygon_vertices, density, method)
        end_time = time.time()
        elased_time = end_time - start_time

        self.all_points_numbered, self.all_outline_vertices_numbered, self.boundaries_numbered, self.triangles = self.mesh.create_mesh()
        fig, ax = self.mesh.show_mesh()

        # create new window
        stats_nbr_of_nodes = len(self.all_points_numbered)
        stats_nbr_of_elements = len(self.triangles)
        stats_nbrs_of_boundaries = len(self.boundaries_numbered)

        stats_string = f"Stats:\n" \
                       f"Number of Nodes: {stats_nbr_of_nodes}\n" \
                       f"Number of Elements: {stats_nbr_of_elements}\n" \
                       f"Number of Boundaries: {stats_nbrs_of_boundaries}\n" \
                       f"Calculation time: {elased_time:.4f}"

        polygon_transformed_values_x_max = max([point[0] for point in self.polygon_coordinates_transformed])
        polygon_transformed_values_y_min = min([point[1] for point in self.polygon_coordinates_transformed])

        top = tk.Toplevel(self.root)
        top.title("Generated Mesh")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack()
        text = ax.text(polygon_transformed_values_x_max * 0.75, polygon_transformed_values_y_min * 1.00, stats_string,
                       fontsize=8, ha='left')


    def clean_canvas(self, canvas):
        all_canvas_elements = canvas.find_all()
        for elem in all_canvas_elements:
            canvas.delete(elem)


    def create_output(self):
        ...

    def calc_fem_main(self):
        if not self.boundaries_set or not self.mesh: # TODO...autoclose of polygon oder so
            warning_window = tk.Toplevel(self.root)
            warning_window.title("Warning")
            warning_label = tk.Label(warning_window, text="Warning: Create Mesh / set boundary conditions first!", padx=20, pady=20, font=("Arial", 12), foreground='red')
            warning_label.pack()
            return ''

        self.calcfem = CalcFEM(self.all_points_numbered, self.all_outline_vertices_numbered,
                               self.boundaries_numbered, self.triangles, self.boundary_values)
        self.solution = self.calcfem.calc_fem()
        self.show_fem_solution_window()

    def show_fem_solution_window(self):

        fig, ax = self.calcfem.plot_solution()
        fem_stats_string = f"Boundary conditions:\n"

        for boundary_nbr, value in self.boundary_values.items():
            fem_stats_string += f"B-{boundary_nbr}: {value}\n"

        polygon_transformed_values_x_max = max([point[0] for point in self.polygon_coordinates_transformed])
        polygon_transformed_values_y_min = min([point[1] for point in self.polygon_coordinates_transformed])

        fem_solution_window = tk.Toplevel(self.root)
        fem_solution_window.title("FEM Solution")
        fem_solution_window.geometry(f"{1200}x{800}")
        canvas = FigureCanvasTkAgg(fig, master=fem_solution_window)
        canvas.get_tk_widget().pack()
        text = ax.text(polygon_transformed_values_x_max * 0.75, polygon_transformed_values_y_min * 1.00, fem_stats_string,
                       fontsize=10, ha='left')

    def on_canvas_click(self, event):
        # Get coordinates of right click
        x, y = event.x, event.y

        # lock coordinates to grid
        x, y = self.find_grid(x, y)
        self.polygon_coordinates_str.set(self.polygon_coordinates_tmp)
        self.polygon_coordinates.append([x, y])
        self.polygon_coordinates_tmp += f"[{x},{y}]  "

        # create lines between points
        if self.firstclick_canvas == True:
            self.firstclick_canvas = False
            self.canvas.create_line(0, CANVAS_HEIGHT, x, y, fill="black", width=LINEWIDTH)
        if 'prevpoint' in FEMGUI.on_canvas_click.__dict__ and not self.firstclick_canvas:
            self.canvas.create_line(FEMGUI.on_canvas_click.prevpoint[0], FEMGUI.on_canvas_click.prevpoint[1], x, y,
                                    fill="black", width=LINEWIDTH)

        # get centerpoint of line and create text
        if 'prevpoint' in FEMGUI.on_canvas_click.__dict__:
            self.click_counter += 1
            boundary_text = f"B-{self.click_counter}"
            x1 = FEMGUI.on_canvas_click.prevpoint[0]
            y1 = FEMGUI.on_canvas_click.prevpoint[1]
            x2 = x
            y2 = y
            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            center_x, center_y = center_point
            self.canvas.create_oval(center_x - 15, center_y - 15, center_x + 15, center_y + 15, outline="black", fill="white")
            self.canvas.create_text(center_x, center_y, text=boundary_text, fill="blue", font=("Helvetica", 11))
        else:
            boundary_text = f"B-{self.click_counter}"
            x1 = 0
            y1 = 600
            x2 = x
            y2 = y
            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            center_x, center_y = center_point
            self.canvas.create_oval(center_x - 15, center_y - 15, center_x + 15, center_y + 15, outline="black",
                                    fill="white")
            self.canvas.create_text(center_x, center_y, text=boundary_text, fill="blue", font=("Helvetica", 11))

        FEMGUI.on_canvas_click.prevpoint = (x, y)  # TODO: Why does this not work with self????

        # create point at click
        self.canvas.create_oval(x - DOTSIZE, y - DOTSIZE, x + DOTSIZE, y + DOTSIZE, outline="black", fill="#851d1f")

        # Clear content
        self.coord_var_x.set('0')
        self.coord_var_y.set('600')

        # Display coordinates
        self.coord_var_x.set(f"{x}")
        self.coord_var_y.set(f"{y}")


    def set_boundaries_window(self):

        if not self.polygon_closed: # TODO...autoclose of polygon oder so
            warning_window = tk.Toplevel(self.root)
            warning_window.title("Warning")
            warning_label = tk.Label(warning_window, text="Warning: Close Polygon first!",
                                     padx=20, pady=20, font=("Arial", 12), foreground='red')
            warning_label.pack()
            return ''

        def set_value():
            selected_boundary = self.boundary_select_var.get()
            selected_boundary = selected_boundary.split('B-')[-1]
            value = self.boundary_value_entry.get()
            self.boundary_values[int(selected_boundary)] = float(value)

            input_boundary_str = ''
            for boundary_nbr, value in self.boundary_values.items():
                input_boundary_str += f"{boundary_nbr}: {value} ;"
            self.input_boundary_str.set(input_boundary_str)



        self.set_boundaries_window = tk.Toplevel(self.root)
        self.set_boundaries_window.geometry(f"{300}x{300}")

        # Select boundary
        boundary_select_label = tk.Label(self.set_boundaries_window, text="Select Boundary", font=("Arial", 12))
        boundary_select_label.place(relx=0.1, rely=0.1)
        boundaries = [f"B-{elem}" for elem in range(len(self.polygon_coordinates) - 1)]
        self.boundary_select_var = tk.StringVar()
        self.boundary_select_var.set(boundaries[0])  # default value
        self.dropdown_boundary_menu = tk.OptionMenu(self.set_boundaries_window, self.boundary_select_var, *boundaries)
        self.dropdown_boundary_menu.place(relx=0.1, rely=0.2)

        # select boundary type Placeholder - TODO: Not yet implemented
        boundary_type_label = tk.Label(self.set_boundaries_window, text="Select Type", font=("Arial", 12))
        boundary_type_label.place(relx=0.6, rely=0.1)
        types = ['Dirichlet', 'Neumann', 'Robin']
        self.boundary_type_var = tk.StringVar()
        self.boundary_type_var.set(types[0])  # default value
        self.dropdown_boundary_type_menu = tk.OptionMenu(self.set_boundaries_window, self.boundary_type_var, *types)
        self.dropdown_boundary_type_menu.place(relx=0.6, rely=0.2)

        # Input value
        boundary_value_label = tk.Label(self.set_boundaries_window, text="Value", font=("Arial", 12))
        boundary_value_label.place(relx=0.1, rely=0.3)
        self.boundary_value_entry = tk.Entry(self.set_boundaries_window)
        self.boundary_value_entry.place(relx=0.1, rely=0.4)

        self.input_boundary_str = tk.StringVar()
        self.input_boundary_str.set('None')
        self.boundary_value_set_label = tk.Entry(self.set_boundaries_window, textvariable=self.input_boundary_str,
                                                 state='readonly', font=("Arial", 10), width=30)
        self.boundary_value_set_label.place(relx=0.1, rely=0.5)

        # Apply value
        self.set_value_button = tk.Button(self.set_boundaries_window, text="Set Value", command=set_value, font=("Arial", 12))
        self.set_value_button.place(relx=0.6, rely=0.38)

        # todo: on close
        self.boundaries_set = True




    def exit(self):
        """
        destroys mainwindow and closes programm
        :return:
        """
        self.root.destroy()

    def clear(self):
        """
        clears all values
        :return:
        """

        self.clean_canvas(self.canvas)
        self.add_grid()

        self.polygon_coordinates = [[0, 600]]
        self.firstclick_canvas = True
        self.polygon_closed = False
        self.mesh = None
        self.calcfem = None
        self.click_counter = 0
        self.boundaries_set = None
        self.all_points_numbered = None
        self.all_outline_vertices_numbered = None
        self.boundaries_numbered = None
        self.triangles = None
        self.boundary_values = dict()
        self.polygon_coordinates_str.set(str(self.polygon_coordinates[0]))
        FEMGUI.on_canvas_click.prevpoint = (0, 600)


    def start(self):
        # Start tkinter
        root = tk.Tk()
        self.root = root
        root.title("Main")
        root.geometry(f"{WINDOWS_WIDTH}x{WINDOW_HEIGHT}")

        # Canvas widget for setting nodes of polygon
        self.canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="gray")
        self.canvas.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH, rely=BORDER_WIDTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        # set initial node
        self.canvas.create_oval(0 - DOTSIZE, CANVAS_HEIGHT - DOTSIZE, 0 + DOTSIZE, CANVAS_HEIGHT + DOTSIZE,
                                outline="black", fill="white")

        # Inital values for widgets
        self.coord_var_x = tk.StringVar()
        self.coord_var_y = tk.StringVar()
        self.polygon_coordinates_str = tk.StringVar()
        self.coord_var_x.set('0')
        self.coord_var_y.set('0')
        self.polygon_coordinates_str.set(str(self.polygon_coordinates[0]))
        self.polygon_coordinates_tmp = str(self.polygon_coordinates[0]) + ' '

        # Dynamic display coordinates last click
        coord_entry_width = 6
        coord_label = tk.Label(root, text="Coordinates:", font=("Arial", 12))
        coord_label.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH,
                          rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT))
        coord_label_x = tk.Label(root, text="X:", font=("Arial", 12))
        coord_label_x.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.075,
                            rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT))
        coord_label_y = tk.Label(root, text="Y:", font=("Arial", 12))
        coord_label_y.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.15,
                            rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT))

        coord_entry_x = tk.Entry(root, textvariable=self.coord_var_x, font=("Arial", 12),
                                 width=coord_entry_width, justify='left')
        coord_entry_x.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.095,
                            rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT))
        coord_entry_y = tk.Entry(root, textvariable=self.coord_var_y, font=("Arial", 12),
                                 width=coord_entry_width, justify='left')
        coord_entry_y.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.168,
                            rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT))

        coord_set_label = tk.Label(root, text="All polygon vertices:", font=("Arial", 12))
        coord_set_label.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH,
                              rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT) + 0.05)
        polygon_coordinates = tk.Entry(root, textvariable=self.polygon_coordinates_str,
                                       state='readonly', font=("Arial", 12), width=100, justify='left')
        polygon_coordinates.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.115,
                                  rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT) + 0.05)

        # finish polyon entry of coordinates
        finish_coord_entry_button = tk.Button(root, text="Close Polygon",
                                              command=self.finish_polygon_coordinates, font=("Arial", 13))
        finish_coord_entry_button.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.775,
                                        rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT) + 0.05)

        # create mesh button
        create_mesh_button = tk.Button(root, text="Create Mesh", command=self.create_mesh, font=("Arial", 13))
        create_mesh_button.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.775,
                                 rely=2 * BORDER_WIDTH + (CANVAS_HEIGHT / WINDOW_HEIGHT) + 0.105)

        # add slider for selecting density and dropdown option menu for method
        self.density_slider = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL,
                                       label="Density", font=("Arial", 12))
        self.density_slider.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH, rely=0.85)

        methods_label = tk.Label(root, text="Mesh Method:", font=("Arial", 12))
        methods_label.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.1,
                            rely=0.85)
        methods = ['uniform', 'random', 'randomuniform']
        self.methods_dropdown_var = tk.StringVar()
        self.methods_dropdown_var.set(methods[2])  # default value
        dropdown_menu = tk.OptionMenu(root, self.methods_dropdown_var, *methods)
        dropdown_menu.place(relx=1 - (CANVAS_WIDTH / WINDOWS_WIDTH) - BORDER_WIDTH + 0.1, rely=0.90)

        # output to file
        write_file_button = tk.Button(root, text="Write Mesh to File", command=self.create_output, font=("Arial", 13))
        write_file_button.place(relx=0.4, rely=0.85)

        # FEM button
        calc_fem_button = tk.Button(root, text="Calculate FEM ", command=self.calc_fem_main, font=("Arial", 13))
        calc_fem_button.place(relx=0.55, rely=0.85)

        # Set Boundaries button
        set_boundaries_button = tk.Button(root, text="Set Boundaries", command=self.set_boundaries_window, font=("Arial", 13))
        set_boundaries_button.place(relx=0.55, rely=0.9)

        # add grid
        self.add_grid()

        # Clear button
        clear_button = tk.Button(root, text="  CLEAR  ", command=self.clear, font=("Arial", 13))
        clear_button.place(relx=0.05, rely=0.1)

        # exit button
        exit_button = tk.Button(root, text="  EXIT      ", command=self.exit, font=("Arial", 13))
        exit_button.place(relx=0.05, rely=0.05)

        # Run the Tkinter main loop
        root.mainloop()


#################################################
# Develop -> skips GUI
dev = True

def develop():
    density = 0.025 # TODO: value density = 0.1 produces divide by zero runtime warning
    method = 'uniform'
    polygon_vertices = np.array([[0, 0], [1, 0], [2, 1], [0.5, 0.5], [0, 1], [0, 0]])
    boundary_values = {1: 1}
    start_time = time.time()
    mesh = CreateMesh(polygon_vertices, density, method)
    all_points_numbered, all_outline_vertices_numbered, boundaries_numbered, triangles = mesh.create_mesh()
    mesh.show_mesh()
    calcfem = CalcFEM(all_points_numbered, all_outline_vertices_numbered, boundaries_numbered, triangles, boundary_values)
    calcfem.equation = 'HH' # WLG for heat equation, 'HH' for Helmholtz equation
    calcfem.calc_fem()
    end_time = time.time()
    runtime = end_time - start_time
    calcfem.plot_solution_dev()
    print(f"Runtime: {runtime:.4f}")

#################################################


def main():
    if not dev:
        start = FEMGUI()
        start.start()
    else:
        develop()



if __name__ == "__main__":
    main()