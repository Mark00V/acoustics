import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
from scipy.spatial import Delaunay
import math
import random
import time
from scipy.spatial import ConvexHull
import tkinter as tk
import math

#################################################
# tkinter static parameters
# window
WINDOW_HEIGHT = 800
WINDOWS_WIDTH = 1400
# canvas
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 600
DOTSIZE = 6    # click points in canvas
LINEWIDTH = 2
GRIDSPACE = 25 # space between grid
# Borderwidth (space between window frame and widgets and widgets between widgets, relative to windowsize)
BORDER_WIDTH = 0.01
#################################################

class CreateMesh:
    """
    Creates an array of mesh points and triangulation
    """


    def __init__(self,polygon_vertices: np.array, density: float, method='uniform'):
        """
        Initializes a Polygon object with the given polygon vertices and density.

        Args:
        :param density: float, specifies distance between 2 points, floor is applied so line = [[0,0],[1,0]], density = 0.4
        returns 3 points instead of 4!
        :param polygon: np.array([
                            [x_coord0, y_coord0],
                            [x_coord1, y_coord1],
                            [x_coord2, y_coord2],
                            ...,
                            [x_coord0, y_coord0]]) # Last vertice has to be first vertice!
        :param method: optional, 'random' for random generation, 'uniform' for equal distance or 'randomuniform' for approx uniform

        Raises:
            ValueError
        Returns:
            None
        """
        self.polygon = polygon_vertices
        self.density = density
        self.method = method
        self.all_points = None
        self.polygon_outline_vertices = None
        self.triangles = None
        self.meshcreated = False
        #if not np.array_equal(self.polygon[0], self.polygon[-1]):
        #    raise ValueError(f"First vertice has to be equal to last vertice: {self.polygon[0]} != {self.polygon[-1]}")


    def check_vertice(self, point: np.array) -> bool:
        """
        Checks if a given point is inside self.polygon:
        returns True if inside polygon, returns False if outside polygon

        Args:
        :param point: np.array([x_coord0, y_coord0])
        :return: bool
        """
        polygon_path = mpath.Path(self.polygon)
        is_inside = polygon_path.contains_point(point)

        return is_inside


    def check_vertice_polygon(self, point: np.array, polygon: np.array) -> bool:
        """
        Checks if a given point is inside the given polygon:
        returns True if inside polygon, returns False if outside polygon

        Args:
        :param point: np.array([x_coord0, y_coord0])
        :return: bool
        """
        polygon_path = mpath.Path(polygon)
        is_inside = polygon_path.contains_point(point)

        return is_inside


    def create_line_vertices(self, line: np.array) -> np.array:
        """
        Creates points on a line specified by line. The number of points is given by density which specifies the average
        distance between 2 points (floored!)
        :param line: np.array([x_coord0, y_coord0], [x_coord1, y_coord1])
        :return: np.array
        """
        start_point = line[0]
        end_point = line[1]
        distance = np.linalg.norm(start_point - end_point)
        num_subdivisions = math.floor(distance / self.density) + 1
        subdivision_points = np.linspace(start_point, end_point, num_subdivisions)

        return subdivision_points

    def create_polygon_outline_vertices(self) -> np.array:
        """
        Creates an array of points with average distance density
        Args:
        Nothing
        :return: np.array
        """
        if not np.array_equal(self.polygon[0], self.polygon[-1]):
            raise ValueError(f"First vertice has to be equal to last vertice: {self.polygon[0]} != {self.polygon[-1]}")
        outline_vertices = None
        for nv, start_point in enumerate(self.polygon[:-1]):
            end_point = self.polygon[nv + 1]
            line = np.array([start_point, end_point])
            if nv == 0:
                outline_vertices = self.create_line_vertices(line)[:-1]
            else:
                outline_vertices = np.append(outline_vertices, self.create_line_vertices(line)[:-1], axis=0)

        return outline_vertices

    def get_min_max_values(self) -> np.array:
        """
        Returns the min and max x,y values for a polygon (outline positions for rectangle)
        :param polygon: np.array([
                            [x_coord0, y_coord0],
                            [x_coord1, y_coord1],
                            [x_coord2, y_coord2],
                            ...,
                            [x_coord0, y_coord0]]) # Last vertice has to be first vertice!
        :return: np.array([[min_x, min_y], [max_x, max_y]])
        """
        x_values = self.polygon[:, 0]
        y_values = self.polygon[:, 1]
        min_x, max_x = np.min(x_values), np.max(x_values)
        min_y, max_y = np.min(y_values), np.max(y_values)

        return np.array([[min_x, min_y], [max_x, max_y]])

    def get_seed_rectangle(self, rect: np.array) -> np.array:
        """
        Creates random points in the boundaries of rect  and average distance density
        between points
        :param rect: np.array([[min_x, min_y], [max_x, max_y]])
        :return: np.array([[x0, y0], [x1, y1], ... ])
        """
        if np.any(rect < 0.0):
            raise ValueError(f"All vertices have to be positive!")
        if not np.array_equal(rect[0], np.array([0, 0])):
            raise ValueError(f"Starting vortex has to be [0, 0]")

        # randomness, should be close to 1.0
        rd = 0.99
        ru = 1.01

        rect_size_x = np.linalg.norm(rect[1][0] - rect[0][0])
        rect_size_y = np.linalg.norm(rect[1][1] - rect[0][1])
        nbr_points_x = math.floor(rect_size_x / self.density) + 1
        nbr_points_y = math.floor(rect_size_y / self.density) + 1

        if self.method == 'random':
            nbr_points = nbr_points_x * nbr_points_y
            np.random.seed()
            rect_seed_points = np.random.rand(nbr_points, 2)
            rect_seed_points[:, 0] = rect_seed_points[:, 0] * rect_size_x
            rect_seed_points[:, 1] = rect_seed_points[:, 1] * rect_size_y
        elif self.method == 'uniform':
            x_points = np.linspace(rect[0][0], rect[1][0], nbr_points_x)
            y_points = np.linspace(rect[0][1], rect[1][1], nbr_points_y)
            x_grid, y_grid = np.meshgrid(x_points, y_points)
            rect_seed_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        elif self.method == 'randomuniform':
            x_points = np.linspace(rect[0][0], rect[1][0], nbr_points_x - 2)
            y_points = np.linspace(rect[0][1], rect[1][1], nbr_points_y - 2)
            x_grid, y_grid = np.meshgrid(x_points, y_points)
            rect_seed_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
            rect_seed_points = np.array(
                [np.array([point[0] * random.uniform(rd, ru), point[1] * random.uniform(rd, ru)])
                 for point in rect_seed_points])

        return rect_seed_points


    def draw_quadrilateral(self, corners: np.array):
        """
        For testing purposes
        :return:
        """
        plt.scatter(corners[:, 0], corners[:, 1], c='b', marker='x', label='Corners')
        plt.show()


    def check_vertice_outline(self, point: np.array) -> bool:
        # TODO: if in proximity (tolerance) of corner point of polygon
        """
        Checks if a point is on outline of polyon
        :param point: np.array([x_coord0, y_coord0])
        :return: bool
        """
        tolerance = self.density/2
        point_on_line = False
        for nv, start_point in enumerate(self.polygon[:-1]):
            end_point = self.polygon[nv + 1]
            direction_vector = end_point - start_point
            normal_vector = np.array([-direction_vector[1], direction_vector[0]])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            third_point_new_rect = np.array([(start_point[0] + tolerance * normal_vector[0]), (start_point[1] + tolerance * normal_vector[1])])
            fourth_point_new_rect = np.array([(end_point[0] + tolerance * normal_vector[0]), (end_point[1] + tolerance * normal_vector[1])])
            check_polygon = np.array([start_point, end_point, fourth_point_new_rect, third_point_new_rect, start_point])
            is_on_line = self.check_vertice_polygon(point, check_polygon)
            if is_on_line:
                point_on_line = True
                break

        return point_on_line

    def get_seed_polygon(self):
        """
        Creates points inside polygon and on polygon outline
        Args:
        Nothing
        :return: np.array
        """
        rect_min_max_coords = self.get_min_max_values()
        rect_seed_points = self.get_seed_rectangle(rect_min_max_coords)

        keep_points = []
        for idn, point in enumerate(rect_seed_points):
            if self.check_vertice(point):
                if not self.check_vertice_outline(point):
                    keep_points.append(idn)
        filtered_seed_points = rect_seed_points[keep_points]
        polygon_outline_vertices = self.create_polygon_outline_vertices()
        all_points = np.append(filtered_seed_points, polygon_outline_vertices, axis=0)

        return all_points

    def show_mesh(self):
        """
        todo
        :param all_points:
        :param polygon_outline_vertices:
        :param triangles:
        :return:
        """
        if not self.meshcreated:
            print(f"Run create_mesh() first!")
        else:
            plt.scatter(self.polygon_outline_vertices[:, 0], self.polygon_outline_vertices[:, 1], c='b', marker='o',
                        label='Boundary Points')
            plt.scatter(self.all_points[:, 0], self.all_points[:, 1], c='b', marker='.', label='Seed Points')
            plt.triplot(self.all_points[:, 0], self.all_points[:, 1], self.triangles, c='gray', label='Mesh')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.title('Mesh generation in Polygon')
            plt.show()


    def create_mesh(self):
        """
        todo:
        :param polygon:
        :param density:
        :param method:
        :return:
        """

        all_points = self.get_seed_polygon()

        # remove duplicates
        all_points = np.unique(all_points, axis=0)

        polygon_outline_vertices = self.create_polygon_outline_vertices()

        # Triangulation
        triangulation = Delaunay(all_points)
        triangles = triangulation.simplices

        # Remove triangulation outside of polygon
        keep_triangles = []
        for idt, triangle in enumerate(triangles):
            triangle_points = np.array([[all_points[triangle[0]][0], all_points[triangle[0]][1]],
                                        [all_points[triangle[1]][0], all_points[triangle[1]][1]],
                                        [all_points[triangle[2]][0], all_points[triangle[2]][1]]])
            center_point = np.mean(triangle_points, axis=0)
            if self.check_vertice(center_point):
                keep_triangles.append(idt)
        triangles_filtered = triangles[keep_triangles]

        self.all_points = all_points
        self.polygon_outline_vertices = polygon_outline_vertices
        self.triangles = triangles_filtered
        self.meshcreated = True

        return all_points, polygon_outline_vertices, self.triangles


class FEMGUI:

    def __init__(self):
        self.polygon_coordinates = [[0, 600]]
        self.firstclick_canvas = True
        self.polygon_closed = False

    def finish_polygon_coordinates(self):
        self.polygon_closed = True
        self.polygon_coordinates.append([0, 600])
        self.canvas.create_line(self.polygon_coordinates[-2][0], self.polygon_coordinates[-2][1],
                                self.polygon_coordinates[-1][0], self.polygon_coordinates[-1][1],
                                fill="black", width=LINEWIDTH)
        print(self.polygon_coordinates)

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
        scale_factor = 0.01 # todo
        polygon_coordinates_transformed = [[scale_factor * elem[0],-1 * scale_factor * (elem[1] - 600)] for elem in self.polygon_coordinates]
        polygon_coordinates_transformed = [[elem[0], 0 if elem[1] == -0 else elem[1]] for elem in polygon_coordinates_transformed]
        return polygon_coordinates_transformed


    def create_mesh(self):
        density = 0.1 # todo, for testing
        method = 'randomuniform' # todo, for testing
        polygon_coordinates_transformed = self.coord_transform()
        polygon_vertices = np.array(polygon_coordinates_transformed)
        mesh = CreateMesh(polygon_vertices, density, method)
        mesh.create_mesh()
        mesh.show_mesh()




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
            self.canvas.create_line(FEMGUI.on_canvas_click.prevpoint[0], FEMGUI.on_canvas_click.prevpoint[1], x, y, fill="black", width=LINEWIDTH)
        FEMGUI.on_canvas_click.prevpoint = (x,y) # TODO: Why does this not work with self????

        # Clear content
        self.coord_var_x.set('0')
        self.coord_var_y.set('600')

        # Display coordinates
        self.coord_var_x.set(f"{x}")
        self.coord_var_y.set(f"{y}")

        self.canvas.create_oval(x - DOTSIZE, y - DOTSIZE, x + DOTSIZE, y + DOTSIZE, outline="black", fill="white")


    def start(self):
        # Start tkinter
        root = tk.Tk()
        root.title("Main")
        root.geometry(f"{WINDOWS_WIDTH}x{WINDOW_HEIGHT}")

        # Canvas widget for setting nodes of polygon
        self.canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="gray")
        self.canvas.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH, rely=BORDER_WIDTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        # set initial node
        self.canvas.create_oval(0 - DOTSIZE, CANVAS_HEIGHT - DOTSIZE, 0 + DOTSIZE, CANVAS_HEIGHT + DOTSIZE, outline="black", fill="white")

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
        coord_label.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))
        coord_label_x = tk.Label(root, text="X:", font=("Arial", 12))
        coord_label_x.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.075, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))
        coord_label_y = tk.Label(root, text="Y:", font=("Arial", 12))
        coord_label_y.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.15, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))

        coord_entry_x = tk.Entry(root, textvariable=self.coord_var_x, font=("Arial", 12), width=coord_entry_width, justify='left')
        coord_entry_x.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.095, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))
        coord_entry_y = tk.Entry(root, textvariable=self.coord_var_y, font=("Arial", 12), width=coord_entry_width, justify='left')
        coord_entry_y.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.168, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))

        coord_set_label = tk.Label(root, text="All polygon vertices:", font=("Arial", 12))
        coord_set_label.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT)+0.05)
        polygon_coordinates = tk.Entry(root, textvariable=self.polygon_coordinates_str, state='readonly', font=("Arial", 12), width=100, justify='left')
        polygon_coordinates.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.115, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT)+0.05)

        # finish polyon entry of coordinates
        finish_coord_entry_button = tk.Button(root, text="Close Polygon", command=self.finish_polygon_coordinates, font=("Arial", 13))
        finish_coord_entry_button.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.775, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT)+0.05)

        # create mesh
        create_mesh_button = tk.Button(root, text="Create Mesh", command=self.create_mesh, font=("Arial", 13))
        create_mesh_button.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.775, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT)+0.105)

        # add grid
        self.add_grid()


        # Run the Tkinter main loop
        root.mainloop()



def main():
    density = 0.05
    start = FEMGUI()
    start.start()






if __name__ == "__main__":
    main()