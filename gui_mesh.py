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
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import datetime

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


class ElementMatrice:

    def __init__(self):
        ...

    @staticmethod
    def calc_2d_triangulat_heatflow(conductivity: float, nodes: list):
        """

        :param conductivity: k
        :param nodes: [[x1, y1],[x2, y2],[x3, y3]]
        :return: np.array
        """

        x1 = nodes[0][0]
        y1 = nodes[0][1]
        x2 = nodes[1][0]
        y2 = nodes[1][1]
        x3 = nodes[2][0]
        y3 = nodes[2][1]
        k = conductivity

        val11 = -((k * (x2 - x3 - y2 + y3) ** 2) / (2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))))
        val12 = (k * (-x2 + x3 + y2 - y3) * (x1 - x3 - y1 + y3)) / (
                2 * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)))
        val13 = -((k * (x1 - x2 - y1 + y2) * (x2 - x3 - y2 + y3)) / (
                2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))))
        val21 = val12
        val22 = -((k * (x1 - x3 - y1 + y3) ** 2) / (2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))))
        val23 = (k * (x1 - x2 - y1 + y2) * (x1 - x3 - y1 + y3)) / (
                2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)))
        val31 = val13
        val32 = val23
        val33 = -((k * (x1 - x2 - y1 + y2) ** 2) / (2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))))
        kmat = np.array([[val11, val12, val13], [val21, val22, val23], [val31, val32, val33]], dtype=np.single)

        return kmat


class CalcFEM:

    def __init__(self, all_points_numbered, all_outline_vertices_numbered, boundaries_numbered, triangles, boundary_values):
        self.all_points_numbered = all_points_numbered
        self.all_outline_vertices_numbered = all_outline_vertices_numbered
        self.boundaries_numbered = boundaries_numbered
        self.triangles = triangles
        self.boundary_values = boundary_values

        self.all_points = np.array([elem[1] for elem in all_points_numbered])
        self.polygon_outline_vertices = np.array([elem[1] for elem in all_outline_vertices_numbered])
        self.k = 0.5  # todo, test
        self.zuordtab = triangles
        self.maxnode = len(self.all_points)
        self.syssteifarray = None
        self.solution = None


    def get_counter_clockwise_triangle(self, nodes: list):
        """
        rearanges the nodes, so that they are counter clockwise
        TODO: IST DAS NOTWENIG????
        :param nodes:
        :return:
        """
        x1 = nodes[0][0]
        y1 = nodes[0][1]
        x2 = nodes[1][0]
        y2 = nodes[1][1]
        new_nodes = [nodes[0]]
        if x2 < x1:
            new_nodes.append(nodes[2])
            new_nodes.append(nodes[1])
        elif x2 == x1:
            if y2 < y1:
                new_nodes.append(nodes[1])
                new_nodes.append(nodes[2])
            else:
                new_nodes.append(nodes[2])
                new_nodes.append(nodes[1])
        elif x2 > x1:
            new_nodes.append(nodes[1])
            new_nodes.append(nodes[2])
        return new_nodes

    def calc_elementmatrices(self):
        self.nbr_of_elements = len(self.triangles)

        self.all_element_matrices = np.zeros((self.nbr_of_elements, 3, 3), dtype=np.single)

        for idx, triangle in enumerate(self.triangles):
            p1, p2, p3 = triangle[0], triangle[1], triangle[2]
            x1 = self.all_points[p1][0]
            y1 = self.all_points[p1][1]
            x2 = self.all_points[p2][0]
            y2 = self.all_points[p2][1]
            x3 = self.all_points[p3][0]
            y3 = self.all_points[p3][1]
            nodes = [[x1, y1], [x2, y2], [x3, y3]]
            elemmatrix = ElementMatrice.calc_2d_triangulat_heatflow(self.k, nodes)
            self.all_element_matrices[idx] = elemmatrix

    def calc_force_vector(self):
        self.lastvektor = np.zeros(self.maxnode, dtype=np.single)
        self.lastvektor[0] = 0.00001

    def calc_system_matrices(self):
        self.syssteifarray = np.zeros((self.maxnode, self.maxnode), dtype=np.single)

        for ielem in range(self.nbr_of_elements):
            elesteifmat = self.all_element_matrices[ielem]
            for a in range(3):
                for b in range(3):
                    zta = int(self.zuordtab[ielem, a])
                    ztb = int(self.zuordtab[ielem, b])
                    self.syssteifarray[zta, ztb] = self.syssteifarray[zta, ztb] + elesteifmat[a, b]

    def implement_diriclet(self, sysmatrix, forcevector, diriclet_list):

        diriclet_list_positions = [pos[0] for pos in diriclet_list]
        diriclet_list_values = [pos[1] for pos in diriclet_list]
        original_sysmatrix = np.copy(sysmatrix)

        # sysmatrix
        for position, value in diriclet_list:
            sysmatrix[:, position] = 0
            sysmatrix[position, :] = 0
            sysmatrix[position, position] = 1

        # force vector
        forcevector = forcevector - np.dot(original_sysmatrix[:, diriclet_list_positions], diriclet_list_values)
        for pos, value in diriclet_list:
            forcevector[pos] = value

        return sysmatrix, forcevector

    def print_matrix(self, matrix):

        if not isinstance(matrix[0], np.ndarray):
            if len(matrix) < 50:
                print("[", end='')
                for idx, val in enumerate(matrix):
                    if idx < len(matrix) - 1:
                        print(f"+{abs(val):.2f}," if val >= 0 else f"-{abs(val):.2f},", end='')
                    else:
                        print(f"+{abs(val):.2f}" if val >= 0 else f"-{abs(val):.2f}", end='')
                print("]")

        else:
            if len(matrix) < 50:
                print("[", end='\n')
                for idx, elem in enumerate(matrix):
                    print("[", end='')
                    for idy, val in enumerate(elem):
                        if val == 0:
                            val_str = '_____'
                        else:
                            val_str = f"+{abs(val):.2f}" if val >= 0 else f"-{abs(val):.2f}"
                        if idy < len(elem) - 1:
                            print(f"{val_str},", end='')
                        elif idy >= len(elem) - 1 and idx < len(matrix) - 1:
                            print(f"{val_str}],", end='\n')
                        else:
                            print(f"{val_str}]", end='\n')
                print("]")

    def solve_linear_system(self):
        if self.syssteifarray is not None:  # since it might be a np.array
            self.solution = np.linalg.solve(self.syssteifarray, self.lastvektor)

    def plot_solution(self):

        solutionallnodes = np.zeros((self.maxnode, 3))
        for i in range(self.maxnode):
            solutionallnodes[i, 0] = self.all_points[i, 0]
            solutionallnodes[i, 1] = self.all_points[i, 1]
        datax = solutionallnodes[:, 0]
        datay = solutionallnodes[:, 1]
        dataz = np.real(self.solution)

        minx = min(datax)
        maxx = max(datax)
        miny = min(datay)
        maxy = max(datay)

        points = solutionallnodes[:, (0, 1)]
        values = dataz
        grid_x, grid_y = np.mgrid[minx:maxx:600j, miny:maxy:600j]
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

        # Contourplot
        nr_of_contours = 100  # Contouren insgesamt, hoher Wert für Quasi-Densityplot
        nr_of_contourlines = 5  # EIngezeichnete Contourlinien, Wert nicht exakt...
        aspectxy = 1
        ctlines = int(nr_of_contours / nr_of_contourlines)

        dataX = grid_x
        dataY = grid_y
        dataZ = grid_z1

        # fig1, ax = plt.subplots()
        # CS1 = ax.contourf(dataX, dataY, dataZ, nr_of_contours, cmap=plt.cm.gnuplot2)
        # ax.set_title('Temperature field')
        # ax.set_xlabel('x [m]')
        # ax.set_ylabel('y [m]')
        # ax.set_aspect(aspectxy)
        #
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.2)
        # cbar = fig1.colorbar(CS1, cax=cax)
        # cbar.ax.set_ylabel('Temp [T]')
        #
        # plt.scatter(self.polygon_outline_vertices[:, 0], self.polygon_outline_vertices[:, 1], c='b', marker='o',
        #             label='Boundary Points')
        # plt.scatter(self.all_points[:, 0], self.all_points[:, 1], c='b', marker='.', label='Seed Points')
        # plt.triplot(self.all_points[:, 0], self.all_points[:, 1], self.triangles, c='gray', label='Mesh')

        triang_mpl = tri.Triangulation(self.all_points[:, 0], self.all_points[:, 1], self.triangles)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Temperature field')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect(aspectxy)

        contour = ax.tricontourf(triang_mpl, values, cmap='viridis', levels=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(contour, cax=cax)
        cbar.ax.set_ylabel('Temp [T]')

        scatter = ax.scatter(self.all_points[:, 0], self.all_points[:, 1], c=values, cmap='viridis', marker='.',
                             edgecolors='w', s=10)
        triplot = ax.triplot(triang_mpl, 'w-', linewidth=0.1)

        return fig, ax

    def calc_fem(self):

        all_diriclet = []
        for boundary_number, boundary_value in self.boundary_values.items():
            this_boundary = self.boundaries_numbered[boundary_number]
            boundary_node_numbers_value = [(nbr[0], boundary_value) for nbr in this_boundary]
            all_diriclet += boundary_node_numbers_value

        self.calc_elementmatrices()
        self.calc_system_matrices()
        self.calc_force_vector()

        sysmatrix_adj, force_vector_adj = self.implement_diriclet(self.syssteifarray, self.lastvektor, all_diriclet)
        self.syssteifarray = sysmatrix_adj
        self.lastvektor = force_vector_adj

        #self.print_matrix(self.syssteifarray)
        #self.print_matrix(self.lastvektor)

        self.solve_linear_system()
        #self.plot_solution()

        return self.solution

class CreateMesh:
    """
    Creates an array of mesh points and triangulation
    """

    def __init__(self, polygon_vertices: np.array, density: float, method='uniform'):
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
        if not np.array_equal(self.polygon[0], self.polygon[-1]):
            raise ValueError(f"First vertice has to be equal to last vertice: {self.polygon[0]} != {self.polygon[-1]}")

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

        boundaries = list()
        n_point = 0
        for nv, start_point in enumerate(self.polygon[:-1]):
            end_point = self.polygon[nv + 1]
            line = np.array([start_point, end_point])
            if nv == 0:
                outline_vertices = self.create_line_vertices(line)[:-1]
                boundary_points = list()
                for vertice in outline_vertices:
                    boundary_points.append([n_point, vertice])
                    n_point += 1
                boundaries.append(boundary_points)
            else:
                this_boundary = self.create_line_vertices(line)[:-1]
                boundary_points = list()
                for vertice in this_boundary:
                    boundary_points.append([n_point, vertice])
                    n_point += 1
                boundaries.append(boundary_points)
                outline_vertices = np.append(outline_vertices, this_boundary, axis=0)

        boundaries_numbered = boundaries
        all_outline_vertices_numbered = [[n_point, outline_vertices[n_point]] for n_point in
                                         range(len(outline_vertices))]

        return all_outline_vertices_numbered, boundaries_numbered

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
        tolerance = self.density / 2
        point_on_line = False
        for nv, start_point in enumerate(self.polygon[:-1]):
            end_point = self.polygon[nv + 1]
            direction_vector = end_point - start_point
            normal_vector = np.array([-direction_vector[1], direction_vector[0]])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            third_point_new_rect = np.array(
                [(start_point[0] + tolerance * normal_vector[0]), (start_point[1] + tolerance * normal_vector[1])])
            fourth_point_new_rect = np.array(
                [(end_point[0] + tolerance * normal_vector[0]), (end_point[1] + tolerance * normal_vector[1])])
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

        all_outline_vertices_numbered, boundaries_numbered = self.create_polygon_outline_vertices()

        rect_min_max_coords = self.get_min_max_values()
        rect_seed_points = self.get_seed_rectangle(rect_min_max_coords)

        keep_points = []
        for idn, point in enumerate(rect_seed_points):
            if self.check_vertice(point):
                if not self.check_vertice_outline(point):
                    keep_points.append(idn)
        filtered_seed_points = rect_seed_points[keep_points]
        last_point_polygon_outline_vertice = all_outline_vertices_numbered[-1][0] + 1
        nbr_of_filtered_seed_points = len(filtered_seed_points)
        maxnode = last_point_polygon_outline_vertice + nbr_of_filtered_seed_points
        filtered_seed_points_numbered = [[n_point, filtered_seed_points[n_point - last_point_polygon_outline_vertice]]
                                         for n_point in range(last_point_polygon_outline_vertice, maxnode)]

        polygon_outline_vertices = [elem[1] for elem in all_outline_vertices_numbered]
        all_points = np.append(polygon_outline_vertices, filtered_seed_points, axis=0)
        all_points_numbered = all_outline_vertices_numbered + filtered_seed_points_numbered

        return all_points, all_points_numbered, all_outline_vertices_numbered, boundaries_numbered

    def show_mesh(self):
        """
        todo
        :param all_points:
        :param polygon_outline_vertices:
        :param triangles:
        :return:
        """
        polygon_outline_vertices = np.array(self.polygon_outline_vertices)
        if not self.meshcreated:
            print(f"Run create_mesh() first!")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(polygon_outline_vertices[:, 0], polygon_outline_vertices[:, 1], c='b', marker='o',
                        label='Boundary Points')
            plt.scatter(self.all_points[:, 0], self.all_points[:, 1], c='b', marker='.', label='Seed Points')
            plt.triplot(self.all_points[:, 0], self.all_points[:, 1], self.triangles, c='gray', label='Mesh')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.title('Mesh generation in Polygon')

            #plt.show()
            return fig, ax


    def output_mesh_param(self):
        """
        todo: boundary nodes and lines
        :return:
        """
        if len(self.all_points) < 1000:
            print("Node Coordinates:")
            for idp, point in enumerate(self.all_points):
                print(f"{idp}: {point}")

            print("Triangulation matrix:")
            for idt, triangle in enumerate(self.triangles):
                print(f"{idt}: {triangle}")
        else:
            print("Too many elements, printing would be unwise...")

    def write_output(self):
        """
        todo
        :return:
        """
        output_to_write = ''
        output_to_write += 'Coordinates of nodes\n'
        for idp, point in enumerate(self.all_points):
            output_to_write += f"{idp}: {point}\n"
        output_to_write += '\n\nTriangulation  matrix\n'
        for idt, triangle in enumerate(self.triangles):
            output_to_write += f"{idt}: {triangle}\n"
        with open('output.txt', 'w') as f:
            f.write(output_to_write)

    def create_mesh(self):
        """
        todo:
        :param polygon:
        :param density:
        :param method:
        :return:
        """
        all_points, all_points_numbered, all_outline_vertices_numbered, boundaries_numbered = self.get_seed_polygon()
        polygon_outline_vertices = [elem[1] for elem in all_outline_vertices_numbered]

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
        # self.output_mesh_param() # prints output

        return all_points_numbered, all_outline_vertices_numbered, boundaries_numbered, self.triangles


class FEMGUI:

    def __init__(self):
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

        start_time = datetime.datetime.now()
        self.mesh = CreateMesh(polygon_vertices, density, method)
        end_time = datetime.datetime.now()
        elased_time = end_time - start_time

        self.all_points_numbered, self.all_outline_vertices_numbered, self.boundaries_numbered, self.triangles = self.mesh.create_mesh()
        fig, ax = self.mesh.show_mesh()

        # create new window
        stats_nbr_of_nodes = len(self.all_points_numbered)
        stats_nbr_of_elements = len(self.triangles)
        stats_nbrs_of_boundaries = len(self.boundaries_numbered)
        stats_calculation_time_mesh = elased_time.total_seconds()

        stats_string = f"Stats:\n" \
                       f"Number of Nodes: {stats_nbr_of_nodes}\n" \
                       f"Number of Elements: {stats_nbr_of_elements}\n" \
                       f"Number of Boundaries: {stats_nbrs_of_boundaries}\n"

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





def main():
    density = 0.05
    start = FEMGUI()
    start.start()


if __name__ == "__main__":
    main()