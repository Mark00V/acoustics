import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy
import numpy as np
from scipy.spatial import Delaunay
import math
import random
import time
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri

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
        all_outline_vertices_numbered = [[n_point, outline_vertices[n_point]] for n_point in range(len(outline_vertices))]

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
        filtered_seed_points_numbered = [[n_point, filtered_seed_points[n_point - last_point_polygon_outline_vertice]] for n_point in range(last_point_polygon_outline_vertice, maxnode)]

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
            plt.scatter(polygon_outline_vertices[:, 0], polygon_outline_vertices[:, 1], c='b', marker='o', label='Boundary Points')
            plt.scatter(self.all_points[:, 0], self.all_points[:, 1], c='b', marker='.', label='Seed Points')
            plt.triplot(self.all_points[:, 0], self.all_points[:, 1], self.triangles, c='gray', label='Mesh')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.title('Mesh generation in Polygon')
            plt.show()


    def output_mesh_param(self):
        """

        :return:
        """
        print("Node Coordinates:")
        for idp, point in enumerate(self.all_points):
            print(f"{idp}: {point}")

        print("Triangulation matrix:")
        for idt, triangle in enumerate(self.triangles):
            print(f"{idt}: {triangle}")


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


    @staticmethod
    def calc_2d_triangulat_heatflow_new(conductivity: float, nodes: list):
        """
                :param nodes: [[x1, y1],[x2, y2],[x3, y3]]
        """

        def n1(xi1, xi2):
            return 1 - xi1 - xi2

        def n2(xi1, xi2):
            return xi1

        def n3(xi1, xi2):
            return xi2

        def ngrad1(xi1, xi2):
            return np.array([-1, -1], dtype=np.single)

        def ngrad2(xi1, xi2):
            return np.array([1, 0], dtype=np.single)

        def ngrad3(xi1, xi2):
            return np.array([0, 1], dtype=np.single)

        def gradmat(xi1, xi2, x1, x2, x3, y1, y2, y3):
            jacobi_inverse_transpose_matrix = np.array(
                [[(y1 - y3) / (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3),
                  (y1 - y2) / (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))],
                 [(x1 - x3) / (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)),
                  (x1 - x2) / (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3)]], dtype=np.single)

            ngrad = np.array([ngrad1(xi1, xi2), ngrad2(xi1, xi2),
                              ngrad3(xi1, xi2)], dtype=np.single)

            return np.transpose(np.dot(jacobi_inverse_transpose_matrix, np.transpose(ngrad)))

        x1 = nodes[0][0]
        y1 = nodes[0][1]
        x2 = nodes[1][0]
        y2 = nodes[1][1]
        x3 = nodes[2][0]
        y3 = nodes[2][1]
        k = conductivity

        intnodes = np.array([[0, 0], [1, 0], [0, 1]])
        intweights = np.array([1 / 6, 1 / 6, 1 / 6])

        jacobi_det = -x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3

        elesteifmat = np.zeros((3, 3), dtype=np.single)
        for i in range(3):
            xi1 = intnodes[i, 0]
            xi2 = intnodes[i, 1]
            gr = gradmat(xi1, xi2, x1, x2, x3, y1, y2, y3)
            grt = np.transpose(gr)
            grxgrt = gr @ grt
            fp = grxgrt * jacobi_det * intweights
            elesteifmat = elesteifmat + fp

        return elesteifmat


    @staticmethod
    def calc_2d_triangulat_acoustic(nodes: list):
        """
                :param nodes: [[x1, y1],[x2, y2],[x3, y3]]
        """

        def n1(xi1, xi2):
            return 1 - xi1 - xi2

        def n2(xi1, xi2):
            return xi1

        def n3(xi1, xi2):
            return xi2

        def ngrad1(xi1, xi2):
            return np.array([-1, -1], dtype=np.single)

        def ngrad2(xi1, xi2):
            return np.array([1, 0], dtype=np.single)

        def ngrad3(xi1, xi2):
            return np.array([0, 1], dtype=np.single)

        def gradmat(xi1, xi2, x1, x2, x3, y1, y2, y3):
            jacobi_inverse_transpose_matrix = np.array(
                [[(y1 - y3) / (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3),
                  (y1 - y2) / (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))],
                 [(x1 - x3) / (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)),
                  (x1 - x2) / (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3)]], dtype=np.single)

            ngrad = np.array([ngrad1(xi1, xi2), ngrad2(xi1, xi2),
                              ngrad3(xi1, xi2)], dtype=np.single)

            return np.transpose(np.dot(jacobi_inverse_transpose_matrix, np.transpose(ngrad)))

        def phiqequdistarray(xi1, xi2):
            f = np.array([[n1(xi1, xi2)], [n2(xi1, xi2)], [n3(xi1, xi2)]], dtype=np.single)
            return f

        x1 = nodes[0][0]
        y1 = nodes[0][1]
        x2 = nodes[1][0]
        y2 = nodes[1][1]
        x3 = nodes[2][0]
        y3 = nodes[2][1]

        intnodes = np.array([[0, 0], [1, 0], [0, 1]])
        intweights = np.array([1 / 6, 1 / 6, 1 / 6])

        jacobi_det = -x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3

        elesteifmat = np.zeros((3, 3), dtype=np.single)
        for i in range(3):
            xi1 = intnodes[i, 0]
            xi2 = intnodes[i, 1]
            gr = gradmat(xi1, xi2, x1, x2, x3, y1, y2, y3)
            grt = np.transpose(gr)
            grxgrt = gr @ grt
            fp = grxgrt * jacobi_det * intweights
            elesteifmat = elesteifmat + fp

        elemassenmat = np.zeros((3, 3), dtype=np.single)
        for i in range(3):
            xi1 = intnodes[i, 0]
            xi2 = intnodes[i, 1]
            phi = phiqequdistarray(xi1, xi2)
            phit = np.transpose(phiqequdistarray(xi1, xi2))
            phixphit = phi @ phit
            fp = phixphit * jacobi_det * intweights
            elemassenmat = elemassenmat + fp

        return elesteifmat, elemassenmat


class CalcFEM:

    def __init__(self, all_points_numbered, all_outline_vertices_numbered, boundaries_numbered, triangles):
        self.all_points_numbered = all_points_numbered
        self.all_outline_vertices_numbered = all_outline_vertices_numbered
        self.boundaries_numbered = boundaries_numbered
        self.triangles = triangles
        self.all_points = np.array([elem[1] for elem in all_points_numbered])
        self.polygon_outline_vertices = np.array([elem[1] for elem in all_outline_vertices_numbered])
        self.k = 0.5 # todo, test
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

        self.all_element_matrices_steif = np.zeros((self.nbr_of_elements, 3, 3), dtype=np.single)
        self.all_element_matrices_mass = np.zeros((self.nbr_of_elements, 3, 3), dtype=np.single)

        for idx, triangle in enumerate(self.triangles):
            p1, p2, p3 = triangle[0], triangle[1], triangle[2]
            x1 = self.all_points[p1][0]
            y1 = self.all_points[p1][1]
            x2 = self.all_points[p2][0]
            y2 = self.all_points[p2][1]
            x3 = self.all_points[p3][0]
            y3 = self.all_points[p3][1]
            nodes = [[x1, y1], [x2, y2], [x3, y3]]
            elemsteif, elemmass = ElementMatrice.calc_2d_triangulat_acoustic(nodes)
            self.all_element_matrices_steif[idx] = elemsteif
            self.all_element_matrices_mass[idx] = elemmass


    def calc_force_vector(self):
        self.lastvektor = np.zeros(self.maxnode, dtype=np.single)
        self.lastvektor[0] = 0.00001

    def calc_system_matrices(self):
        self.syssteifarray = np.zeros((self.maxnode, self.maxnode), dtype=np.single)
        self.sysmassarray = np.zeros((self.maxnode, self.maxnode), dtype=np.single)
        self.sysarray = np.zeros((self.maxnode, self.maxnode), dtype=np.single)

        for ielem in range(self.nbr_of_elements):
            elesteifmat = self.all_element_matrices_steif[ielem]
            elemassmat = self.all_element_matrices_mass[ielem]
            for a in range(3):
                for b in range(3):
                    zta = int(self.zuordtab[ielem, a])
                    ztb = int(self.zuordtab[ielem, b])
                    self.syssteifarray[zta, ztb] = self.syssteifarray[zta, ztb] + elesteifmat[a, b]
                    self.sysmassarray[zta, ztb] = self.sysmassarray[zta, ztb] + elemassmat[a, b]
        freq = 1
        omega = freq * math.pi * 2
        self.sysarray = self.syssteifarray - omega**2 * self.sysmassarray


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

        if not isinstance(matrix[0], numpy.ndarray):
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
                        elif idy >= len(elem) - 1 and idx < len(matrix) -1:
                            print(f"{val_str}],", end='\n')
                        else:
                            print(f"{val_str}]", end='\n')
                print("]")

    def solve_linear_system(self):
        if self.syssteifarray is not None: # since it might be a np.array
            self.solution = np.linalg.solve(self.sysarray, self.lastvektor)



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

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Pressure Field')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect(aspectxy)


        contour = ax.tricontourf(triang_mpl, values, cmap='viridis', levels=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(contour, cax=cax)
        cbar.ax.set_ylabel('P [pa]')

        #scatter = ax.scatter(self.all_points[:, 0], self.all_points[:, 1], c=values, cmap='viridis', marker='.', edgecolors='w', s=10)
        triplot = ax.triplot(triang_mpl, 'w-', linewidth=0.1)

        plt.show()

    def calc_fem(self):

        boundary0 = self.boundaries_numbered[0]
        boundary1 = self.boundaries_numbered[1]
        boundary2 = self.boundaries_numbered[2]
        boundary3 = self.boundaries_numbered[3]
        boundary4 = self.boundaries_numbered[4]
        value_boundary0 = 0
        value_boundary1 = 1
        value_boundary2 = 2
        value_boundary3 = 3
        value_boundary4 = 4
        boundary0_nodes = [(nbr[0], value_boundary0) for nbr in boundary0]
        boundary1_nodes = [(nbr[0], value_boundary1) for nbr in boundary1]
        boundary2_nodes = [(nbr[0], value_boundary2) for nbr in boundary2]
        boundary3_nodes = [(nbr[0], value_boundary3) for nbr in boundary3]
        boundary4_nodes = [(nbr[0], value_boundary4) for nbr in boundary4]

        all_diriclet = boundary1_nodes

        self.calc_elementmatrices()
        self.calc_system_matrices()
        self.calc_force_vector()

        sysmatrix_adj, force_vector_adj = self.implement_diriclet(self.sysarray, self.lastvektor, all_diriclet)
        self.sysarray = sysmatrix_adj
        self.lastvektor = force_vector_adj

        self.print_matrix(self.syssteifarray)
        self.print_matrix(self.lastvektor)

        self.solve_linear_system()
        self.plot_solution()



def main():
    density = 0.025
    polygon_vertices = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1], [0.5, 0.5], [0, 0.5], [0, 0]])
    polygon_vertices = np.array([[0, 0], [1, 0], [2, 1], [0.5, 0.5], [0, 1], [0, 0]])
    # start_time = time.time()
    # all_points, polygon_outline_vertices, triangles_filtered = create_mesh(polygon_vertices, density, method='randomuniform')
    # print(f"Nbr of vertices: {len(all_points)}")
    # end_time = time.time()
    # runtime = end_time - start_time
    # print(f"Function runtime: {runtime} seconds")
    # show_mesh(all_points, polygon_outline_vertices, triangles_filtered)
    method = 'uniform'
    tst = CreateMesh(polygon_vertices, density, method)
    all_points_numbered, all_outline_vertices_numbered, boundaries_numbered, triangles = tst.create_mesh()
    calcfem = CalcFEM(all_points_numbered, all_outline_vertices_numbered, boundaries_numbered, triangles)
    calcfem.calc_fem()
    #tst.show_mesh()




if __name__ == "__main__":
    main()